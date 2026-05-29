"""
executor.py — All Alpaca API interactions: placing orders, confirming fills,
placing hard stops, and submitting exits.

Order flow for entries:
  1. 9:25 AM → submit_entry() places a Market OPG (on open) buy order
  2. 9:45 AM → confirm_fills_and_place_stops() checks if filled, then places
               a GTC hard stop order on the exchange using the ACTUAL fill price

The hard stop sits on Alpaca's servers — it executes even if this bot crashes.
"""
import logging
import time
from datetime import date, datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce

import config
import position_tracker as pt
import risk
import trade_log

logger = logging.getLogger(__name__)

_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=config.PAPER)


# ── Account ───────────────────────────────────────────────────────────────────

def get_equity() -> float:
    account = _client.get_account()
    return float(account.equity)


# ── Market Calendar ───────────────────────────────────────────────────────────
# Alpaca's calendar API is the single source of truth for which days the US
# equities market is open. It already accounts for weekends, federal holidays,
# and early-close days (e.g. day after Thanksgiving). We use it to skip the
# scheduled jobs on non-trading days instead of trying to maintain our own
# holiday table.

def is_trading_day_today() -> bool:
    """Return True if today has (or had) a regular US equities trading session."""
    today = date.today()
    # Calendar request for a one-day window — empty list means no session.
    sessions = _client.get_calendar(
        GetCalendarRequest(start=today, end=today)
    )
    return len(sessions) > 0


# ── Position Checks ───────────────────────────────────────────────────────────

def get_alpaca_positions() -> dict:
    """Returns {symbol: position_object} for all open Alpaca positions."""
    return {p.symbol: p for p in _client.get_all_positions()}


def has_position(symbol: str) -> bool:
    return symbol in get_alpaca_positions()


# ── Entry ─────────────────────────────────────────────────────────────────────

def submit_entry(scan_result: dict) -> bool:
    """
    Submit a Market OPG (next-day open) buy order for a given entry signal.

    Position size is estimated using yesterday's close price + active stop.
    The actual hard stop is placed after fill confirmation in confirm_fills_and_place_stops().

    Args:
        scan_result: Dict returned by scanner.evaluate_symbol() with entry_signal=True.

    Returns:
        True if order was successfully submitted, False otherwise.
    """
    symbol      = scan_result["symbol"]
    est_entry   = scan_result["close"]       # Yesterday's close — size estimate only
    active_stop = scan_result["active_stop"]
    stop_a      = scan_result["stop_a"]
    stop_b      = scan_result["stop_b"]
    atr_val     = scan_result["atr14"]

    if has_position(symbol):
        logger.info(f"Skipping {symbol} entry — position already open.")
        return False

    equity = get_equity()
    shares = risk.calculate_position_size(equity, est_entry, active_stop)

    if shares <= 0:
        logger.warning(f"Skipping {symbol} — calculated 0 shares.")
        return False

    try:
        limit_price = round(est_entry * (1 + config.ENTRY_LIMIT_PCT), 2)
        order = _client.submit_order(
            LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.OPG,  # LOO = Limit on Open
                limit_price=limit_price,
            )
        )

        logger.info(
            f"📥 BUY ORDER submitted | {symbol} | {shares} shares | "
            f"Est. entry=${est_entry:.2f} | Limit=${limit_price:.2f} | Order ID={order.id}"
        )

        # Record position with estimated values — actual fill updates later
        pt.add(
            symbol=symbol,
            entry_price=est_entry,
            stop_price=active_stop,
            stop_price_a=stop_a,
            stop_price_b=stop_b,
            shares=shares,
            stop_mult=config.ACTIVE_STOP_MULT,
            stop_order_id=None,  # Set after fill confirmation
            rsi2=scan_result.get("rsi2"),
            sma200=scan_result.get("sma200"),
            ema50=scan_result.get("ema50"),
            atr14=scan_result.get("atr14"),
        )
        return True

    except Exception as e:
        logger.error(f"Entry order failed for {symbol}: {e}", exc_info=True)
        return False


# ── Fill Confirmation + Hard Stop Placement ───────────────────────────────────

def confirm_fills_and_place_stops():
    """
    Run ~15 minutes after open to:
      1. Confirm buy orders have been filled.
      2. Calculate stop price from ACTUAL fill price (not estimated).
      3. Place a GTC hard stop order on the exchange.

    This is the critical step — the hard stop is what protects the position
    even if the bot crashes or loses connectivity.
    """
    logger.info("Confirming fills and placing hard stops...")

    tracked = pt.all_positions()
    alpaca_positions = get_alpaca_positions()

    for symbol, pos_data in tracked.items():
        if pos_data.get("stop_order_id"):
            logger.info(f"{symbol}: Stop already placed (ID={pos_data['stop_order_id']}), skipping.")
            continue

        if symbol not in alpaca_positions:
            # Check whether an OPG buy order is still open (submitted but not yet
            # processed by the exchange). If one is still live, don't interfere.
            try:
                open_orders = _client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                )
                open_buys = [o for o in open_orders if str(o.side).lower() == "buy"]
            except Exception as e:
                logger.error(f"Could not check open orders for {symbol}: {e}", exc_info=True)
                open_buys = []

            if open_buys:
                logger.info(
                    f"{symbol}: OPG order still pending (ID={open_buys[0].id}). "
                    f"Skipping fallback — will retry at next confirm."
                )
                continue

            # LOO entry expired unfilled — do not chase with a market order.
            # The stock opened above our limit; entering now would be buying into
            # a gap-up, which is the opposite of what this strategy wants.
            logger.warning(
                f"{symbol}: LOO entry expired unfilled — removing tracker record, will not chase."
            )
            pt.remove(symbol)
            continue

        alpaca_pos  = alpaca_positions[symbol]
        fill_price  = float(alpaca_pos.avg_entry_price)
        shares      = int(float(alpaca_pos.qty))
        atr_val     = (fill_price - pos_data["stop_price"]) / config.ACTIVE_STOP_MULT
        actual_stop = round(fill_price - (config.ACTIVE_STOP_MULT * atr_val), 2)

        # Update tracker with actual fill price
        pt.update_entry_price(symbol, fill_price)

        # Place the hard stop on the exchange
        stop_order_id = _place_stop_order(symbol, actual_stop, shares)

        if stop_order_id:
            pt.update_stop_order_id(symbol, stop_order_id)
            logger.info(
                f"✅ Fill confirmed + stop placed | {symbol} | "
                f"Fill=${fill_price:.2f} | Stop=${actual_stop:.2f} | "
                f"Stop Order ID={stop_order_id}"
            )


def confirm_exit_fills():
    """
    Run ~15 minutes after open alongside confirm_fills_and_place_stops().

    Iterates all exit_pending positions from the tracker and resolves each:
      - Position gone from Alpaca  → fill confirmed, remove from tracker.
      - Open sell order still live → order in flight, leave as exit_pending.
      - Position still held, no open sell → LOO expired unfilled; submit a
        DAY market sell as fallback so the position is guaranteed to close.

    Driving off the tracker (not order timestamps) means missed exits from
    prior days are caught on the next scan cycle, not silently abandoned.
    """
    logger.info("Checking exit_pending positions for fill confirmation...")

    exit_pending = pt.get_exit_pending_symbols()
    if not exit_pending:
        logger.info("No exit_pending positions to confirm.")
        return

    alpaca_positions = get_alpaca_positions()

    for symbol in exit_pending:
        if symbol not in alpaca_positions:
            # Position is gone from Alpaca — fill confirmed.
            pos_data   = pt.get(symbol)
            exit_price = get_last_fill_price(symbol, side="sell")
            log_reason = (pos_data.get("exit_reason", "signal_exit") if pos_data else "signal_exit")
            if pos_data and exit_price:
                trade_log.log_closed_trade(pos_data, exit_price, log_reason)
            logger.info(f"{symbol}: Exit fill confirmed. Removing from tracker.")
            pt.remove(symbol)
            continue

        # Position still held — check for an open sell order.
        try:
            open_orders = _client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            open_sells = [o for o in open_orders if str(o.side).lower() == "sell"]
        except Exception as e:
            logger.error(f"Could not check open orders for {symbol}: {e}", exc_info=True)
            open_sells = []

        if open_sells:
            logger.info(
                f"{symbol}: Exit order still open (ID={open_sells[0].id}). "
                f"Leaving as exit_pending."
            )
            continue

        # No open sell and position still exists — LOO expired unfilled.
        shares = int(float(alpaca_positions[symbol].qty))
        logger.warning(
            f"{symbol}: LOO exit expired unfilled. "
            f"Submitting fallback DAY market sell for {shares} shares..."
        )
        try:
            pos_data = pt.get(symbol)
            fallback = _client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info(
                f"🔄 FALLBACK SELL submitted | {symbol} | {shares} shares | "
                f"Order ID={fallback.id}"
            )
            # Log trade using estimated fill = last Alpaca position price.
            # The actual fill will be close to market price at time of submission.
            if pos_data:
                fallback_price = float(alpaca_positions[symbol].current_price or alpaca_positions[symbol].avg_entry_price)
                log_reason = pos_data.get("exit_reason", "signal_exit") + "_fallback"
                trade_log.log_closed_trade(pos_data, fallback_price, log_reason)
            pt.remove(symbol)
        except Exception as e:
            logger.error(f"Fallback exit order failed for {symbol}: {e}", exc_info=True)


def get_last_fill_price(symbol: str, side: str) -> float | None:
    """
    Return the filled_avg_price of the most recent filled order for symbol on
    the given side ("buy" or "sell"). Returns None if no filled order is found.
    Used to retrieve exit prices for trade logging.
    """
    try:
        orders = _client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=10,
            )
        )
        for order in orders:
            if (str(order.side).lower() == side.lower()
                    and order.filled_at
                    and order.filled_avg_price):
                return float(order.filled_avg_price)
    except Exception as e:
        logger.error(f"Could not fetch fill price for {symbol} ({side}): {e}", exc_info=True)
    return None


def _place_stop_order(symbol: str, stop_price: float, shares: int) -> str | None:
    """
    Place a GTC hard stop (sell) order on the exchange.
    This order lives on Alpaca's servers — it will trigger even if the bot is down.
    """
    try:
        order = _client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
            )
        )
        logger.info(f"🛑 STOP ORDER placed | {symbol} | ${stop_price:.2f} | Order ID={order.id}")
        return str(order.id)

    except Exception as e:
        logger.error(f"Failed to place stop order for {symbol}: {e}", exc_info=True)
        return None


# ── Exit ──────────────────────────────────────────────────────────────────────

def submit_exit(symbol: str, reason: str = "signal", close_price: float = 0.0) -> bool:
    """
    Submit a LOO (Limit on Open) sell order for the next day's open.
    Cancels any open stop orders first to avoid a double-sell.
    Marks the position as exit_pending instead of removing it immediately —
    confirm_exit_fills() removes it once the fill is confirmed.

    Args:
        symbol:      Ticker to exit.
        reason:      Human-readable exit reason for logging (e.g. 'rsi_exit', 'time_stop').
        close_price: Prior session close used to set the LOO limit price.
                     Falls back to market OPG if not provided.

    Returns:
        True if exit order submitted, False otherwise.
    """
    if not has_position(symbol):
        logger.info(f"No open position in {symbol} — nothing to exit.")
        return False

    # Cancel the hard stop FIRST to prevent a duplicate exit
    _cancel_stop_for_symbol(symbol)

    alpaca_positions = get_alpaca_positions()
    shares = int(float(alpaca_positions[symbol].qty))

    try:
        limit_price = round(close_price * (1 - config.EXIT_LIMIT_PCT), 2) if close_price > 0 else None

        if limit_price:
            order = _client.submit_order(
                LimitOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.OPG,  # LOO = Limit on Open
                    limit_price=limit_price,
                )
            )
            logger.info(
                f"📤 SELL ORDER submitted | {symbol} | {shares} shares | "
                f"Reason={reason} | Limit=${limit_price:.2f} | Order ID={order.id}"
            )
        else:
            # No close price available — fall back to market OPG
            logger.warning(f"{symbol}: No close price for limit exit — submitting market OPG sell.")
            order = _client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.OPG,
                )
            )
            logger.info(
                f"📤 SELL ORDER submitted | {symbol} | {shares} shares | "
                f"Reason={reason} (market fallback) | Order ID={order.id}"
            )

        # Mark as pending — do NOT remove yet. confirm_exit_fills() removes after fill.
        pt.mark_exit_pending(symbol, reason=reason)
        return True

    except Exception as e:
        logger.error(f"Exit order failed for {symbol}: {e}", exc_info=True)
        return False


def _cancel_stop_for_symbol(symbol: str):
    """Cancel all open stop orders for a symbol before submitting a market exit."""
    stop_order_id = pt.get_stop_order_id(symbol)

    if stop_order_id:
        try:
            _client.cancel_order_by_id(stop_order_id)
            logger.info(f"Cancelled stop order {stop_order_id} for {symbol}")
            return
        except Exception as e:
            logger.warning(f"Could not cancel stop by ID for {symbol}: {e}. Falling back to full scan.")

    # Fallback: scan all open orders for this symbol
    try:
        open_orders = _client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for order in open_orders:
            if hasattr(order, "order_type") and "stop" in str(order.order_type).lower():
                _client.cancel_order_by_id(str(order.id))
                logger.info(f"Cancelled stop order {order.id} for {symbol} (fallback scan)")
    except Exception as e:
        logger.error(f"Error in stop cancellation fallback for {symbol}: {e}", exc_info=True)
