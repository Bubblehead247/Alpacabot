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
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce

import config
import position_tracker as pt
import risk

logger = logging.getLogger(__name__)

_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=config.PAPER)


# ── Account ───────────────────────────────────────────────────────────────────

def get_equity() -> float:
    account = _client.get_account()
    return float(account.equity)


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
        order = _client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.OPG,  # Execute at market open
            )
        )

        logger.info(
            f"📥 BUY ORDER submitted | {symbol} | {shares} shares | "
            f"Est. entry=${est_entry:.2f} | Order ID={order.id}"
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
            logger.warning(
                f"{symbol}: In local tracker but not in Alpaca positions. "
                f"Order may not have filled yet or was rejected."
            )
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

def submit_exit(symbol: str, reason: str = "signal") -> bool:
    """
    Submit a Market OPG (next-day open) sell order.
    Cancels any open stop orders first to avoid a double-sell.

    Args:
        symbol: Ticker to exit.
        reason: Human-readable exit reason for logging (e.g. 'rsi_exit', 'time_stop').

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
            f"Reason={reason} | Order ID={order.id}"
        )
        pt.remove(symbol)
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
