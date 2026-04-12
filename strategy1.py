"""
Swing Test Strategy — Alpaca Paper Trading Automation
==================================================
Thematic buy-and-hold core (DCA monthly) + Momentum swing sleeve

Sleeves:
  Nuclear/SMR:       LEU, CCJ       ~$300 core
  AI/Semiconductors: NVDA, TSM      ~$300 core
  Crypto-adjacent:   COIN, MSTR     ~$300 core
  Swing sleeve:      stocks/ETFs/crypto  $150 active

Rules:
  - Monthly DCA on the 1st: $16.67 into each core holding
  - Swing entry: RSI > 55 AND volume > 20-day average
  - Swing exit:  -8% stop loss | +15% take profit | 14-day max hold
  - Rebalance alert if any theme exceeds 50% of total portfolio
"""

import os
import time
import logging
import smtplib
import schedule
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import pandas as pd
import ta  # pip install ta

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

API_KEY        = os.getenv("ALPACA_API_KEY")
API_SECRET     = os.getenv("ALPACA_API_SECRET")
PAPER          = True   # ← flip to False when ready for live trading

EMAIL_FROM     = os.getenv("EMAIL_FROM")
EMAIL_TO       = os.getenv("EMAIL_TO")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Core holdings: symbol → monthly DCA dollar amount
CORE_HOLDINGS = {
    "LEU":  16.67,   # Nuclear / SMR
    "CCJ":  16.67,
    "NVDA": 16.67,   # AI / Semiconductors
    "TSM":  16.67,
    "COIN": 16.67,   # Crypto-adjacent
    "MSTR": 16.67,
}

# Theme groupings (for rebalance alert)
THEMES = {
    "Nuclear/SMR":       ["LEU",  "CCJ"],
    "AI/Semiconductors": ["NVDA", "TSM"],
    "Crypto-adjacent":   ["COIN", "MSTR"],
}

# Swing watchlist
SWING_STOCKS = ["NVDA", "AMD", "COIN", "TSLA", "TQQQ", "SOXL"]
SWING_CRYPTO = ["BTC/USD", "ETH/USD"]

# Swing rules
SWING_BUDGET        = 50.00   # $ per swing position
MAX_SWING_POSITIONS = 3
STOP_LOSS_PCT       = 0.08    # exit at -8%
TAKE_PROFIT_PCT     = 0.15    # exit at +15%
MAX_HOLD_DAYS       = 14
REBALANCE_THRESHOLD = 0.50    # warn if any theme > 50% of portfolio

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("strategy1.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── API Clients ───────────────────────────────────────────────────────────────

trading     = TradingClient(API_KEY, API_SECRET, paper=PAPER)
stock_data  = StockHistoricalDataClient(API_KEY, API_SECRET)
crypto_data = CryptoHistoricalDataClient(API_KEY, API_SECRET)

# ── Helper Utilities ──────────────────────────────────────────────────────────

def send_email(subject: str, body: str):
    """Send a plain text email alert via Gmail."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        log.info(f"EMAIL SENT -- {subject}")
    except Exception as exc:
        log.error(f"Email failed: {exc}")


def is_market_open() -> bool:
    return trading.get_clock().is_open


def get_positions() -> dict:
    """Return {symbol: position} for all current positions."""
    return {p.symbol: p for p in trading.get_all_positions()}


def place_notional_order(symbol: str, notional: float, side: OrderSide):
    """Submit a fractional notional market order."""
    req = MarketOrderRequest(
        symbol=symbol,
        notional=round(notional, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    order = trading.submit_order(req)
    log.info(f"ORDER  {side.value.upper():5s}  {symbol:<10}  ${notional:.2f}  id={order.id}")
    return order


def get_rsi_and_volume(symbol: str, crypto: bool = False) -> tuple:
    """Return (rsi_latest, volume_latest, volume_20d_avg) for a symbol."""
    end   = datetime.utcnow()
    start = end - timedelta(days=60)

    if crypto:
        bars = crypto_data.get_crypto_bars(
            CryptoBarsRequest(symbol_or_symbols=symbol,
                              timeframe=TimeFrame.Day,
                              start=start, end=end)
        ).df
    else:
        bars = stock_data.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbol,
                             timeframe=TimeFrame.Day,
                             start=start, end=end)
        ).df

    # Flatten MultiIndex if present
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)

    bars["rsi"] = ta.momentum.RSIIndicator(bars["close"], window=14).rsi()
    rsi     = bars["rsi"].iloc[-1]
    vol     = bars["volume"].iloc[-1]
    vol_avg = bars["volume"].iloc[-20:].mean()
    return rsi, vol, vol_avg

# ── Core Strategy Functions ───────────────────────────────────────────────────

def run_monthly_dca():
    """Buy a fixed notional amount of every core holding (runs on the 1st)."""
    log.info("=" * 60)
    log.info("MONTHLY DCA — starting")
    if not is_market_open():
        log.warning("Market closed — DCA skipped (will retry next run)")
        return
    for symbol, amount in CORE_HOLDINGS.items():
        try:
            place_notional_order(symbol, amount, OrderSide.BUY)
        except Exception as exc:
            log.error(f"DCA failed for {symbol}: {exc}")
    log.info("MONTHLY DCA — complete")


def check_swing_entries():
    """
    Scan swing watchlist for momentum entry signals.
    Entry criteria: RSI > 55 AND today's volume > 20-day average volume.
    Skip if already at max open swing positions.
    """
    log.info("SWING SCAN — checking for entries")
    if not is_market_open():
        return

    positions   = get_positions()
    core_syms   = set(CORE_HOLDINGS.keys())
    swing_count = sum(1 for s in positions if s not in core_syms)

    if swing_count >= MAX_SWING_POSITIONS:
        log.info(f"SWING SCAN — at max positions ({MAX_SWING_POSITIONS}), skipping")
        return

    watchlist = [(s, False) for s in SWING_STOCKS] + [(s, True) for s in SWING_CRYPTO]

    for symbol, is_crypto in watchlist:
        if symbol in positions:
            continue  # Already in this position
        try:
            rsi, vol, vol_avg = get_rsi_and_volume(symbol, crypto=is_crypto)
            log.info(f"  {symbol:<12}  RSI={rsi:5.1f}  vol={vol:>12,.0f}  avg={vol_avg:>12,.0f}")

            if rsi > 55 and vol > vol_avg:
                log.info(f"  ENTRY SIGNAL — {symbol}  (RSI {rsi:.1f}, vol {vol/vol_avg:.1f}x avg)")
                send_email(
                    f"Alpaca Bot -- Entry: {symbol}",
                    f"Bought ${SWING_BUDGET:.0f} of {symbol}\nRSI: {rsi:.1f}\nVolume: {vol:,.0f} (avg {vol_avg:,.0f})"
                )
                place_notional_order(symbol, SWING_BUDGET, OrderSide.BUY)
                swing_count += 1
                if swing_count >= MAX_SWING_POSITIONS:
                    log.info("  Max swing positions reached — stopping scan")
                    break
        except Exception as exc:
            log.error(f"Swing entry scan error for {symbol}: {exc}")


def check_swing_exits():
    """
    Check each open swing position against exit rules:
      - Stop loss:    P&L <= -8%
      - Take profit:  P&L >= +15%
      - Max hold:     position age >= 14 days
    Core holdings are never exited by this function.
    """
    log.info("SWING EXIT — checking positions")
    positions = get_positions()
    core_syms = set(CORE_HOLDINGS.keys())

    for symbol, pos in positions.items():
        if symbol in core_syms:
            continue

        entry    = float(pos.avg_entry_price)
        current  = float(pos.current_price)
        pnl_pct  = (current - entry) / entry
        age_days = (datetime.utcnow() - pos.created_at.replace(tzinfo=None)).days

        reason = None
        if pnl_pct <= -STOP_LOSS_PCT:
            reason = f"STOP LOSS  ({pnl_pct:+.1%})"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            reason = f"TAKE PROFIT ({pnl_pct:+.1%})"
        elif age_days >= MAX_HOLD_DAYS:
            reason = f"MAX HOLD   ({age_days} days)"

        if reason:
            try:
                mkt_val = float(pos.market_value)
                place_notional_order(symbol, mkt_val, OrderSide.SELL)
                log.info(f"  EXIT  {symbol:<10}  {reason}")
                send_email(
                    f"Alpaca Bot -- Exit: {symbol}",
                    f"Sold {symbol}\nReason: {reason}\nP&L: {pnl_pct:+.1%}"
                )
            except Exception as exc:
                log.error(f"Exit failed for {symbol}: {exc}")


def check_rebalance():
    """Log theme allocations and warn if any theme exceeds 50% of portfolio."""
    log.info("REBALANCE CHECK")
    positions = get_positions()
    total_val = sum(float(p.market_value) for p in positions.values())
    if total_val == 0:
        log.info("  No positions yet.")
        return

    for theme, symbols in THEMES.items():
        theme_val = sum(
            float(positions[s].market_value) for s in symbols if s in positions
        )
        pct = theme_val / total_val
        flag = "  ⚠  REBALANCE RECOMMENDED" if pct > REBALANCE_THRESHOLD else ""
        log.info(f"  {theme:<25}  {pct:>6.1%}  (${theme_val:,.2f}){flag}")
        if pct > REBALANCE_THRESHOLD:
            send_email(
                f"Alpaca Bot -- Rebalance Alert: {theme}",
                f"{theme} is {pct:.1%} of your portfolio (limit is {REBALANCE_THRESHOLD:.0%})\nCurrent value: ${theme_val:,.2f}"
            )


def print_portfolio_summary():
    """Print a daily snapshot of all positions and account P&L, then email summary."""
    log.info("=" * 60)
    log.info("PORTFOLIO SUMMARY")
    account   = trading.get_account()
    positions = get_positions()

    log.info(f"  Cash:           ${float(account.cash):>10,.2f}")
    log.info(f"  Portfolio value:${float(account.portfolio_value):>10,.2f}")
    log.info(f"  Day P&L:        ${float(account.equity) - float(account.last_equity):>+10,.2f}")
    log.info("")
    log.info(f"  {'Symbol':<12} {'Qty':>8} {'Entry':>9} {'Current':>9} {'P&L $':>9} {'P&L %':>7}")
    log.info(f"  {'-' * 58}")

    for sym, p in sorted(positions.items()):
        entry   = float(p.avg_entry_price)
        current = float(p.current_price)
        pnl_pct = (current - entry) / entry
        log.info(
            f"  {sym:<12} {float(p.qty):>8.4f}"
            f"  ${entry:>8.2f}  ${current:>8.2f}"
            f"  ${float(p.unrealized_pl):>+8.2f}  {pnl_pct:>+6.1%}"
        )
    log.info("")
    check_rebalance()

    # Email daily summary
    body = (
        f"Portfolio Value: ${float(account.portfolio_value):,.2f}\n"
        f"Cash:            ${float(account.cash):,.2f}\n"
        f"Day P&L:         ${float(account.equity) - float(account.last_equity):+,.2f}\n\n"
        f"Positions:\n"
    )
    for sym, p in sorted(positions.items()):
        pnl_pct = (float(p.current_price) - float(p.avg_entry_price)) / float(p.avg_entry_price)
        body += f"  {sym:<12} ${float(p.market_value):>9,.2f}  {pnl_pct:>+6.1%}\n"
    send_email("Alpaca Bot -- Daily Summary", body)
    log.info("=" * 60)

# ── Scheduler Setup ───────────────────────────────────────────────────────────

def setup_schedule():
    WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    # Monthly DCA — 1st of the month at 10:00 AM ET
    schedule.every().day.at("10:00").do(
        lambda: run_monthly_dca() if datetime.now().day == 1 else None
    )

    for day in WEEKDAYS:
        # Swing entry scan — 10:30 AM ET (after open volatility settles)
        getattr(schedule.every(), day).at("10:30").do(check_swing_entries)
        # Swing exit check — 3:45 PM ET (15 min before close)
        getattr(schedule.every(), day).at("15:45").do(check_swing_exits)
        # Daily summary — 4:00 PM ET
        getattr(schedule.every(), day).at("16:00").do(print_portfolio_summary)

    log.info("Schedule configured:")
    log.info("  Monthly DCA      - 1st of each month at 10:00 AM ET")
    log.info("  Swing entries    - Weekdays at 10:30 AM ET")
    log.info("  Swing exits      - Weekdays at 3:45 PM ET")
    log.info("  Portfolio summary- Weekdays at 4:00 PM ET")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Swing Test Strategy — Alpaca Paper Trading")
    log.info(f"Mode: {'PAPER' if PAPER else 'LIVE'}")
    log.info("=" * 60)

    print_portfolio_summary()
    setup_schedule()

    log.info("Running — press Ctrl+C to stop")
    while True:
        schedule.run_pending()
        time.sleep(30)
