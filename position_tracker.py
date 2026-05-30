"""
position_tracker.py — JSON-based persistence for open position metadata.

Alpaca tracks shares, cost basis, and P&L. We track what Alpaca doesn't:
  - Entry date (for the 7-day time stop)
  - Which stop multiplier variant was used
  - The stop order ID (so we can cancel it before placing a market exit)
  - Both stop price variants (for performance comparison logging)
"""
import json
import logging
from datetime import date, datetime
from pathlib import Path

import config

logger     = logging.getLogger(__name__)
STATE_FILE = Path("positions.json")


# ── I/O Helpers ───────────────────────────────────────────────────────────────

def _load() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save(data: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Public Interface ──────────────────────────────────────────────────────────

def add(
    symbol: str,
    entry_price: float,
    stop_price: float,
    stop_price_a: float,
    stop_price_b: float,
    shares: int,
    stop_mult: float,
    stop_order_id: str = None,
    rsi2: float = None,
    sma200_weekly: float = None,
    sma50_daily: float = None,
    atr14: float = None,
):
    """Record a new open position."""
    data = _load()
    data[symbol] = {
        "symbol":                  symbol,
        "entry_price":             entry_price,
        "stop_price":              stop_price,      # Active stop (the one on exchange)
        "stop_price_a":            stop_price_a,    # 1.5×ATR — tracked for comparison
        "stop_price_b":            stop_price_b,    # 2.5×ATR — tracked for comparison
        "shares":                  shares,
        "stop_mult":               stop_mult,
        "stop_order_id":           stop_order_id,
        "exit_status":             "open",          # "open" | "exit_pending"
        "rsi2_at_signal":          rsi2,
        "sma200_weekly_at_signal": sma200_weekly,
        "sma50_daily_at_signal":   sma50_daily,
        "atr14_at_signal":         atr14,
        "entry_date":              date.today().isoformat(),
        "opened_at":               datetime.utcnow().isoformat(),
    }
    _save(data)
    logger.info(
        f"Position recorded: {symbol} | {shares} shares @ ${entry_price:.2f} | "
        f"Stop=${stop_price:.2f} ({stop_mult}×ATR)"
    )


def update_stop_order_id(symbol: str, stop_order_id: str):
    """Update the stop order ID after the hard stop is placed on exchange."""
    data = _load()
    if symbol in data:
        data[symbol]["stop_order_id"] = stop_order_id
        _save(data)
        logger.info(f"Stop order ID updated for {symbol}: {stop_order_id}")


def mark_exit_pending(symbol: str, reason: str = "signal_exit"):
    """Mark a position as having an exit order submitted but not yet confirmed filled."""
    data = _load()
    if symbol in data:
        data[symbol]["exit_status"] = "exit_pending"
        data[symbol]["exit_reason"] = reason
        _save(data)
        logger.info(f"Position marked exit_pending: {symbol} | reason={reason}")


def get_exit_pending_symbols() -> list[str]:
    """Return all symbols currently awaiting exit fill confirmation."""
    return [sym for sym, pos in _load().items() if pos.get("exit_status") == "exit_pending"]


def update_entry_price(symbol: str, actual_fill_price: float):
    """Update entry price to actual fill (vs. estimated previous close)."""
    data = _load()
    if symbol in data:
        data[symbol]["entry_price"] = actual_fill_price
        _save(data)
        logger.info(f"Entry price updated for {symbol}: ${actual_fill_price:.2f} (actual fill)")


def remove(symbol: str):
    """Remove a position after it's been closed."""
    data = _load()
    if symbol in data:
        pos = data.pop(symbol)
        _save(data)
        logger.info(
            f"Position removed: {symbol} | "
            f"Was: {pos['shares']} shares @ ${pos['entry_price']:.2f}, "
            f"entered {pos['entry_date']}"
        )


def get(symbol: str) -> dict | None:
    return _load().get(symbol)


def all_positions() -> dict:
    return _load()


def get_stop_order_id(symbol: str) -> str | None:
    pos = get(symbol)
    return pos.get("stop_order_id") if pos else None


def days_open(symbol: str) -> int:
    """Returns calendar days since entry."""
    pos = get(symbol)
    if not pos:
        return 0
    entry = date.fromisoformat(pos["entry_date"])
    return (date.today() - entry).days


def time_stop_triggered(symbol: str) -> bool:
    return days_open(symbol) >= config.MAX_HOLD_DAYS


def get_time_stop_symbols() -> list[str]:
    """Return all symbols whose positions have exceeded the max hold period."""
    expired = [sym for sym in _load() if time_stop_triggered(sym)]
    for sym in expired:
        logger.info(f"⏰ Time stop: {sym} has been open {days_open(sym)} days (max={config.MAX_HOLD_DAYS})")
    return expired
