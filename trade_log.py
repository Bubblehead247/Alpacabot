"""
trade_log.py — Appends one row per closed trade to trades.csv.

Called from executor.confirm_exit_fills() (signal/time exits) and
main.post_close_scan() (hard stop exits). Headers are written automatically
on first use.
"""
import csv
import logging
from datetime import date
from pathlib import Path

logger   = logging.getLogger(__name__)
CSV_FILE = Path("trades.csv")

HEADERS = [
    "symbol", "entry_date", "entry_price", "shares",
    "stop_price", "stop_mult",
    "rsi2_at_signal", "sma200_at_signal", "ema50_at_signal", "atr14_at_signal",
    "exit_date", "exit_price", "exit_reason",
    "days_held", "realized_pnl", "realized_pnl_pct",
]


def log_closed_trade(pos_data: dict, exit_price: float, exit_reason: str):
    """
    Append a closed-trade row to trades.csv.

    Args:
        pos_data:    Position dict from position_tracker (must have entry_price,
                     shares, entry_date, and optional signal indicator fields).
        exit_price:  Actual fill price of the closing order.
        exit_reason: One of: rsi_exit, time_stop, hard_stop, exit_pending_retry.
    """
    entry_price = pos_data["entry_price"]
    shares      = pos_data["shares"]
    entry_date  = date.fromisoformat(pos_data["entry_date"])
    exit_date   = date.today()

    pnl     = round((exit_price - entry_price) * shares, 2)
    pnl_pct = round((exit_price / entry_price - 1) * 100, 3) if entry_price else 0.0

    row = {
        "symbol":           pos_data["symbol"],
        "entry_date":       entry_date.isoformat(),
        "entry_price":      round(entry_price, 4),
        "shares":           shares,
        "stop_price":       pos_data.get("stop_price", ""),
        "stop_mult":        pos_data.get("stop_mult", ""),
        "rsi2_at_signal":   pos_data.get("rsi2_at_signal", ""),
        "sma200_at_signal": pos_data.get("sma200_at_signal", ""),
        "ema50_at_signal":  pos_data.get("ema50_at_signal", ""),
        "atr14_at_signal":  pos_data.get("atr14_at_signal", ""),
        "exit_date":        exit_date.isoformat(),
        "exit_price":       round(exit_price, 4),
        "exit_reason":      exit_reason,
        "days_held":        (exit_date - entry_date).days,
        "realized_pnl":     pnl,
        "realized_pnl_pct": pnl_pct,
    }

    write_header = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        f"📊 Trade logged | {row['symbol']} | "
        f"Entry=${entry_price:.2f} Exit=${exit_price:.2f} | "
        f"P&L=${pnl:+.2f} ({pnl_pct:+.3f}%) | "
        f"Reason={exit_reason} | Days={row['days_held']}"
    )
