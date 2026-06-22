"""
eval_checkpoint.py — Paper-trading evaluation checkpoint for the Connors v1.4 config.

Counts CLOSED trades logged to trades.csv since a baseline (captured when the
checkpoint was set) and fires a ONE-TIME reminder when 25 new trades accumulate —
the point at which to evaluate EXECUTION FIDELITY. (25 trades is NOT enough to
confirm the edge statistically — that needs ~170+ trades / years; the backtest is
the edge evidence. 25 is the "is the bot executing the backtest faithfully" gate.)

The bot calls check_and_notify() once a day from main.status_report(). You can
also run it manually:

    python eval_checkpoint.py            # progress (N of 25) + checklist if reached
    python eval_checkpoint.py --reset    # re-baseline to the current trade count
                                         # >> run this right after restarting into v1.4 <<
"""
import argparse
import csv
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE      = Path(__file__).resolve().parent
TRADES_CSV = _HERE / "trades.csv"
MARKER     = _HERE / "eval_checkpoint.json"
TARGET     = 25

CHECKLIST = [
    "Slippage: realized entry/exit fills within ~5 bps/side of the reference price (edge is cost-fragile).",
    "Stops: GTC ATR(2.5x) stop placed on every fill and triggers correctly; 7-day time stop fires.",
    "Signals: live entries/exits match backtest.py for the same dates (no live/backtest drift).",
    "Caps: MAX_POSITIONS(5) and MAX_PER_SECTOR(2) respected; sizing ~1% risk.",
    "Stats vs backtest: win ~68%, avg hold ~3-4 days, exit-reason mix sane; realized PF in ~1.1-1.4 band.",
    "GUARDRAIL: halt and reassess regardless of count if realized drawdown >25% or any stop fails to trigger.",
]


def _total_closed() -> int:
    """Closed-trade rows in trades.csv (excludes the header)."""
    if not TRADES_CSV.exists():
        return 0
    with open(TRADES_CSV, newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def _load() -> dict:
    if MARKER.exists():
        return json.loads(MARKER.read_text())
    m = {"baseline": _total_closed(), "target": TARGET,
         "notified": False, "set_at": date.today().isoformat()}
    MARKER.write_text(json.dumps(m, indent=2))
    return m


def _save(m: dict) -> None:
    MARKER.write_text(json.dumps(m, indent=2))


def progress() -> tuple[int, dict]:
    m = _load()
    return _total_closed() - m["baseline"], m


def check_and_notify() -> dict:
    """Daily hook for the bot: sends a one-time push when the checkpoint is hit.
    Never raises — a reminder must not be able to crash the trading loop."""
    try:
        new, m = progress()
        if new >= m["target"] and not m["notified"]:
            try:
                import notifier
                notifier.send_checkpoint(new, m["target"])
            except Exception as e:
                logger.warning(f"checkpoint notify failed: {e}")
            m["notified"] = True
            _save(m)
            logger.info(f"EVAL CHECKPOINT REACHED — {new} closed trades since "
                        f"{m['set_at']}. Time to review execution fidelity.")
        return {"new": new, "target": m["target"], "reached": new >= m["target"]}
    except Exception as e:
        logger.warning(f"checkpoint check failed: {e}")
        return {"new": 0, "target": TARGET, "reached": False}


def _print_status() -> None:
    new, m = progress()
    print(f"Connors v1.4 paper checkpoint: {new}/{m['target']} new closed trades "
          f"(baseline {m['baseline']} set {m['set_at']}).")
    if new >= m["target"]:
        print("\n✅ CHECKPOINT REACHED — evaluate EXECUTION FIDELITY (not edge significance):")
        for c in CHECKLIST:
            print(f"  • {c}")
    else:
        print(f"  {m['target'] - new} to go — let it run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Connors v1.4 25-trade evaluation checkpoint")
    ap.add_argument("--reset", action="store_true",
                    help="re-baseline to the current trade count (use right after the v1.4 restart)")
    args = ap.parse_args()
    if args.reset:
        m = _load()
        m["baseline"], m["notified"], m["set_at"] = _total_closed(), False, date.today().isoformat()
        _save(m)
        print(f"Re-baselined: {m['baseline']} trades as of {m['set_at']}. "
              f"Counting the next {m['target']} from here.")
    else:
        _print_status()
