"""
main.py — Scheduler and main loop for the Mean Reversion Bot.

Daily schedule (all times Eastern):
  16:30 → Post-close scan:  evaluate entry/exit signals, queue actions
  09:25 → Pre-open execute: submit queued market orders for next open
  09:45 → Fill confirm:     verify fills, place hard stop orders on exchange

Run with:  python main.py
Logs to:   bot.log  (and stdout)

To switch stop variants, change ACTIVE_STOP_MULT in config.py (1.5 or 2.5).
"""
import logging
import time
from datetime import datetime, timezone

import schedule

import config
import executor
import position_tracker as pt
import scanner

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Pending Signal Queue ──────────────────────────────────────────────────────
# Populated by post_close_scan(), consumed by pre_open_execute()
_pending: list[dict] = []


# ── Scheduled Jobs ────────────────────────────────────────────────────────────

def post_close_scan():
    """
    16:30 ET — Scan all symbols and build the pending action queue.
    Checks for: new entry signals, RSI exit signals, time stop triggers.
    """
    global _pending
    _pending.clear()

    logger.info("▶  POST-CLOSE SCAN STARTED")

    scan_results = scanner.run_scan()
    open_positions = executor.get_alpaca_positions()

    for symbol, result in scan_results.items():
        if result.get("error"):
            continue

        in_position = symbol in open_positions

        # ── Time Stop Check ──
        if in_position and pt.time_stop_triggered(symbol):
            _pending.append({"symbol": symbol, "action": "EXIT", "reason": "time_stop"})
            logger.info(f"  ⏰ Queued EXIT (time_stop): {symbol}")
            continue  # Don't also queue an RSI exit for the same symbol

        # ── RSI Exit Check ──
        if in_position and result["exit_signal"]:
            _pending.append({"symbol": symbol, "action": "EXIT", "reason": "rsi_exit"})
            logger.info(f"  📤 Queued EXIT (rsi_exit): {symbol}")
            continue

        # ── Entry Check ──
        if not in_position and result["entry_signal"]:
            _pending.append({"symbol": symbol, "action": "ENTRY", "scan_result": result})
            logger.info(f"  📥 Queued ENTRY: {symbol}")

    logger.info(
        f"▶  SCAN COMPLETE | Pending queue: "
        f"{[(p['symbol'], p['action']) for p in _pending]}"
    )


def pre_open_execute():
    """
    09:25 ET — Execute all queued actions before the market opens.
    Market OPG orders fill at the official opening auction price.
    """
    global _pending

    if not _pending:
        logger.info("▶  PRE-OPEN EXECUTE | No pending actions.")
        return

    logger.info(f"▶  PRE-OPEN EXECUTE | Processing {len(_pending)} action(s)...")

    for action in _pending:
        symbol = action["symbol"]
        kind   = action["action"]

        if kind == "EXIT":
            executor.submit_exit(symbol, reason=action.get("reason", "signal"))

        elif kind == "ENTRY":
            executor.submit_entry(action["scan_result"])

        else:
            logger.warning(f"Unknown action type '{kind}' for {symbol} — skipping.")

    _pending.clear()
    logger.info("▶  PRE-OPEN EXECUTE COMPLETE")


def fill_confirm():
    """
    09:45 ET — Confirm fills and place hard stop orders on the exchange.
    Runs 15 minutes after open to ensure OPG orders have been processed.
    """
    logger.info("▶  FILL CONFIRMATION + STOP PLACEMENT")
    executor.confirm_fills_and_place_stops()


def status_report():
    """
    Log a snapshot of open positions and their current age.
    Runs daily at 16:30 alongside the scan (injected separately for clarity).
    """
    positions = pt.all_positions()
    if not positions:
        logger.info("STATUS | No open positions.")
        return

    logger.info(f"STATUS | {len(positions)} open position(s):")
    for sym, pos in positions.items():
        days = pt.days_open(sym)
        logger.info(
            f"  {sym} | {pos['shares']} shares @ ${pos['entry_price']:.2f} | "
            f"Stop=${pos['stop_price']:.2f} ({pos['stop_mult']}×ATR) | "
            f"Days open: {days}/{config.MAX_HOLD_DAYS} | "
            f"Stop A (1.5×): ${pos['stop_price_a']:.2f} | "
            f"Stop B (2.5×): ${pos['stop_price_b']:.2f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("MEAN REVERSION BOT — STARTING UP")
    logger.info(f"  Universe:         {config.SYMBOLS}")
    logger.info(f"  RSI(2) entry:     <= {config.RSI_ENTRY_THRESHOLD}")
    logger.info(f"  RSI(2) exit:      >= {config.RSI_EXIT_THRESHOLD}")
    logger.info(f"  Trend filter:     Close > SMA({config.SMA_PERIOD})")
    logger.info(f"  Active stop:      {config.ACTIVE_STOP_MULT}× ATR({config.ATR_PERIOD})")
    logger.info(f"  Tracking stop A:  {config.STOP_MULT_A}× ATR (logged, not traded)")
    logger.info(f"  Tracking stop B:  {config.STOP_MULT_B}× ATR (logged, not traded)")
    logger.info(f"  Risk per trade:   {config.RISK_PER_TRADE * 100:.1f}% of equity")
    logger.info(f"  Time stop:        {config.MAX_HOLD_DAYS} days")
    logger.info(f"  Paper trading:    {config.PAPER}")
    logger.info(f"  Schedule:         Scan@{config.SCAN_TIME} | Execute@{config.EXECUTE_TIME} | Confirm@{config.FILL_CONFIRM_TIME}")
    logger.info("=" * 70)

    # Wire up the schedule (server must be in US/Eastern time zone)
    schedule.every().day.at(config.SCAN_TIME).do(post_close_scan)
    schedule.every().day.at(config.SCAN_TIME).do(status_report)
    schedule.every().day.at(config.EXECUTE_TIME).do(pre_open_execute)
    schedule.every().day.at(config.FILL_CONFIRM_TIME).do(fill_confirm)

    logger.info("Scheduler running. Waiting for next event...")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
