"""
main.py — Scheduler and main loop for the Mean Reversion Bot.

Daily schedule (all times Eastern):
  16:30 → Post-close scan:  evaluate entry/exit signals, queue actions
  09:25 → Pre-open execute: submit LOO limit orders (market fallback if unfilled by 9:45)
  09:45 → Fill confirm:     verify fills, place GTC hard stops, resolve exit orders

Run with:  python main.py
Logs to:   bot.log  (and stdout)

To switch stop variants, change ACTIVE_STOP_MULT in config.py (1.5 or 2.5).
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import schedule

import config
import eval_checkpoint
import executor
import notifier
import position_tracker as pt
import scanner
import sectors
import trade_log

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
# Populated by post_close_scan(), consumed by pre_open_execute().
# Cleared at the start of each scan so stale signals never carry over.
# Mirrored to pending.json so a restart between scan and execute can't drop
# queued signals (learned the hard way: a 4:58 PM restart ate an SPY exit).
_pending: list[dict] = []

PENDING_FILE         = Path(__file__).resolve().parent / "pending.json"
PENDING_MAX_AGE_DAYS = 5  # Discard a restored queue older than this — signals are stale.


def _json_scalar(o):
    """JSON fallback for numpy scalars coming out of pandas indicators."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _save_pending():
    """Mirror the in-memory queue to disk (called after scan and after execute)."""
    payload = {
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "actions":   _pending,
    }
    PENDING_FILE.write_text(json.dumps(payload, default=_json_scalar, indent=2))


def _load_pending():
    """Restore the queue from disk on startup, unless it has gone stale."""
    if not PENDING_FILE.exists():
        return
    try:
        payload   = json.loads(PENDING_FILE.read_text())
        queued_at = datetime.fromisoformat(payload["queued_at"])
        age_days  = (datetime.now(timezone.utc) - queued_at).days
        if age_days > PENDING_MAX_AGE_DAYS:
            logger.warning(f"pending.json is {age_days} days old — discarding stale queue.")
            return
        _pending.extend(payload["actions"])
        if _pending:
            logger.info(
                f"Restored pending queue from pending.json: "
                f"{[(p['symbol'], p['action']) for p in _pending]}"
            )
    except Exception as e:
        logger.error(f"Failed to restore pending.json: {e}", exc_info=True)


# ── Market-Day Guard ──────────────────────────────────────────────────────────

def _skip_if_closed(job_name: str) -> bool:
    """Returns True (and logs) if today is NOT a US equities trading day."""
    if executor.is_trading_day_today():
        return False
    logger.info(f"⏭  {job_name}: market closed today — skipping.")
    return True


# ── Scheduled Jobs ────────────────────────────────────────────────────────────

def post_close_scan():
    """
    16:30 ET — Scan all symbols and build the pending action queue.

    Exit priority (first match wins per symbol):
      1. Hard stop    — detected from Alpaca position disappearing (handled here)
      2. Time stop    — position age >= MAX_HOLD_DAYS
      3. Weekly break — close below weekly SMA(200)
      4. RSI exit     — RSI(2) crosses back below 70
    """
    if _skip_if_closed("POST-CLOSE SCAN"):
        return

    global _pending
    _pending.clear()

    logger.info("▶  POST-CLOSE SCAN STARTED")

    scan_results     = scanner.run_scan()
    open_positions   = executor.get_alpaca_positions()

    # Detect hard stop exits: positions tracked as "open" that vanished from Alpaca
    # must have been closed by the GTC stop during the session.
    for symbol, pos_data in list(pt.all_positions().items()):
        if pos_data.get("exit_status") == "open" and symbol not in open_positions:
            exit_price = executor.get_last_fill_price(symbol, side="sell")
            trade_log.log_closed_trade(
                pos_data,
                exit_price or pos_data["stop_price"],
                "hard_stop",
            )
            pt.remove(symbol)
            logger.info(f"  🛑 Hard stop exit detected and logged: {symbol}")

    for symbol, result in scan_results.items():
        if result.get("error"):
            continue

        in_position = symbol in open_positions

        # ── Time Stop (highest soft-exit priority) ──
        if in_position and pt.time_stop_triggered(symbol):
            _pending.append({"symbol": symbol, "action": "EXIT", "reason": "time_stop", "close": result["close"]})
            logger.info(f"  ⏰ Queued EXIT (time_stop): {symbol}")
            continue

        # ── Weekly Trend Break (High priority exit) ──
        if in_position and result.get("weekly_exit_signal"):
            _pending.append({"symbol": symbol, "action": "EXIT", "reason": "weekly_trend_break", "close": result["close"]})
            logger.info(f"  📤 Queued EXIT (weekly_trend_break): {symbol}")
            continue

        # ── RSI Exit (Standard priority) ──
        if in_position and result["exit_signal"]:
            _pending.append({"symbol": symbol, "action": "EXIT", "reason": "rsi_exit", "close": result["close"]})
            logger.info(f"  📤 Queued EXIT (rsi_exit): {symbol}")
            continue

        # ── Entry ──
        if not in_position and result["entry_signal"]:
            queued_entries = sum(1 for p in _pending if p["action"] == "ENTRY")
            # Per-sector cap: count this sector across open positions + queued entries.
            cand_sector  = sectors.sector_of(symbol)
            sector_count = sum(1 for s in open_positions if sectors.sector_of(s) == cand_sector)
            sector_count += sum(1 for p in _pending if p["action"] == "ENTRY"
                                and sectors.sector_of(p["symbol"]) == cand_sector)
            if cand_sector == sectors.UNKNOWN:
                logger.warning(f"  ⚠️  {symbol} has no sector mapping — re-run screener.py to refresh sectors.csv")

            if len(open_positions) + queued_entries >= config.MAX_POSITIONS:
                logger.info(
                    f"  ⛔ Skipping ENTRY {symbol} — at max positions "
                    f"({len(open_positions)} open + {queued_entries} queued = {config.MAX_POSITIONS})"
                )
            elif sector_count >= config.MAX_PER_SECTOR:
                logger.info(
                    f"  ⛔ Skipping ENTRY {symbol} — sector '{cand_sector}' at cap "
                    f"({sector_count}/{config.MAX_PER_SECTOR})"
                )
            else:
                _pending.append({"symbol": symbol, "action": "ENTRY", "scan_result": result})
                logger.info(f"  📥 Queued ENTRY: {symbol} | sector={cand_sector}")
                notifier.send_signal(symbol, result["rsi2"])
        elif not in_position and result.get("rsi2", 100) <= 20 and result.get("prereqs_ok", False):
            logger.info(f"  ⚠️  Warning zone: {symbol} RSI(2)={result['rsi2']:.1f}")
            notifier.send_warning(symbol, result["rsi2"])

    # Re-queue exits for positions marked exit_pending from a prior cycle
    for symbol in pt.get_exit_pending_symbols():
        if symbol in open_positions:
            already_queued = any(p["symbol"] == symbol and p["action"] == "EXIT" for p in _pending)
            if not already_queued:
                close_ref = float(open_positions[symbol].avg_entry_price)
                _pending.append({
                    "symbol": symbol,
                    "action": "EXIT",
                    "reason": "exit_pending_retry",
                    "close":  close_ref,
                })
                logger.warning(f"  🔁 Re-queued EXIT (exit_pending_retry): {symbol} | close_ref=${close_ref:.2f}")

    _save_pending()
    logger.info(
        f"▶  SCAN COMPLETE | Pending queue: "
        f"{[(p['symbol'], p['action']) for p in _pending]}"
    )


def pre_open_execute():
    """
    09:25 ET — Submit LOO limit orders for all queued actions.
    Unfilled limits are caught at 09:45 and replaced with market orders.
    """
    if _skip_if_closed("PRE-OPEN EXECUTE"):
        return

    global _pending

    if not _pending:
        logger.info("▶  PRE-OPEN EXECUTE | No pending actions.")
        return

    logger.info(f"▶  PRE-OPEN EXECUTE | Processing {len(_pending)} action(s)...")

    for action in _pending:
        symbol = action["symbol"]
        kind   = action["action"]

        if kind == "EXIT":
            executor.submit_exit(symbol, reason=action.get("reason", "signal"), close_price=action.get("close", 0.0))
        elif kind == "ENTRY":
            executor.submit_entry(action["scan_result"])
        else:
            logger.warning(f"Unknown action type '{kind}' for {symbol} — skipping.")

    _pending.clear()
    _save_pending()
    logger.info("▶  PRE-OPEN EXECUTE COMPLETE")


def fill_confirm():
    """
    09:45 ET — Confirm fills, place hard stops, and resolve pending exits.
    LOO orders unfilled by this point get replaced with DAY market orders.
    """
    if _skip_if_closed("FILL CONFIRM"):
        return

    logger.info("▶  FILL CONFIRMATION + STOP PLACEMENT")
    executor.confirm_fills_and_place_stops()
    executor.confirm_exit_fills()


def status_report():
    """Log a snapshot of open positions and their current age."""
    if _skip_if_closed("STATUS"):
        return

    # Paper-trading evaluation checkpoint: one-time push when 25 closed trades
    # accumulate since the v1.4 go-live (see eval_checkpoint.py). Never raises.
    cp = eval_checkpoint.check_and_notify()
    logger.info(f"STATUS | Eval checkpoint: {cp['new']}/{cp['target']} trades"
                + ("  ✅ REACHED — review fidelity" if cp["reached"] else ""))

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
    if config.USE_TREND_FILTER:
        logger.info(f"  Weekly gate:      Close > SMA({config.SMA_WEEKLY_FAST},W) AND SMA({config.SMA_WEEKLY_SLOW},W)")
    else:
        logger.info(f"  Weekly gate:      OFF (config.USE_TREND_FILTER)")
    if config.USE_DAILY_SMA200_FILTER:
        logger.info(f"  Daily trend gate: Close > SMA({config.SMA_DAILY_TREND},D)  [Connors regime]")
    else:
        logger.info(f"  Daily trend gate: OFF (config.USE_DAILY_SMA200_FILTER)")
    if config.ENTRY_MODE == "oversold":
        logger.info(f"  RSI(2) entry:     OVERSOLD — RSI(2) < {config.RSI_ENTRY_THRESHOLD} (Connors, buy the dip)")
    else:
        logger.info(f"  RSI(2) entry:     CROSSBACK — crosses ABOVE {config.RSI_ENTRY_THRESHOLD} (legacy)")
    logger.info(f"  RSI(2) exit:      crosses below {config.RSI_EXIT_THRESHOLD} (prev >= {config.RSI_EXIT_THRESHOLD})")
    if config.USE_VOLUME_FILTER:
        logger.info(f"  Volume filter:    ON — volume > {config.VOLUME_SPIKE_MULT}× MA({config.VOLUME_MA_PERIOD})")
    else:
        logger.info(f"  Volume filter:    OFF (logged only; see config.USE_VOLUME_FILTER)")
    if config.USE_REGIME_FILTER:
        logger.info(f"  Regime filter:    ON — weekly ADX({config.REGIME_ADX_PERIOD}) in [{config.REGIME_ADX_MIN}, {config.REGIME_ADX_MAX})")
    else:
        logger.info(f"  Regime filter:    OFF (logged only; see config.USE_REGIME_FILTER)")
    logger.info(f"  Active stop:      {config.ACTIVE_STOP_MULT}× ATR({config.ATR_PERIOD})")
    logger.info(f"  Tracking stop A:  {config.STOP_MULT_A}× ATR (logged, not traded)")
    logger.info(f"  Tracking stop B:  {config.STOP_MULT_B}× ATR (logged, not traded)")
    logger.info(f"  Risk per trade:   {config.RISK_PER_TRADE * 100:.1f}% of equity")
    logger.info(f"  Max positions:    {config.MAX_POSITIONS} ({config.MAX_POSITIONS * config.RISK_PER_TRADE * 100:.0f}% max total risk)")
    logger.info(f"  Time stop:        {config.MAX_HOLD_DAYS} days")
    logger.info(f"  Orders:           LOO limit first; market fallback at {config.FILL_CONFIRM_TIME}")
    logger.info(f"  Paper trading:    {config.PAPER}")
    logger.info(f"  Schedule:         Scan@{config.SCAN_TIME} | Execute@{config.EXECUTE_TIME} | Confirm@{config.FILL_CONFIRM_TIME}")
    logger.info("=" * 70)

    _load_pending()

    schedule.every().day.at(config.SCAN_TIME).do(post_close_scan)
    schedule.every().day.at(config.SCAN_TIME).do(status_report)
    schedule.every().day.at(config.EXECUTE_TIME).do(pre_open_execute)
    schedule.every().day.at(config.FILL_CONFIRM_TIME).do(fill_confirm)

    logger.info("Scheduler running. Waiting for next event...")

    # A job that raises is never marked as run by `schedule`, so run_pending()
    # retries it on the next tick — transient network errors (the 2026-07-06
    # DNS crash) heal themselves. Alert only once the streak reaches 3 (~90s of
    # continuous failure): single stale-connection resets self-heal on the
    # first retry and aren't worth a page.
    consecutive_errors = 0
    while True:
        try:
            schedule.run_pending()
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(
                f"Scheduled job failed (attempt {consecutive_errors}) — retrying in 30s: {e}",
                exc_info=True,
            )
            if consecutive_errors == 3:
                notifier.send_error(f"{type(e).__name__}: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
