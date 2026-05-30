"""
scanner.py — Fetches daily and weekly OHLCV bars and evaluates entry/exit signals.

Entry (ALL must pass):
  Weekly gate:   Close > SMA(50,W)  AND  Close > SMA(200,W)
  Daily trend:   Close > SMA(50,D)
  RSI signal:    RSI(2) crossed back ABOVE 10 today (prev <= 10, now > 10)
  Volume:        Optional — today's volume > 1.5× the 20-day average
                 (only enforced when config.USE_VOLUME_FILTER is True)

Exit (first triggered wins, checked in this priority order):
  Weekly break:  Close < SMA(200,W)  — structural trend broken        [High]
  RSI target:    RSI(2) crossed back BELOW 70 (prev >= 70, now < 70)  [Standard]
  Time stop:     Position held >= 7 calendar days                     [Standard] (checked in main.py)
  Hard stop:     ATR-based GTC order on exchange                      [Highest]  (handled by Alpaca)
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config
from indicators import sma, rsi, atr

logger = logging.getLogger(__name__)

_data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


# ── Data Fetching ─────────────────────────────────────────────────────────────

def _fetch_bars(symbol: str, timeframe: TimeFrame, calendar_days_back: int) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=calendar_days_back)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = _data_client.get_stock_bars(request).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars.sort_index()


def fetch_bars(symbol: str) -> pd.DataFrame:
    """Daily OHLCV — enough for SMA(50), ATR(14), and volume MA(20)."""
    return _fetch_bars(
        symbol, TimeFrame.Day, int(config.LOOKBACK_DAYS * 1.6)
    ).tail(config.LOOKBACK_DAYS)


def fetch_weekly_bars(symbol: str) -> pd.DataFrame:
    """Weekly OHLCV — enough for SMA(50,W) and SMA(200,W)."""
    calendar_days = int(config.WEEKLY_LOOKBACK_WEEKS * 7 * 1.3)
    return _fetch_bars(
        symbol, TimeFrame.Week, calendar_days
    ).tail(config.WEEKLY_LOOKBACK_WEEKS)


# ── Signal Evaluation ─────────────────────────────────────────────────────────

def evaluate_symbol(symbol: str) -> dict:
    """
    Evaluate entry and exit conditions for a symbol.

    Returns a dict with keys:
        symbol, close,
        sma50_daily, sma50_weekly, sma200_weekly,
        rsi2, prev_rsi2, atr14, volume, volume_ma20,
        stop_a, stop_b, active_stop,
        entry_signal (bool), exit_signal (bool), weekly_exit_signal (bool)
    """
    # ── Daily bars ────────────────────────────────────────────────────────────
    bars = fetch_bars(symbol)

    if len(bars) < 60:
        logger.warning(f"{symbol}: Insufficient daily data ({len(bars)} bars). Skipping.")
        return {
            "symbol": symbol, "entry_signal": False,
            "exit_signal": False, "weekly_exit_signal": False,
            "error": "insufficient_data",
        }

    close  = bars["close"]
    high   = bars["high"]
    low    = bars["low"]
    volume = bars["volume"]

    sma50_d  = sma(close, config.SMA_DAILY)
    rsi2     = rsi(close, config.RSI_PERIOD)
    atr14    = atr(high, low, close, config.ATR_PERIOD)
    vol_ma20 = sma(volume, config.VOLUME_MA_PERIOD)

    last_close    = round(float(close.iloc[-1]),   2)
    last_sma50_d  = round(float(sma50_d.iloc[-1]), 2)
    last_rsi2     = round(float(rsi2.iloc[-1]),    2)
    prev_rsi2     = round(float(rsi2.iloc[-2]),    2)
    last_atr      = round(float(atr14.iloc[-1]),   2)
    last_volume   = int(volume.iloc[-1])
    last_vol_ma20 = float(vol_ma20.iloc[-1])

    above_daily_sma50 = last_close > last_sma50_d
    # Entry: RSI(2) was <= 10 last bar, now crossed back above 10
    rsi_crossed_above = (prev_rsi2 <= config.RSI_ENTRY_THRESHOLD) and (last_rsi2 > config.RSI_ENTRY_THRESHOLD)
    # Exit: RSI(2) was >= 70 last bar, now crossed back below 70
    rsi_crossed_below = (prev_rsi2 >= config.RSI_EXIT_THRESHOLD)  and (last_rsi2 < config.RSI_EXIT_THRESHOLD)
    volume_spike      = last_volume > (config.VOLUME_SPIKE_MULT * last_vol_ma20)

    # ── Weekly bars ───────────────────────────────────────────────────────────
    above_weekly_sma50  = False
    above_weekly_sma200 = False
    last_weekly_sma50   = 0.0
    last_weekly_sma200  = 0.0

    try:
        weekly_bars = fetch_weekly_bars(symbol)
        if len(weekly_bars) >= config.SMA_WEEKLY_SLOW + 10:
            w_close  = weekly_bars["close"]
            w_sma50  = sma(w_close, config.SMA_WEEKLY_FAST)
            w_sma200 = sma(w_close, config.SMA_WEEKLY_SLOW)
            last_weekly_sma50  = round(float(w_sma50.iloc[-1]),  2)
            last_weekly_sma200 = round(float(w_sma200.iloc[-1]), 2)
            above_weekly_sma50  = last_close > last_weekly_sma50
            above_weekly_sma200 = last_close > last_weekly_sma200
        else:
            logger.warning(
                f"{symbol}: Insufficient weekly data ({len(weekly_bars)} bars) "
                f"— weekly gate FAILED (need {config.SMA_WEEKLY_SLOW + 10})."
            )
    except Exception as e:
        logger.error(f"{symbol}: Weekly bar fetch failed: {e}", exc_info=True)

    weekly_trend_ok = above_weekly_sma50 and above_weekly_sma200

    # ── Stops ─────────────────────────────────────────────────────────────────
    stop_a      = round(last_close - (config.STOP_MULT_A      * last_atr), 2)
    stop_b      = round(last_close - (config.STOP_MULT_B      * last_atr), 2)
    active_stop = round(last_close - (config.ACTIVE_STOP_MULT * last_atr), 2)

    # ── Signal logic ──────────────────────────────────────────────────────────
    # Volume spike is an optional confirmation (see config.USE_VOLUME_FILTER).
    volume_ok          = volume_spike if config.USE_VOLUME_FILTER else True
    entry_signal       = weekly_trend_ok and above_daily_sma50 and rsi_crossed_above and volume_ok
    exit_signal        = rsi_crossed_below
    weekly_exit_signal = not above_weekly_sma200  # close below weekly SMA(200)

    result = {
        "symbol":              symbol,
        "close":               last_close,
        "sma50_daily":         last_sma50_d,
        "sma50_weekly":        last_weekly_sma50,
        "sma200_weekly":       last_weekly_sma200,
        "rsi2":                last_rsi2,
        "prev_rsi2":           prev_rsi2,
        "atr14":               last_atr,
        "volume":              last_volume,
        "volume_ma20":         round(last_vol_ma20, 0),
        "above_daily_sma50":   above_daily_sma50,
        "above_weekly_sma50":  above_weekly_sma50,
        "above_weekly_sma200": above_weekly_sma200,
        "volume_spike":        volume_spike,
        "stop_a":              stop_a,
        "stop_b":              stop_b,
        "active_stop":         active_stop,
        "entry_signal":        entry_signal,
        "exit_signal":         exit_signal,
        "weekly_exit_signal":  weekly_exit_signal,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"[{symbol}] Close={last_close} | "
        f"SMA50d={last_sma50_d} above={above_daily_sma50} | "
        f"SMA50w={last_weekly_sma50} SMA200w={last_weekly_sma200} "
        f"above50w={above_weekly_sma50} above200w={above_weekly_sma200} | "
        f"RSI2={last_rsi2:.2f} prev={prev_rsi2:.2f} xabove={rsi_crossed_above} | "
        f"Vol={last_volume:,} VolMA={last_vol_ma20:,.0f} spike={volume_spike} | "
        f"ATR={last_atr:.2f} Stop_A={stop_a} Stop_B={stop_b} | "
        f"Entry={entry_signal} RsiExit={exit_signal} WeeklyExit={weekly_exit_signal}"
    )

    if entry_signal:
        logger.info(f"  ✅ ENTRY SIGNAL: {symbol} | Active stop ({config.ACTIVE_STOP_MULT}×ATR): {active_stop}")
    if exit_signal:
        logger.info(
            f"  📤 RSI EXIT: {symbol} | RSI(2) crossed below {config.RSI_EXIT_THRESHOLD} "
            f"({prev_rsi2:.2f} → {last_rsi2:.2f})"
        )
    if weekly_exit_signal:
        logger.info(
            f"  📤 WEEKLY EXIT: {symbol} | Close={last_close} below weekly SMA(200)={last_weekly_sma200}"
        )

    return result


# ── Full Scan ─────────────────────────────────────────────────────────────────

def run_scan() -> dict[str, dict]:
    """Scan all configured symbols. Returns dict of symbol → evaluation result."""
    logger.info("=" * 60)
    logger.info(f"Post-close scan started | {datetime.now(timezone.utc).isoformat()}")

    results = {}
    for symbol in config.SYMBOLS:
        try:
            results[symbol] = evaluate_symbol(symbol)
        except Exception as e:
            logger.error(f"Scan error for {symbol}: {e}", exc_info=True)
            results[symbol] = {
                "symbol": symbol, "entry_signal": False,
                "exit_signal": False, "weekly_exit_signal": False,
                "error": str(e),
            }

    entries       = [s for s, r in results.items() if r.get("entry_signal")]
    rsi_exits     = [s for s, r in results.items() if r.get("exit_signal")]
    weekly_exits  = [s for s, r in results.items() if r.get("weekly_exit_signal")]

    logger.info(
        f"Scan complete | Entries: {entries} | "
        f"RSI exits: {rsi_exits} | Weekly exits: {weekly_exits}"
    )
    logger.info("=" * 60)

    return results
