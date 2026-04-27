"""
scanner.py — Fetches daily OHLCV bars from Alpaca and evaluates entry/exit conditions.

Entry signal:   Close > SMA(200)  AND  RSI(2) <= 10
Exit signal:    RSI(2) >= 70  OR  position age >= 7 days (time stop handled in main.py)
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

def fetch_bars(symbol: str, lookback_days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Alpaca.
    Requests 1.6× the lookback to account for weekends and market holidays.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days * 1.6))

    # feed=IEX is required on the free Alpaca data plan — the default SIP feed
    # returns 401/"subscription does not permit recent SIP data". IEX is a single
    # exchange but its bars are sufficient for daily-bar mean-reversion signals.
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = _data_client.get_stock_bars(request).df

    # Alpaca returns MultiIndex (symbol, timestamp) for single-symbol requests too
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")

    return bars.sort_index().tail(lookback_days)


# ── Signal Evaluation ─────────────────────────────────────────────────────────

def evaluate_symbol(symbol: str) -> dict:
    """
    Evaluate a single symbol for entry and exit conditions.

    Returns a dict with keys:
        symbol, close, sma200, rsi2, atr14,
        stop_a, stop_b, active_stop,
        entry_signal (bool), exit_signal (bool)
    """
    bars = fetch_bars(symbol)

    if len(bars) < config.SMA_PERIOD + 10:
        logger.warning(f"{symbol}: Insufficient data ({len(bars)} bars). Need {config.SMA_PERIOD + 10}.")
        return {"symbol": symbol, "entry_signal": False, "exit_signal": False, "error": "insufficient_data"}

    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]

    sma200 = sma(close, config.SMA_PERIOD)
    rsi2   = rsi(close, config.RSI_PERIOD)
    atr14  = atr(high, low, close, config.ATR_PERIOD)

    last_close  = round(float(close.iloc[-1]),  2)
    last_sma200 = round(float(sma200.iloc[-1]), 2)
    last_rsi2   = round(float(rsi2.iloc[-1]),   2)
    last_atr    = round(float(atr14.iloc[-1]),  2)

    above_sma200  = last_close > last_sma200
    rsi_oversold  = last_rsi2 <= config.RSI_ENTRY_THRESHOLD
    rsi_overbought = last_rsi2 >= config.RSI_EXIT_THRESHOLD

    entry_signal = above_sma200 and rsi_oversold
    exit_signal  = rsi_overbought

    # Calculate both stop variants from yesterday's close (used for sizing estimate)
    stop_a      = round(last_close - (config.STOP_MULT_A * last_atr), 2)
    stop_b      = round(last_close - (config.STOP_MULT_B * last_atr), 2)
    active_stop = round(last_close - (config.ACTIVE_STOP_MULT * last_atr), 2)

    result = {
        "symbol":       symbol,
        "close":        last_close,
        "sma200":       last_sma200,
        "rsi2":         last_rsi2,
        "atr14":        last_atr,
        "above_sma200": above_sma200,
        "stop_a":       stop_a,       # 1.5 × ATR stop
        "stop_b":       stop_b,       # 2.5 × ATR stop
        "active_stop":  active_stop,  # The one we actually use
        "entry_signal": entry_signal,
        "exit_signal":  exit_signal,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    # Structured log for easy parsing/analysis later
    logger.info(
        f"[{symbol}] Close={last_close} SMA200={last_sma200} RSI2={last_rsi2:.2f} "
        f"ATR={last_atr:.2f} | AboveSMA={above_sma200} | "
        f"Entry={entry_signal} Exit={exit_signal} | "
        f"Stop_A={stop_a} Stop_B={stop_b}"
    )

    if entry_signal:
        logger.info(
            f"  ✅ ENTRY SIGNAL: {symbol} | Active stop ({config.ACTIVE_STOP_MULT}×ATR): {active_stop}"
        )
    if exit_signal:
        logger.info(f"  📤 EXIT SIGNAL:  {symbol} | RSI(2) = {last_rsi2:.2f}")

    return result


# ── Full Scan ─────────────────────────────────────────────────────────────────

def run_scan() -> dict[str, dict]:
    """
    Scan all configured symbols.
    Returns dict of symbol -> evaluation result for all symbols.
    """
    logger.info("=" * 60)
    logger.info(f"Post-close scan started | {datetime.now(timezone.utc).isoformat()}")

    results = {}
    for symbol in config.SYMBOLS:
        try:
            results[symbol] = evaluate_symbol(symbol)
        except Exception as e:
            logger.error(f"Scan error for {symbol}: {e}", exc_info=True)
            results[symbol] = {"symbol": symbol, "entry_signal": False, "exit_signal": False, "error": str(e)}

    entries = [s for s, r in results.items() if r.get("entry_signal")]
    exits   = [s for s, r in results.items() if r.get("exit_signal")]

    logger.info(f"Scan complete | Entry signals: {entries} | Exit signals: {exits}")
    logger.info("=" * 60)

    return results
