"""
screener.py — Standalone scan of all Alpaca US equities for sideways stocks.

This tool is SEPARATE from the live bot. It does not place orders, does not
touch scanner.run_scan(), and is never imported by main.py. It scans the whole
tradable universe and ranks names by a combined "sideways score" — how
range-bound (low-trend), tight, and liquid a stock is.

A high score means: weak trend (low ADX), price that hugs its own mean, and
enough dollar volume to actually trade. These are the kind of oscillating names
a mean-reversion strategy likes.

Funnel (keeps API cost to ~a couple dozen requests):
  Stage 0  fetch_tradable_universe()  — all active, tradable US equities
  Stage 1  liquidity_prefilter()      — cheap 1-bar snapshot screen on price/volume
  Stage 2  fetch_daily_bars_batched() — full daily history for survivors only
  Score    sideways_score()           — blend trend + range + liquidity → 0..100

Run:  python screener.py --top 30
"""
import argparse
import csv
import logging
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

import config
import sectors
from indicators import adx, sma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_data_client    = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
_trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=config.PAPER)

_PACE_SECONDS = 0.3   # small gap between requests to stay friendly to the API


# ── Rate-limit safety ─────────────────────────────────────────────────────────

def _with_backoff(fn, *args, **kwargs):
    """Call fn, retrying on HTTP 429 with a doubling wait (0.5s → 1s → 2s …)."""
    delay = 0.5
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            if getattr(e, "status_code", None) == 429 and attempt < 5:
                logger.warning(f"Rate limited (429). Backing off {delay:.1f}s…")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Exceeded retry budget after repeated 429s")


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ── Stage 0: universe ─────────────────────────────────────────────────────────

def fetch_tradable_universe() -> list[str]:
    """All active, tradable US-equity symbols, minus obvious non-common tickers."""
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets  = _with_backoff(_trading_client.get_all_assets, request)

    symbols = [
        a.symbol for a in assets
        if a.tradable
        and "." not in a.symbol      # drop share-class / preferreds (e.g. BRK.B)
        and "/" not in a.symbol      # drop pair-style tickers
        and len(a.symbol) <= 5       # warrants/units usually run longer
    ]
    logger.info(f"Stage 0: {len(symbols)} tradable US equities from {len(assets)} assets.")
    return symbols


# ── Stage 1: cheap liquidity screen ───────────────────────────────────────────

def liquidity_prefilter(symbols: list[str]) -> dict[str, float]:
    """
    One snapshot bar per symbol → keep liquid, non-penny names.

    Returns {symbol: dollar_volume} for survivors. Costs one request per
    SCREEN_SNAPSHOT_CHUNK symbols (~1-3 requests total).
    """
    survivors: dict[str, float] = {}

    for chunk in _chunked(symbols, config.SCREEN_SNAPSHOT_CHUNK):
        request   = StockSnapshotRequest(symbol_or_symbols=chunk, feed=DataFeed.IEX)
        snapshots = _with_backoff(_data_client.get_stock_snapshot, request)

        for symbol, snap in snapshots.items():
            bar = snap.daily_bar or snap.previous_daily_bar
            if bar is None or not bar.close or not bar.volume:
                continue
            dollar_volume = bar.close * bar.volume
            if bar.close >= config.SCREEN_MIN_PRICE and dollar_volume >= config.SCREEN_MIN_DOLLAR_VOLUME:
                survivors[symbol] = dollar_volume

        time.sleep(_PACE_SECONDS)

    logger.info(f"Stage 1: {len(survivors)} liquid names survived the pre-filter.")
    return survivors


# ── Stage 2: full daily history for survivors ─────────────────────────────────

def fetch_daily_bars_batched(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """
    Batched daily OHLCV for the survivor set. Sizes each request by
    SCREEN_BARS_CHUNK symbols and follows pagination inside the SDK.

    Returns {symbol: DataFrame sorted by date}.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days * 1.6))  # calendar→trading-day buffer
    out: dict[str, pd.DataFrame] = {}

    for chunk in _chunked(symbols, config.SCREEN_BARS_CHUNK):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        df = _with_backoff(_data_client.get_stock_bars, request).df
        if df.empty:
            continue
        # Multi-symbol result is a (symbol, timestamp) MultiIndex — same unpack
        # pattern as scanner.py.
        for symbol in chunk:
            if symbol in df.index.get_level_values("symbol"):
                out[symbol] = df.xs(symbol, level="symbol").sort_index()
        time.sleep(_PACE_SECONDS)

    logger.info(f"Stage 2: fetched daily bars for {len(out)} symbols.")
    return out


# ── Sideways score ────────────────────────────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sideways_score(bars: pd.DataFrame, dollar_volume: float) -> dict | None:
    """
    Blend three 0..1 components into a 0..100 sideways score for one symbol.

    trend     low ADX → high score (1 - ADX/40, clamped)
    range     fraction of recent closes within ±band% of SMA → high score
    liquidity log-scaled dollar volume → high score

    Returns a result dict, or None if there isn't enough data.
    """
    lookback = config.SCREEN_LOOKBACK_DAYS
    if len(bars) < lookback:
        return None

    recent = bars.tail(lookback)
    close, high, low = recent["close"], recent["high"], recent["low"]

    # Trendlessness — ADX near 0 is dead-sideways; 40+ is a strong trend.
    last_adx = float(adx(high, low, close, config.SCREEN_ADX_PERIOD).iloc[-1])
    trend_score = _clamp(1.0 - last_adx / 40.0)

    # Range-bound — how often price sits inside ±band% of its own SMA.
    sma_n = sma(close, config.SCREEN_RANGE_SMA)
    valid = sma_n.notna()
    within = (close[valid] - sma_n[valid]).abs() / sma_n[valid] <= config.SCREEN_RANGE_BAND_PCT
    range_score = float(within.mean()) if valid.any() else 0.0

    # Liquidity — log scale so a $20M and a $2B name aren't worlds apart.
    liq_score = _clamp((math.log10(dollar_volume) - 7.0) / 3.0)  # $10M→0, $10B→1

    score = 100.0 * (
        config.SCREEN_W_TREND     * trend_score +
        config.SCREEN_W_RANGE     * range_score +
        config.SCREEN_W_LIQUIDITY * liq_score
    )

    return {
        "symbol":        recent.index.name or "",
        "score":         round(score, 1),
        "adx":           round(last_adx, 1),
        "range_pct":     round(range_score * 100, 1),
        "dollar_volume": round(dollar_volume, 0),
        "close":         round(float(close.iloc[-1]), 2),
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_screen(top_n: int) -> list[dict]:
    """Run the full funnel and return the top_n ranked sideways candidates."""
    logger.info("=" * 60)
    logger.info("Sideways screener started.")

    universe   = fetch_tradable_universe()
    liquid     = liquidity_prefilter(universe)
    bars_by_sym = fetch_daily_bars_batched(list(liquid), config.SCREEN_LOOKBACK_DAYS)

    results = []
    for symbol, bars in bars_by_sym.items():
        try:
            res = sideways_score(bars, liquid[symbol])
        except Exception as e:
            logger.warning(f"{symbol}: scoring failed: {e}")
            continue
        if res:
            res["symbol"] = symbol   # index name can be lost on .xs; set it explicitly
            results.append(res)

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:top_n]

    # Tag the reported names with sector + asset type (yfinance for stocks),
    # then persist the map so the live bot can enforce the per-sector cap.
    logger.info(f"Classifying sectors for top {len(top)} names…")
    for r in top:
        r["sector"], r["type"] = sectors.classify(r["symbol"])
    sectors.persist(top)

    _write_csv(top)
    _print_table(top)
    logger.info(f"Scored {len(results)} names; reported top {len(top)}.")
    logger.info("=" * 60)
    return top


def _write_csv(rows: list[dict]) -> None:
    headers = ["symbol", "score", "adx", "range_pct", "dollar_volume", "close", "sector", "type"]
    with open(config.SCREEN_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {config.SCREEN_RESULTS_CSV}")


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'Symbol':<8}{'Score':>7}{'ADX':>7}{'Range%':>9}{'$Vol(M)':>11}{'Close':>10}  {'Sector':<22}{'Type'}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['symbol']:<8}{r['score']:>7.1f}{r['adx']:>7.1f}"
            f"{r['range_pct']:>9.1f}{r['dollar_volume'] / 1e6:>11.1f}{r['close']:>10.2f}  "
            f"{r.get('sector', ''):<22}{r.get('type', '')}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Alpaca for sideways stocks.")
    parser.add_argument("--top", type=int, default=config.SCREEN_TOP_N,
                        help="How many top candidates to report.")
    parser.add_argument("--min-volume", type=float, default=None,
                        help="Override dollar-volume floor (e.g. 50000000).")
    parser.add_argument("--min-price", type=float, default=None,
                        help="Override minimum price.")
    args = parser.parse_args()

    if args.min_volume is not None:
        config.SCREEN_MIN_DOLLAR_VOLUME = args.min_volume
    if args.min_price is not None:
        config.SCREEN_MIN_PRICE = args.min_price

    run_screen(args.top)


if __name__ == "__main__":
    main()
