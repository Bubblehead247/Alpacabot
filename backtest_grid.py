"""
backtest_grid.py — Parameter-optimization grid search for the Connors RSI(2)
mean-reversion strategy, with regime tagging, walk-forward validation, and
curve-fitting safeguards.

This is a RESEARCH tool, separate from the live strategy. Two things to know up
front before reading the numbers:

  1. ENTRY DIFFERS FROM THE LIVE BOT. This script uses the classic Larry Connors
     setup — a signal fires when RSI(period) closes BELOW the oversold threshold
     (e.g. < 10). The live scanner.py instead enters when RSI(2) crosses back
     ABOVE 10. Results here are not directly comparable to scanner.py; they
     answer "does the textbook Connors setup have an edge on these names?"

  2. THE PF >= 1.5 TARGET IS LIKELY UNREACHABLE HONESTLY. The project's own
     validated baseline (see memory / backtest_optimize.py) tops out at
     out-of-sample PF ~1.14 with a marginal, regime-bound edge. A combo that
     clears 1.5 out-of-sample across walk-forward windows is almost certainly
     overfit — which the curve-fitting safeguards below are built to catch. We
     run the honest search and report whatever actually passes.

DATA
  - Daily bars: yfinance, 2010-01-01 → today (split/dividend adjusted). This is
    the project standard; the Alpaca free/IEX feed only reaches ~2016. Actual
    start date per ticker is logged; tickers with < 2 years are skipped.
  - 15-min bars (for the MTF filter only): Alpaca IEX, which reaches back ~13
    months. The 15-min multi-timeframe confirmation therefore CANNOT be tested
    to 2010 — MTF_ON combos are evaluated ONLY over that recent window and are
    clearly labelled as not comparable to the full-history (MTF_OFF) combos.

GRID  (3·3·5·2·2·2·2 = 720 combos per universe)
  RSI period           : 2, 3, 4
  Entry threshold (<)  : 5, 10, 15
  Exit method          : A=close>SMA5 | B=RSI(2)>65 | C2/C3/C5=fixed 2/3/5-day hold
  200-SMA filter       : on / off   (enter only if close > 200-day SMA)
  Volume filter        : on / off   (today's volume > 1.5× 20-day avg)
  ATR filter           : on / off   (ATR14 > 20-day median ATR)
  MTF (15-min) confirm : on / off   (recent window only — see above)

RULES  (no look-forward bias)
  - Signal at close of bar N → entry at OPEN of bar N+1 (× 1.0005 slippage).
  - Hard stop = entry − 3×ATR(14), fixed at entry, never moved. Intraday fill,
    modelled gap-through (fill at min(stop, open)).
  - Exit condition met at close of bar N → exit at OPEN of bar N+1 (× 0.9995).
  - Max hold 10 days → forced exit at the open of day 11.
  - One open position per ticker. $10k per-ticker sandbox; risk 2% of starting
    capital per trade; never size a position above 20% of current capital.

Usage:
    python backtest_grid.py                 # Test A (ETFs) + Test B (large-caps)
    python backtest_grid.py --universe A    # one universe
    python backtest_grid.py --no-cache      # force fresh downloads
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

import config
from indicators import sma, rsi, atr

warnings.filterwarnings("ignore")  # yfinance / pandas resample chatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# rich is in requirements; fall back to plain print if it is ever missing.
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _console = Console(width=130)   # fixed width so tables don't truncate when piped
    HAVE_RICH = True
except Exception:                                    # pragma: no cover
    _console = None
    HAVE_RICH = False

_alpaca = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

# ── Universes ───────────────────────────────────────────────────────────────────
UNIVERSE_A = ["SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "XLV", "XLU", "GLD", "TLT"]
UNIVERSE_B = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "JNJ", "UNH", "HD",
              "NVDA", "META", "BRK.B", "PG", "V"]

# yfinance uses a dash for class shares; Alpaca uses a dot. Map per source.
def _yf_symbol(sym: str) -> str:
    return sym.replace(".", "-")

# ── Strategy constants (per the spec — independent of config.py live settings) ───
START_DATE      = "2010-01-01"
MIN_YEARS       = 2.0          # skip a ticker with less than this much history
STARTING_CAP    = 10_000.0
RISK_PCT        = 0.02         # risk 2% of STARTING capital per trade
MAX_POS_PCT     = 0.20         # never exceed 20% of current capital in one position
SLIP            = 0.0005       # 0.05% per side
ATR_STOP_MULT   = 3.0          # hard stop = entry − 3×ATR(14)
MAX_HOLD        = 10           # forced exit at open of day 11
VOL_SPIKE_MULT  = 1.5
PF_TARGET       = 1.5

# Grid axes
RSI_PERIODS     = [2, 3, 4]
ENTRY_THRESH    = [5, 10, 15]
EXIT_METHODS    = ["A", "B", "C2", "C3", "C5"]   # A=SMA5, B=RSI(2)>65, Cn=fixed hold
SMA_FILTER      = [True, False]
VOL_FILTER      = [True, False]
ATR_FILTER      = [True, False]
MTF_FILTER      = [True, False]

# Curve-fitting safeguard thresholds
MIN_FULL_TRADES = 50           # full-window trade floor to even be eligible for PASS
WF_MIN_TRADES   = 30           # min trades in a validation window (spec disqualifier)
WF_TRAIN_M      = 12           # walk-forward training window (months)
WF_VALID_M      = 6            # validation window (months)
WF_STEP_M       = 6            # step forward (months)
MAX_DD_LIMIT    = 25.0         # disqualify if max drawdown exceeds this
MAX_WIN_RATE    = 80.0         # disqualify above this (almost certainly overfit)
OOS_DROP_FLAG   = 0.40         # flag if OOS PF drops >40% below IS PF

CACHE_DAILY = "_grid_daily_cache.pkl"
CACHE_MTF   = "_grid_mtf_cache.pkl"


# ── Data fetch ───────────────────────────────────────────────────────────────────

def fetch_daily(symbols: list[str], use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Daily OHLCV per symbol from yfinance, 2010→today. Cached to a pickle."""
    cache = {}
    if use_cache and os.path.exists(CACHE_DAILY):
        with open(CACHE_DAILY, "rb") as f:
            cache = pickle.load(f)

    out, missing = {}, []
    for sym in symbols:
        if sym in cache:
            out[sym] = cache[sym]
        else:
            missing.append(sym)

    end = datetime.now(timezone.utc)
    for sym in missing:
        logger.info(f"Downloading daily {sym} (yfinance)…")
        df = yf.download(_yf_symbol(sym), start=START_DATE, end=end.date().isoformat(),
                         interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            logger.warning(f"{sym}: yfinance returned no data — SKIPPED.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        out[sym] = df[["open", "high", "low", "close", "volume"]].sort_index()
        cache[sym] = out[sym]

    if missing:
        with open(CACHE_DAILY, "wb") as f:
            pickle.dump(cache, f)
    return out


def fetch_mtf(symbols: list[str], use_cache: bool = True) -> dict[str, pd.Series]:
    """
    Per-symbol 15-min RSI(2) signal collapsed to one boolean per calendar day:
    True if RSI(2) on the 15-min chart dipped below 30 at any point that day.

    Source is Alpaca IEX, which only reaches back ~13 months — so this is the
    'recent window' the MTF_ON combos are restricted to.
    """
    cache = {}
    if use_cache and os.path.exists(CACHE_MTF):
        with open(CACHE_MTF, "rb") as f:
            cache = pickle.load(f)

    out, missing = {}, []
    for sym in symbols:
        if sym in cache:
            out[sym] = cache[sym]
        else:
            missing.append(sym)

    end   = datetime.now(timezone.utc) - timedelta(minutes=20)   # respect IEX delay
    start = end - timedelta(days=400)
    for sym in missing:
        logger.info(f"Downloading 15-min {sym} (Alpaca IEX)…")
        try:
            req = StockBarsRequest(symbol_or_symbols=sym,
                                   timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                                   start=start, end=end, feed=DataFeed.IEX)
            bars = _alpaca.get_stock_bars(req).df
        except Exception as e:
            logger.warning(f"{sym}: 15-min fetch failed ({e}) — MTF unavailable.")
            out[sym] = pd.Series(dtype=bool)
            cache[sym] = out[sym]
            continue
        if bars.empty:
            out[sym] = pd.Series(dtype=bool)
            cache[sym] = out[sym]
            continue
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(sym, level="symbol")
        bars = bars.sort_index()
        rsi15 = rsi(bars["close"], 2)
        oversold = rsi15 < 30
        # collapse to one flag per trading day (any 15-min bar oversold)
        daily_flag = oversold.groupby(oversold.index.tz_convert("America/New_York").date).any()
        daily_flag.index = pd.to_datetime(list(daily_flag.index))
        out[sym] = daily_flag
        cache[sym] = out[sym]

    if missing:
        with open(CACHE_MTF, "wb") as f:
            pickle.dump(cache, f)
    return out


# ── Per-ticker precompute ─────────────────────────────────────────────────────────

class Precomp:
    """Indicators + regime + filter flags for one ticker, as numpy arrays."""
    __slots__ = ("symbol", "dates", "open", "high", "low", "close", "volume",
                 "rsi", "atr14", "sma5", "regime", "pass_sma200", "pass_vol",
                 "pass_atr", "mtf_ok", "mtf_start", "n", "warmup", "years")

    def __init__(self, symbol, bars, mtf_daily):
        c = bars["close"]
        self.symbol = symbol
        self.dates  = bars.index
        self.open   = bars["open"].to_numpy(float)
        self.high   = bars["high"].to_numpy(float)
        self.low    = bars["low"].to_numpy(float)
        self.close  = c.to_numpy(float)
        self.volume = bars["volume"].to_numpy(float)
        self.n      = len(bars)
        self.years  = (self.dates[-1] - self.dates[0]).days / 365.25

        # RSI for every period the grid needs (RSI(2) is always present for exit B)
        self.rsi = {p: rsi(c, p).to_numpy(float) for p in set(RSI_PERIODS) | {2}}
        self.atr14 = atr(bars["high"], bars["low"], c, 14).to_numpy(float)
        self.sma5  = sma(c, 5).to_numpy(float)

        sma200    = sma(c, 200)
        sma200_np = sma200.to_numpy(float)
        vol_ma20  = sma(bars["volume"], 20).to_numpy(float)
        atr_med20 = pd.Series(self.atr14).rolling(20).median().to_numpy(float)

        # Regime via 200-day SMA + its 20-day slope (NaN until both exist).
        regime = np.full(self.n, -1, dtype=np.int8)   # 0=UP 1=DOWN 2=SIDE
        for i in range(self.n):
            s = sma200_np[i]
            if np.isnan(s) or i < 20 or np.isnan(sma200_np[i - 20]):
                continue
            up_slope = s > sma200_np[i - 20]
            if self.close[i] > s and up_slope:
                regime[i] = 0
            elif self.close[i] < s and not up_slope:
                regime[i] = 1
            else:
                regime[i] = 2
        self.regime = regime

        # Entry filter flags (independent of the grid combo — precompute once).
        self.pass_sma200 = self.close > sma200_np
        self.pass_vol    = self.volume > VOL_SPIKE_MULT * vol_ma20
        self.pass_atr    = self.atr14 > atr_med20

        # MTF flag aligned onto the daily index; mtf_start = first index covered.
        self.mtf_ok    = np.zeros(self.n, dtype=bool)
        self.mtf_start = self.n
        if mtf_daily is not None and len(mtf_daily):
            flag = mtf_daily.reindex(self.dates.normalize()).fillna(False).to_numpy(bool)
            self.mtf_ok = flag
            covered = np.where([d.normalize() >= mtf_daily.index.min() for d in self.dates])[0]
            if len(covered):
                self.mtf_start = int(covered[0])

        # Warmup: need SMA200 + 20-day slope, ATR median(20), vol MA(20).
        self.warmup = 220


REGIME_NAME = {0: "UPTREND", 1: "DOWNTREND", 2: "SIDEWAYS", -1: "NONE"}


# ── Single-ticker simulation for one combo ─────────────────────────────────────────

def _entry_signal(pc: Precomp, i, period, thresh):
    """Connors oversold signal at close of bar i: RSI(period) < threshold."""
    r = pc.rsi[period][i]
    return (not np.isnan(r)) and r < thresh


def _exit_today(pc: Precomp, i, entry_idx, method):
    """
    Did an exit CONDITION (not the stop) become true at the close of bar i-1,
    so we exit at the open of bar i? Returns (should_exit, reason).
    The hard stop and max-hold are handled by the caller.
    """
    days_held = i - entry_idx
    if method == "A":                                  # close above 5-day SMA
        s = pc.sma5[i - 1]
        if not np.isnan(s) and pc.close[i - 1] > s:
            return True, "sma5_exit"
    elif method == "B":                                # RSI(2) crossed back above 65
        r1, r2 = pc.rsi[2][i - 1], pc.rsi[2][i - 2]
        if not np.isnan(r1) and not np.isnan(r2) and r2 <= 65 and r1 > 65:
            return True, "rsi65_exit"
    else:                                              # C2/C3/C5 fixed hold
        hold = int(method[1:])
        if days_held >= hold:
            return True, "fixed_hold"
    return False, ""


def _entry_fires(pc: Precomp, i, period, thresh, entry_mode):
    """Entry trigger at close of bar i, per mode.
       oversold  = Connors: RSI(period) < thresh (buy while still oversold).
       crossback = our live bot: RSI crossed back UP through thresh (buy the turn)."""
    r, rp = pc.rsi[period][i], pc.rsi[period][i - 1]
    if np.isnan(r) or np.isnan(rp):
        return False
    if entry_mode == "crossback":
        return rp <= thresh and r > thresh
    return r < thresh                                            # "oversold"


def simulate_ticker(pc: Precomp, combo, lo, hi,
                    entry_mode="oversold", use_stop=True, use_time_stop=True):
    """
    Run one ticker over [lo, hi) for one combo. Returns a list of trade dicts.
    Each trade carries pnl_dollars, pnl_pct, hold_days, exit_date, entry regime.

    entry_mode/use_stop/use_time_stop default to the grid's behaviour. The
    Connors comparison (backtest_connors.py) flips them: oversold entry with no
    ATR stop and no time stop (Connors found fixed stops hurt this strategy).
    Any position still open at the last bar is marked out at that close so the
    no-stop variant doesn't silently drop unrealised P&L.
    """
    period, thresh, method, f_sma, f_vol, f_atr, f_mtf = combo

    cap = STARTING_CAP
    risk_dollars = STARTING_CAP * RISK_PCT      # 2% of STARTING capital, fixed
    trades = []
    in_pos = False
    entry_price = stop_price = 0.0
    entry_idx = 0
    shares = 0
    entry_regime = -1

    start = max(lo, pc.warmup)
    if f_mtf:                                   # MTF only valid in the recent window
        start = max(start, pc.mtf_start)

    for i in range(start, hi):
        # ── Exit (if in a position) ──────────────────────────────────────────
        if in_pos:
            days_held = i - entry_idx
            exit_price = exit_reason = None
            if use_stop and pc.low[i] <= stop_price:             # hard stop (gap-through)
                exit_price = min(stop_price, pc.open[i])
                exit_reason = "stop_loss"
            elif use_time_stop and days_held >= MAX_HOLD:        # forced exit, day 11 open
                exit_price, exit_reason = pc.open[i], "time_stop"
            elif i == hi - 1:                                    # window end: mark out
                exit_price, exit_reason = pc.close[i], "window_end"
            else:
                hit, reason = _exit_today(pc, i, entry_idx, method)
                if hit:
                    exit_price, exit_reason = pc.open[i], reason

            if exit_price is not None:
                exit_price *= (1 - SLIP)
                pnl = (exit_price - entry_price) * shares
                cap += pnl
                trades.append({
                    "symbol": pc.symbol,
                    "entry_date": str(pc.dates[entry_idx].date()),
                    "exit_date": str(pc.dates[i].date()),
                    "entry_price": round(float(entry_price), 4),
                    "exit_price": round(float(exit_price), 4),
                    "shares": shares,
                    "pnl_dollars": round(float(pnl), 2),
                    "pnl_pct": round(float((exit_price - entry_price) / entry_price * 100), 2),
                    "hold_days": days_held,
                    "exit_reason": exit_reason,
                    "regime": REGIME_NAME[entry_regime],
                })
                in_pos = False

        # ── Entry (only if flat) ─────────────────────────────────────────────
        if not in_pos and i + 1 < hi:
            if not _entry_fires(pc, i, period, thresh, entry_mode):
                continue
            if f_sma and not pc.pass_sma200[i]:
                continue
            if f_vol and not pc.pass_vol[i]:
                continue
            if f_atr and not pc.pass_atr[i]:
                continue
            if f_mtf and not pc.mtf_ok[i]:
                continue
            cur_atr = pc.atr14[i]
            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            entry_price = pc.open[i + 1] * (1 + SLIP)            # next-day open + slippage
            if use_stop:
                stop_price = entry_price - ATR_STOP_MULT * cur_atr
                stop_dist  = entry_price - stop_price
                if stop_dist <= 0:
                    continue
                shares = int(risk_dollars / stop_dist)
            else:
                # No stop (Connors): size by the notional cap (fixed fraction).
                stop_price = -1.0                                # never triggers
                shares = int(cap * MAX_POS_PCT / entry_price)
            shares = min(shares, int(cap * MAX_POS_PCT / entry_price))   # 20% notional cap
            if shares <= 0:
                continue
            entry_idx    = i + 1
            entry_regime = pc.regime[i]
            in_pos = True

    return trades


# ── Combo-level aggregation across a universe ──────────────────────────────────────

def _pf(trades):
    gw = sum(t["pnl_dollars"] for t in trades if t["pnl_dollars"] > 0)
    gl = abs(sum(t["pnl_dollars"] for t in trades if t["pnl_dollars"] <= 0))
    return (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)


def _win_rate(trades):
    if not trades:
        return 0.0
    return 100.0 * sum(1 for t in trades if t["pnl_dollars"] > 0) / len(trades)


def _max_dd(trades, base_equity):
    """Closed-trade drawdown: cumulative equity in exit-date order."""
    if not trades:
        return 0.0
    ordered = sorted(trades, key=lambda t: t["exit_date"])
    eq = base_equity
    peak = base_equity
    mdd = 0.0
    for t in ordered:
        eq += t["pnl_dollars"]
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    return mdd


def run_combo(precomps: dict, combo, lo_map, hi_map, engine=None):
    """
    Run one combo across all tickers, pooling trades for combo-level metrics.
    lo_map/hi_map: {symbol: (lo_index, hi_index)} so the same combo can be run
    over the full window or any walk-forward sub-window.
    engine: optional dict of simulate_ticker kwargs (entry_mode/use_stop/
    use_time_stop) — used by the Connors comparison; defaults to grid behaviour.
    """
    engine = engine or {}
    all_trades = []
    for sym, pc in precomps.items():
        lo, hi = lo_map[sym], hi_map[sym]
        if hi - lo < 5:
            continue
        all_trades.extend(simulate_ticker(pc, combo, lo, hi, **engine))

    base = STARTING_CAP * len(precomps)
    pf = _pf(all_trades)
    by_regime = defaultdict(list)
    for t in all_trades:
        by_regime[t["regime"]].append(t)

    return {
        "pf": round(pf, 3) if np.isfinite(pf) else 999.0,
        "win_rate": round(_win_rate(all_trades), 1),
        "trades": len(all_trades),
        "avg_hold": round(np.mean([t["hold_days"] for t in all_trades]), 1) if all_trades else 0.0,
        "max_dd": round(_max_dd(all_trades, base), 1),
        "by_regime": {
            rg: {"pf": round(_pf(ts), 2) if np.isfinite(_pf(ts)) else 999.0,
                 "win_rate": round(_win_rate(ts), 1), "trades": len(ts)}
            for rg, ts in by_regime.items()
        },
    }


# ── Index helpers for windows ──────────────────────────────────────────────────────

def _full_window(precomps):
    return ({s: 0 for s in precomps}, {s: pc.n for s, pc in precomps.items()})


def _date_window(precomps, start_dt, end_dt):
    """Map a [start, end) date range to per-symbol index bounds."""
    lo_map, hi_map = {}, {}
    for s, pc in precomps.items():
        idx = pc.dates
        lo_map[s] = int(idx.searchsorted(pd.Timestamp(start_dt)))
        hi_map[s] = int(idx.searchsorted(pd.Timestamp(end_dt)))
    return lo_map, hi_map


# ── Walk-forward validation ────────────────────────────────────────────────────────

def walk_forward(precomps, combo):
    """
    Rolling 12-month train / 6-month validate / 6-month step. Returns a list of
    window dicts (is_pf, oos_pf, win rates, trade counts, drop flag) and the
    average OOS PF across windows.
    """
    starts = [pc.dates[0] for pc in precomps.values()]
    ends   = [pc.dates[-1] for pc in precomps.values()]
    t0, tN = min(starts), max(ends)
    # begin after warmup (~1 year of bars) so the first training window has data
    cursor = (t0 + pd.DateOffset(months=12)).to_pydatetime()

    windows = []
    while True:
        train_start = pd.Timestamp(cursor)
        train_end   = train_start + pd.DateOffset(months=WF_TRAIN_M)
        valid_end   = train_end + pd.DateOffset(months=WF_VALID_M)
        if valid_end > tN:
            break
        is_lo, is_hi   = _date_window(precomps, train_start, train_end)
        oos_lo, oos_hi = _date_window(precomps, train_end, valid_end)
        is_r  = run_combo(precomps, combo, is_lo, is_hi)
        oos_r = run_combo(precomps, combo, oos_lo, oos_hi)
        drop = (is_r["pf"] > 0 and oos_r["pf"] < is_r["pf"] * (1 - OOS_DROP_FLAG))
        windows.append({
            "train": f"{train_start.date()}→{train_end.date()}",
            "valid": f"{train_end.date()}→{valid_end.date()}",
            "is_pf": is_r["pf"], "oos_pf": oos_r["pf"],
            "is_win": is_r["win_rate"], "oos_win": oos_r["win_rate"],
            "is_trades": is_r["trades"], "oos_trades": oos_r["trades"],
            "oos_drop_flag": bool(drop),
        })
        cursor = (train_start + pd.DateOffset(months=WF_STEP_M)).to_pydatetime()

    valid_oos = [w["oos_pf"] for w in windows if np.isfinite(w["oos_pf"]) and w["oos_pf"] < 999]
    avg_oos = round(float(np.mean(valid_oos)), 3) if valid_oos else 0.0
    return windows, avg_oos


# ── Curve-fitting safeguards / status ──────────────────────────────────────────────

def classify(full, combo):
    """Assign PASS / FLAGGED / FAIL and collect warnings for a full-window result."""
    is_mtf = combo[6]
    warnings_ = []
    # ── Hard disqualifiers (a high PF here is almost always a small-sample artifact) ──
    if is_mtf:
        warnings_.append("MTF recent-window only (~13mo) — not comparable / too few trades")
    if full["trades"] < MIN_FULL_TRADES:
        warnings_.append(f"thin sample ({full['trades']} < {MIN_FULL_TRADES} trades)")
    if full["max_dd"] > MAX_DD_LIMIT:
        warnings_.append(f"DD {full['max_dd']}% > {MAX_DD_LIMIT}%")
    if full["win_rate"] > MAX_WIN_RATE:
        warnings_.append(f"win {full['win_rate']}% > {MAX_WIN_RATE}% (overfit-suspect)")
    # Soft warning: profitable in only one regime.
    profitable = [rg for rg, r in full["by_regime"].items()
                  if r["trades"] >= 10 and r["pf"] > 1.0]
    if len(profitable) == 1:
        warnings_.append(f"works only in {profitable[0]}")

    disqualified = (is_mtf or full["trades"] < MIN_FULL_TRADES
                    or full["max_dd"] > MAX_DD_LIMIT or full["win_rate"] > MAX_WIN_RATE)
    if full["pf"] < PF_TARGET or disqualified:
        return "FAIL", warnings_
    return ("FLAGGED" if warnings_ else "PASS"), warnings_


# ── Previous results comparison ─────────────────────────────────────────────────────

def load_previous_best():
    """
    Scan the folder for prior backtest result files and pull the best PF found.
    Returns (description, pf) or (None, None) if this is the baseline run.
    """
    candidates = []
    # Prior grid runs (this tool)
    for fn in sorted(f for f in os.listdir(".")
                     if f.startswith("backtest_results_") and f.endswith(".json")):
        try:
            with open(fn) as f:
                data = json.load(f)
            # Only eligible, full-history combos with a finite PF — never the
            # 999 infinite-PF sentinel from thin 1-trade combos.
            best = max((c["full"]["pf"] for u in data.get("universes", {}).values()
                        for c in u.get("combos", [])
                        if not c["params"]["mtf"] and c["full"]["trades"] >= MIN_FULL_TRADES
                        and c["full"]["pf"] < 999), default=None)
            if best is not None:
                candidates.append((fn, best))
        except Exception:
            continue
    # Prior CSV summaries from the other backtesters
    for fn in ["backtest_summary.csv", "backtest_wf_summary.csv"]:
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn)
                if "profit_factor" in df.columns:
                    finite = df["profit_factor"].replace([np.inf], np.nan).dropna()
                    if len(finite):
                        candidates.append((fn, round(float(finite.max()), 3)))
            except Exception:
                continue
    if not candidates:
        return None, None
    best = max(candidates, key=lambda x: x[1])
    return best[0], best[1]


# ── Output ───────────────────────────────────────────────────────────────────────

def _combo_label(combo):
    period, thresh, method, f_sma, f_vol, f_atr, f_mtf = combo
    exit_lbl = {"A": "SMA5", "B": "RSI>65", "C2": "hold2", "C3": "hold3", "C5": "hold5"}[method]
    return (f"RSI{period}<{thresh}", exit_lbl,
            "on" if f_sma else "off", "on" if f_vol else "off",
            "on" if f_atr else "off", "on" if f_mtf else "off")


def print_header(label, universe, precomps, n_combos, total_trades):
    line = "=" * 78
    print(f"\n{line}")
    print(f"  GRID SEARCH — Test {label}  ({'ETFs' if label == 'A' else 'Large-caps'})")
    print(line)
    print(f"  Run            : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Tickers tested : {len(precomps)}")
    for s, pc in precomps.items():
        print(f"      {s:<6} {pc.dates[0].date()} → {pc.dates[-1].date()}  ({pc.years:.1f}y, {pc.n} bars)")
    print(f"  Combos tested  : {n_combos}")
    print(f"  Total trades   : {total_trades:,}")
    print(line)


def _eligible(combo, full):
    """A combo with a statistically meaningful, full-history sample (not MTF)."""
    return (not combo[6]) and full["trades"] >= MIN_FULL_TRADES


def print_combo_table(results):
    """Top combos by full-window PF — eligible (full-history, >= trade floor) only."""
    elig = [r for r in results if _eligible(r[0], r[1])]
    excluded = len(results) - len(elig)
    ordered = sorted(elig, key=lambda r: r[1]["pf"], reverse=True)[:25]
    print(f"\n  Showing top {len(ordered)} of {len(elig)} eligible combos "
          f"(>= {MIN_FULL_TRADES} full-history trades, MTF excluded). "
          f"{excluded} thin/MTF combos hidden as noise.")
    if HAVE_RICH:
        t = Table(title="Top combos by full-window Profit Factor (eligible only)", box=box.SIMPLE_HEAD)
        for col in ["Signal", "Exit", "SMA", "Vol", "ATR", "MTF", "PF", "Win%",
                    "Hold", "MaxDD", "Trades", "Status"]:
            t.add_column(col, justify="right" if col in
                         ("PF", "Win%", "Hold", "MaxDD", "Trades") else "left")
        for combo, full, status, _ in ordered:
            sig, ex, fs, fv, fa, fm = _combo_label(combo)
            colour = {"PASS": "green", "FLAGGED": "yellow", "FAIL": "dim"}[status]
            t.add_row(sig, ex, fs, fv, fa, fm, f"{full['pf']:.2f}",
                      f"{full['win_rate']:.0f}", f"{full['avg_hold']:.1f}",
                      f"{full['max_dd']:.1f}", f"{full['trades']}",
                      f"[{colour}]{status}[/{colour}]")
        _console.print(t)
    else:                                                       # pragma: no cover
        print(f"\n  {'Signal':<10}{'Exit':<8}{'SMA':<5}{'Vol':<5}{'ATR':<5}{'MTF':<5}"
              f"{'PF':>7}{'Win%':>6}{'Hold':>6}{'MaxDD':>7}{'Trades':>8}  Status")
        for combo, full, status, _ in ordered:
            sig, ex, fs, fv, fa, fm = _combo_label(combo)
            print(f"  {sig:<10}{ex:<8}{fs:<5}{fv:<5}{fa:<5}{fm:<5}{full['pf']:>7.2f}"
                  f"{full['win_rate']:>6.0f}{full['avg_hold']:>6.1f}{full['max_dd']:>7.1f}"
                  f"{full['trades']:>8}  {status}")


def print_regime_table(results):
    shown = [r for r in results if r[2] in ("PASS", "FLAGGED")]
    if not shown:
        print("\n  Regime breakdown: no PASS/FLAGGED combos (PF target not met).")
        return
    print(f"\n  REGIME BREAKDOWN (PASS/FLAGGED combos — PF | Win% | Trades)")
    print(f"  {'Signal':<10}{'Exit':<8}{'UPTREND':>20}{'DOWNTREND':>20}{'SIDEWAYS':>20}")
    for combo, full, status, _ in sorted(shown, key=lambda r: r[1]["pf"], reverse=True):
        sig, ex, *_ = _combo_label(combo)
        cells = []
        for rg in ("UPTREND", "DOWNTREND", "SIDEWAYS"):
            r = full["by_regime"].get(rg)
            cells.append(f"{r['pf']:.2f}/{r['win_rate']:.0f}%/{r['trades']}" if r else "—")
        print(f"  {sig:<10}{ex:<8}{cells[0]:>20}{cells[1]:>20}{cells[2]:>20}")


def print_walkforward(wf_results):
    if not wf_results:
        print("\n  Walk-forward: no combo reached the in-sample PF gate to validate.")
        return
    for combo, windows, avg_oos in wf_results:
        sig, ex, *_ = _combo_label(combo)
        print(f"\n  WALK-FORWARD — {sig} / {ex}   (avg OOS PF = {avg_oos})")
        print(f"  {'Validation window':<26}{'IS_PF':>7}{'OOS_PF':>8}{'IS_Trd':>8}{'OOS_Trd':>9}  Flag")
        for w in windows:
            flag = "!! >40% drop" if w["oos_drop_flag"] else ""
            few  = "  <30 OOS trades" if w["oos_trades"] < WF_MIN_TRADES else ""
            print(f"  {w['valid']:<26}{w['is_pf']:>7.2f}{w['oos_pf']:>8.2f}"
                  f"{w['is_trades']:>8}{w['oos_trades']:>9}  {flag}{few}")


# ── Universe driver ─────────────────────────────────────────────────────────────────

def run_universe(label, symbols, use_cache):
    daily = fetch_daily(symbols, use_cache)
    mtf   = fetch_mtf(symbols, use_cache)

    precomps = {}
    for sym in symbols:
        if sym not in daily:
            logger.warning(f"{sym}: no daily data — skipped.")
            continue
        bars = daily[sym]
        years = (bars.index[-1] - bars.index[0]).days / 365.25
        if years < MIN_YEARS:
            logger.warning(f"{sym}: only {years:.1f}y (< {MIN_YEARS}y) — skipped.")
            continue
        precomps[sym] = Precomp(sym, bars, mtf.get(sym))

    if not precomps:
        logger.error(f"Test {label}: no usable tickers.")
        return None

    combos = [(p, th, m, fs, fv, fa, fm)
              for p in RSI_PERIODS for th in ENTRY_THRESH for m in EXIT_METHODS
              for fs in SMA_FILTER for fv in VOL_FILTER for fa in ATR_FILTER
              for fm in MTF_FILTER]

    lo_full, hi_full = _full_window(precomps)
    results, total_trades = [], 0
    for k, combo in enumerate(combos, 1):
        full = run_combo(precomps, combo, lo_full, hi_full)
        status, warns = classify(full, combo)
        results.append((combo, full, status, warns))
        total_trades += full["trades"]
        if k % 120 == 0:
            logger.info(f"Test {label}: {k}/{len(combos)} combos done…")

    print_header(label, label, precomps, len(combos), total_trades)
    print_combo_table(results)
    print_regime_table(results)

    # Walk-forward: validate combos that cleared the in-sample (full) PF gate.
    # MTF_ON combos only have ~13mo of data, so they can't fill an 18-month
    # window — exclude them and note it.
    gated = [r for r in results if _eligible(r[0], r[1]) and r[1]["pf"] >= PF_TARGET]
    note = ""
    if not gated:
        # Honest fallback: validate the top 3 ELIGIBLE combos by full PF so there's
        # something to see (thin/MTF combos are excluded — their PF is noise).
        gated = sorted((r for r in results if _eligible(r[0], r[1])),
                       key=lambda r: r[1]["pf"], reverse=True)[:3]
        note = (f"  (NOTE: no eligible combo reached full-window PF ≥ {PF_TARGET}; "
                f"validating the top {len(gated)} eligible by PF instead — all below target.)")
    if note:
        print(f"\n{note}")
    wf_results = []
    for combo, full, status, warns in gated[:5]:
        windows, avg_oos = walk_forward(precomps, combo)
        wf_results.append((combo, windows, avg_oos))
    print_walkforward(wf_results)

    return {
        "label": label,
        "tickers": {s: {"start": str(pc.dates[0].date()), "end": str(pc.dates[-1].date()),
                        "years": round(pc.years, 1), "bars": pc.n} for s, pc in precomps.items()},
        "n_combos": len(combos),
        "total_trades": total_trades,
        "combos": [{"params": dict(zip(
                        ["rsi_period", "entry_thresh", "exit", "sma_filter",
                         "vol_filter", "atr_filter", "mtf"], combo)),
                    "full": full, "status": status, "warnings": warns}
                   for combo, full, status, warns in results],
        "walk_forward": [{"params": dict(zip(
                            ["rsi_period", "entry_thresh", "exit", "sma_filter",
                             "vol_filter", "atr_filter", "mtf"], combo)),
                          "avg_oos_pf": avg_oos, "windows": windows}
                         for combo, windows, avg_oos in wf_results],
    }


def print_recommendations(all_universes):
    print(f"\n{'=' * 78}\n  FINAL RECOMMENDATIONS\n{'=' * 78}")
    # Rank by avg OOS PF across walk-forward, then full PF.
    ranked = []
    for u in all_universes:
        wf_lookup = {tuple(w["params"].values()):
                     (w["avg_oos_pf"],
                      min((win["oos_trades"] for win in w["windows"]), default=0))
                     for w in u["walk_forward"]}
        for c in u["combos"]:
            # Only recommend statistically meaningful, full-history combos.
            if c["params"]["mtf"] or c["full"]["trades"] < MIN_FULL_TRADES:
                continue
            key = tuple(c["params"].values())
            oos, min_oos_trades = wf_lookup.get(key, (None, 0))
            ranked.append((u["label"], c, oos, min_oos_trades))
    ranked.sort(key=lambda x: (x[2] if x[2] is not None else -1, x[1]["full"]["pf"]), reverse=True)

    top = ranked[:5]
    if not top:
        print("  No eligible combos to recommend.")
        return
    for i, (lab, c, oos, min_oos_trades) in enumerate(top, 1):
        p = c["params"]
        exit_lbl = {"A": "close>SMA5", "B": "RSI(2)>65", "C2": "2-day hold",
                    "C3": "3-day hold", "C5": "5-day hold"}[p["exit"]]
        regimes = c["full"]["by_regime"]
        best_rg = max((rg for rg in regimes if regimes[rg]["trades"] >= 10),
                      key=lambda rg: regimes[rg]["pf"], default=None)
        print(f"\n  #{i}  Test {lab}: RSI({p['rsi_period']}) < {p['entry_thresh']}, "
              f"exit {exit_lbl}")
        filt = [n for n, on in [("200-SMA", p["sma_filter"]), ("volume", p["vol_filter"]),
                                ("ATR", p["atr_filter"]), ("15m-MTF", p["mtf"])] if on]
        print(f"        Filters: {', '.join(filt) if filt else 'none'}")
        print(f"        Full-window PF {c['full']['pf']}, win {c['full']['win_rate']}%, "
              f"DD {c['full']['max_dd']}%, {c['full']['trades']} trades")
        print(f"        Best regime: {best_rg or 'n/a'}"
              + (f" (PF {regimes[best_rg]['pf']})" if best_rg else ""))
        thin_wf = (oos is not None and min_oos_trades < WF_MIN_TRADES)
        print(f"        Avg OOS PF (walk-forward): {oos if oos is not None else 'not validated'}"
              + (f"  [min {min_oos_trades} trades/window < {WF_MIN_TRADES} — fails sample rule]"
                 if thin_wf else ""))
        passed = (c["status"] == "PASS" and oos is not None and oos >= PF_TARGET
                  and not thin_wf)
        print(f"        Curve-fit checks: {c['status']}"
              + (f" — warnings: {'; '.join(c['warnings'])}" if c["warnings"] else ""))
        verdict = ("CONSIDER for live testing" if passed
                   else "DO NOT trade live — below PF target / unvalidated")
        print(f"        Verdict: {verdict}")


def main():
    ap = argparse.ArgumentParser(description="Connors RSI(2) parameter grid search")
    ap.add_argument("--universe", choices=["A", "B", "both"], default="both")
    ap.add_argument("--no-cache", action="store_true", help="force fresh downloads")
    args = ap.parse_args()
    use_cache = not args.no_cache

    prev_file, prev_pf = load_previous_best()

    universes = []
    if args.universe in ("A", "both"):
        r = run_universe("A", UNIVERSE_A, use_cache)
        if r:
            universes.append(r)
    if args.universe in ("B", "both"):
        r = run_universe("B", UNIVERSE_B, use_cache)
        if r:
            universes.append(r)

    if not universes:
        logger.error("No results produced.")
        return

    # ── Previous vs current comparison ──────────────────────────────────────
    cur_best = max((c["full"]["pf"] for u in universes for c in u["combos"]
                    if not c["params"]["mtf"] and c["full"]["trades"] >= MIN_FULL_TRADES
                    and np.isfinite(c["full"]["pf"]) and c["full"]["pf"] < 999), default=0.0)
    print(f"\n{'=' * 78}\n  PREVIOUS vs CURRENT (best Profit Factor)\n{'=' * 78}")
    if prev_pf is None:
        print("  No prior result files found — this is the BASELINE grid run.")
    else:
        print(f"  Previous best : PF {prev_pf}  (from {prev_file})")
        print(f"  Current best  : PF {round(cur_best, 3)}  (this grid run, full-window)")
    print("  Reminder: the project's validated out-of-sample baseline is PF ~1.14")
    print("            (backtest_optimize.py). Full-window PF above is in-sample.")

    print_recommendations(universes)

    # ── Save ─────────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "run_timestamp": datetime.now().isoformat(),
        "data": {"daily_source": "yfinance", "start": START_DATE,
                 "mtf_source": "alpaca_iex_15min_recent_window_only"},
        "entry_note": "Connors oversold (RSI < threshold) — differs from live scanner.py crossback-above entry",
        "previous_best": {"file": prev_file, "pf": prev_pf},
        "universes": {u["label"]: u for u in universes},
    }
    fn = f"backtest_results_{stamp}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✅ Full results saved → {fn}")


if __name__ == "__main__":
    main()
