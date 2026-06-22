"""
backtest_connors.py — "Move toward Connors" ladder comparison.

We found (see the strategy review) that our live bot is RSI(2)-flavoured but NOT
Larry Connors' published strategy. It differs in three big ways:
  1. Entry  — we buy the TURN (RSI crosses back above 10); Connors buys while
              STILL oversold (RSI(2) < 10 / < 5).
  2. Regime — we trade with NO trend filter on a sideways-screened universe;
              Connors trades long-only ABOVE the 200-day SMA (dips in uptrends).
  3. Stops  — we use a 2.5×ATR hard stop + 7-day time stop; Connors used NONE
              (his testing found fixed stops hurt this strategy).

This script flips those knobs ONE AT A TIME so we can see how much of the gap
each change closes, and whether it survives an out-of-sample split. It reuses
the grid engine (backtest_grid.py) — same no-look-ahead fills, slippage, and
pooled metrics — so results are directly comparable to the grid run.

Universe = Connors' kind of instruments: liquid ETFs (Test A) and large-caps
(Test B). NOTE: Test B is today's mega-caps, so it carries survivorship bias —
the ETF table is the trustworthy read. We report both.

Usage:  python backtest_connors.py
"""
import logging

import numpy as np
import pandas as pd

from backtest_grid import (UNIVERSE_A, UNIVERSE_B, fetch_daily, Precomp,
                           run_combo, _full_window, _date_window, MIN_YEARS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Each rung: (label, combo, engine-kwargs). combo = (rsi_period, entry_thresh,
# exit_method, sma_filter, vol_filter, atr_filter, mtf). We walk from our live
# character to full Connors, changing exactly one thing per step.
LADDER = [
    ("Ours (live-style)",
     (2, 10, "B", False, False, False, False),
     {"entry_mode": "crossback", "use_stop": True, "use_time_stop": True}),

    ("+ Connors entry (RSI<10)",
     (2, 10, "B", False, False, False, False),
     {"entry_mode": "oversold", "use_stop": True, "use_time_stop": True}),

    ("+ 200-SMA trend filter",
     (2, 10, "B", True, False, False, False),
     {"entry_mode": "oversold", "use_stop": True, "use_time_stop": True}),

    ("+ SMA5 exit (Connors)",
     (2, 10, "A", True, False, False, False),
     {"entry_mode": "oversold", "use_stop": True, "use_time_stop": True}),

    ("+ drop stops = FULL Connors",
     (2, 10, "A", True, False, False, False),
     {"entry_mode": "oversold", "use_stop": False, "use_time_stop": False}),

    ("FULL Connors, RSI<5 (deeper)",
     (2, 5, "A", True, False, False, False),
     {"entry_mode": "oversold", "use_stop": False, "use_time_stop": False}),
]


def _precomps(symbols):
    daily = fetch_daily(symbols)
    out = {}
    for s in symbols:
        if s not in daily:
            continue
        bars = daily[s]
        if (bars.index[-1] - bars.index[0]).days / 365.25 < MIN_YEARS:
            continue
        out[s] = Precomp(s, bars, None)          # no MTF needed here
    return out


def _split_dates(precomps):
    """Calendar midpoint for the in-sample / out-of-sample split."""
    t0 = min(pc.dates[0] for pc in precomps.values())
    tN = max(pc.dates[-1] for pc in precomps.values())
    mid = t0 + (tN - t0) / 2
    return t0, pd.Timestamp(mid), tN


def run_ladder(label, symbols):
    precomps = _precomps(symbols)
    if not precomps:
        logger.warning(f"{label}: no usable tickers."); return
    lo_f, hi_f = _full_window(precomps)
    t0, mid, tN = _split_dates(precomps)
    is_lo, is_hi   = _date_window(precomps, t0, mid)
    oos_lo, oos_hi = _date_window(precomps, mid, tN)

    print(f"\n{'=' * 96}")
    print(f"  MOVE-TOWARD-CONNORS LADDER — {label}  ({len(precomps)} tickers, "
          f"{t0.date()}→{tN.date()}, split @ {mid.date()})")
    print(f"{'=' * 96}")
    print(f"  {'Step':<30}{'FullPF':>8}{'Win%':>7}{'Trades':>8}{'AvgHold':>8}"
          f"{'IS_PF':>8}{'OOS_PF':>8}{'OOS_Trd':>9}")
    print("  " + "-" * 92)
    for step, combo, engine in LADDER:
        full = run_combo(precomps, combo, lo_f, hi_f, engine)
        is_r = run_combo(precomps, combo, is_lo, is_hi, engine)
        oos  = run_combo(precomps, combo, oos_lo, oos_hi, engine)
        print(f"  {step:<30}{full['pf']:>8.2f}{full['win_rate']:>7.1f}{full['trades']:>8}"
              f"{full['avg_hold']:>8.1f}{is_r['pf']:>8.2f}{oos['pf']:>8.2f}{oos['trades']:>9}")
    print(f"{'=' * 96}")
    print("  Reads left→right as we make the strategy more Connors-like. OOS_PF is the")
    print("  honesty check (second half of history, never used to choose anything).")


if __name__ == "__main__":
    run_ladder("Test A — liquid ETFs (survivorship-clean, trust this one)", UNIVERSE_A)
    run_ladder("Test B — large-caps (SURVIVORSHIP-BIASED, read with caution)", UNIVERSE_B)
    print("\n✅ Done. Note: PF here is full-history in-sample unless the OOS column "
          "is used; DD is closed-trade only (see backtest_grid.py caveats).")
