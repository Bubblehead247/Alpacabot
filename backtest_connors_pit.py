"""
backtest_connors_pit.py — Point-in-time (bias-free) test of the Connors config.

The move-toward-Connors ladder (backtest_connors.py) showed a real lift, but on a
FIXED modern universe — so survivorship/selection bias was still in play. This is
the honest gate: it reuses backtest_walkforward.py's point-in-time machinery (the
universe is rebuilt every month from a broad pool using ONLY trailing data, so no
look-ahead and no "picked because it won" bias) and applies Connors' rules.

backtest_walkforward.py screens for SIDEWAYS names — the opposite of what Connors
wants. So here the monthly universe is the top-K most LIQUID names as of that date
(point-in-time dollar volume), and the uptrend requirement is enforced per-trade by
Connors' 200-day SMA filter rather than by the screen.

Everything else matches the live engine: pooled capital, MAX_POSITIONS /
MAX_PER_SECTOR / MAX_POSITION_PCT caps, per-side slippage, gap-through stop fills.

Rows compared (Full / in-sample / out-of-sample on the same point-in-time data):
  - Baseline = our current strategy (sideways screen + RSI crossback entry + stops)
  - Connors entry + 200-SMA filter, with stops
  - Connors, stops dropped
  - Connors, deeper entry (RSI<5), stops dropped

Usage:  python backtest_connors_pit.py
"""
import logging
from collections import defaultdict

import numpy as np

import config
from indicators import sma
from backtest_walkforward import (POOL, prepare, _active_sets, WF_START,
                                  STOP_MULT, REBAL_DAYS, ACTIVE_TOP_K, RANGE_WINDOW)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

SLIP = config.SLIPPAGE_PCT


# ── Point-in-time liquidity screen (Connors-style universe, no sideways bias) ────

def _liq_score(prep, s, r):
    """Trailing average dollar volume as of bar r (uses only data ≤ r)."""
    if np.isnan(prep["sma50"][s][r]):
        return None
    lo = max(0, r - RANGE_WINDOW + 1)
    seg_c, seg_v = prep["c"][s][lo:r + 1], prep["v"][s][lo:r + 1]
    valid = ~np.isnan(seg_c) & ~np.isnan(seg_v)
    if valid.sum() < RANGE_WINDOW // 2:
        return None
    dv = np.nanmean(seg_c[valid] * seg_v[valid])
    return dv if dv >= config.SCREEN_MIN_DOLLAR_VOLUME else None


def _active_sets_liq(prep, lo, hi, rebal_days, top_k):
    """{rebalance_bar: set(top-K most liquid symbols)} over [lo, hi)."""
    out = {}
    for r in range(lo, hi, rebal_days):
        scored = [(s, _liq_score(prep, s, r)) for s in prep["tradeable"]]
        scored = [(s, v) for s, v in scored if v is not None]
        scored.sort(key=lambda x: -x[1])
        out[r] = {s for s, _ in scored[:top_k]}
    return out


# ── Connors engine (shared-capital, point-in-time) ───────────────────────────────

class _P:
    __slots__ = ("sec", "entry_price", "stop", "bar", "shares")
    def __init__(s, sec, entry_price, stop, bar, shares):
        s.sec, s.entry_price, s.stop, s.bar, s.shares = sec, entry_price, stop, bar, shares


def engine(prep, sma200, lo, hi, active_sets, *, thresh, entry_mode,
           use_stop, use_time_stop, use_200sma):
    """One run over [lo, hi). Returns a stats dict. Mirrors the live engine's
    caps/fills; entry/stop behaviour is switchable to compare ours vs Connors."""
    o, h, l, c = prep["o"], prep["h"], prep["l"], prep["c"]
    rsi2, atr14 = prep["rsi2"], prep["atr14"]
    rebal_bars = sorted(active_sets)

    equity = peak = 100_000.0
    max_dd = 0.0
    positions, pending, pnls = {}, [], []
    cur_active, ri = set(), -1

    for i in range(lo, hi):
        while ri + 1 < len(rebal_bars) and rebal_bars[ri + 1] <= i:
            ri += 1; cur_active = active_sets[rebal_bars[ri]]

        # 1. fill pending entries at today's open
        if pending:
            added = defaultdict(int)
            for sym, sig_atr in sorted(pending, key=lambda x: x[0]):
                if len(positions) >= config.MAX_POSITIONS: break
                if sym in positions: continue
                sec = POOL[sym]
                if (sum(1 for q in positions.values() if q.sec == sec) + added[sec]) >= config.MAX_PER_SECTOR:
                    continue
                open_px = o[sym][i]
                if np.isnan(open_px) or open_px <= 0 or np.isnan(sig_atr): continue
                entry_px = open_px * (1 + SLIP)
                if use_stop:
                    stop_dist = STOP_MULT * sig_atr
                    if stop_dist <= 0: continue
                    shares = int(equity * config.RISK_PER_TRADE / stop_dist)
                    stop_px = entry_px - stop_dist
                else:
                    shares, stop_px = 0, -1.0           # sized by notional below; stop never hits
                cap_shares = int(equity * config.MAX_POSITION_PCT / entry_px)
                shares = cap_shares if not use_stop else min(shares, cap_shares)
                if shares <= 0: continue
                positions[sym] = _P(sec, entry_px, stop_px, i, shares)
                added[sec] += 1
        pending = []

        # 2. exits — stop (optional), time stop (optional), RSI exit, window-end mark-out
        for sym in list(positions.keys()):
            pos = positions[sym]
            low_px, open_px, close_px = l[sym][i], o[sym][i], c[sym][i]
            prev_rsi, prev2_rsi = rsi2[sym][i - 1], rsi2[sym][i - 2]
            if np.isnan(low_px) or np.isnan(open_px): continue
            exit_px = None
            if use_stop and low_px <= pos.stop:
                exit_px = min(pos.stop, open_px)                                   # gap-through
            elif use_time_stop and (i - pos.bar) >= config.MAX_HOLD_DAYS:
                exit_px = open_px
            elif (not np.isnan(prev_rsi) and not np.isnan(prev2_rsi)
                  and prev2_rsi >= config.RSI_EXIT_THRESHOLD and prev_rsi < config.RSI_EXIT_THRESHOLD):
                exit_px = open_px                                                  # RSI overbought exit
            elif i == hi - 1 and not np.isnan(close_px):
                exit_px = close_px                                                 # window end: mark out
            if exit_px is not None:
                exit_px *= (1 - SLIP)
                pnl = (exit_px - pos.entry_price) * pos.shares
                equity += pnl; pnls.append(pnl)
                del positions[sym]

        # 3. scan new signals among the active universe — Connors vs crossback entry
        if len(positions) < config.MAX_POSITIONS:
            sig = []
            for sym in cur_active:
                if sym in positions: continue
                cur_rsi, prev_rsi = rsi2[sym][i], rsi2[sym][i - 1]
                cc, aa = c[sym][i], atr14[sym][i]
                if np.isnan(cur_rsi) or np.isnan(aa) or np.isnan(cc): continue
                if entry_mode == "crossback":
                    if np.isnan(prev_rsi) or not (prev_rsi <= thresh and cur_rsi > thresh): continue
                else:                                                              # oversold (Connors)
                    if not (cur_rsi < thresh): continue
                if use_200sma:
                    s200 = sma200[sym][i]
                    if np.isnan(s200) or cc <= s200: continue                      # uptrend only
                sig.append((sym, aa))
            pending = sig

        # 4. MTM + drawdown
        mtm = equity + sum((c[s][i] - p.entry_price) * p.shares
                           for s, p in positions.items() if not np.isnan(c[s][i]))
        peak = max(peak, mtm); max_dd = max(max_dd, (peak - mtm) / peak * 100)

    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    yrs = (prep["dates"][hi - 1] - prep["dates"][lo]).days / 365.25
    cagr = ((equity / 100_000.0) ** (1 / yrs) - 1) * 100 if yrs > 0 and equity > 0 else 0
    return {"pf": round(pf, 2), "cagr": round(cagr, 2), "dd": round(max_dd, 1),
            "win": round(len(wins) / len(pnls) * 100, 1) if pnls else 0, "trades": len(pnls)}


if __name__ == "__main__":
    if config.USE_TREND_FILTER or config.USE_REGIME_FILTER:
        logger.warning("Weekly gate ON — daily-only engine; results approximate.")

    prep = prepare(12)
    n = prep["n"]; mid = (WF_START + n) // 2
    # 200-day SMA per symbol (Connors' trend filter), from the same close arrays.
    sma200 = {s: sma(__import__("pandas").Series(prep["c"][s]), 200).to_numpy(float)
              for s in prep["tradeable"]}

    # Point-in-time universes (built once over the full range; engine activates the
    # rebalance bars inside each window). Sideways = current strategy; liquidity =
    # Connors-style (uptrend handled by the per-trade 200-SMA filter).
    logger.info("Building point-in-time universes (sideways + liquidity)…")
    sets_sw  = _active_sets(prep, WF_START, n, REBAL_DAYS, ACTIVE_TOP_K)
    sets_liq = _active_sets_liq(prep, WF_START, n, REBAL_DAYS, ACTIVE_TOP_K)

    # (label, active_sets, kwargs)
    rows = [
        ("Ours: sideways + crossback (current)", sets_sw,
         dict(thresh=config.RSI_ENTRY_THRESHOLD, entry_mode="crossback",
              use_stop=True, use_time_stop=True, use_200sma=False)),
        ("Connors entry+200SMA, with stops", sets_liq,
         dict(thresh=10.0, entry_mode="oversold",
              use_stop=True, use_time_stop=True, use_200sma=True)),
        ("Connors, stops dropped", sets_liq,
         dict(thresh=10.0, entry_mode="oversold",
              use_stop=False, use_time_stop=False, use_200sma=True)),
        ("Connors RSI<5, stops dropped", sets_liq,
         dict(thresh=5.0, entry_mode="oversold",
              use_stop=False, use_time_stop=False, use_200sma=True)),
    ]

    print(f"\n{'=' * 104}")
    print(f"  POINT-IN-TIME CONNORS TEST (bias-free universe, monthly rebuild, top-{ACTIVE_TOP_K}) | "
          f"{config.MAX_POSITIONS}pos/{config.MAX_PER_SECTOR}sector")
    print(f"  Full {prep['dates'][WF_START].date()}→{prep['dates'][n-1].date()}, "
          f"split @ {prep['dates'][mid].date()}")
    print(f"{'=' * 104}")
    print(f"  {'Config':<40}{'FullPF':>8}{'FullCAGR':>10}{'FullDD':>8}{'Win%':>7}"
          f"{'Trades':>8}{'IS_PF':>7}{'OOS_PF':>8}{'OOS_Trd':>9}")
    print("  " + "-" * 100)
    for label, sets_, kw in rows:
        full = engine(prep, sma200, WF_START, n, sets_, **kw)
        is_r = engine(prep, sma200, WF_START, mid, sets_, **kw)
        oos  = engine(prep, sma200, mid, n, sets_, **kw)
        print(f"  {label:<40}{full['pf']:>8.2f}{full['cagr']:>+9.2f}%{full['dd']:>7.1f}%"
              f"{full['win']:>7.1f}{full['trades']:>8}{is_r['pf']:>7.2f}{oos['pf']:>8.2f}{oos['trades']:>9}")
    print(f"{'=' * 104}")
    print("  Bias-free read: universe rebuilt monthly from trailing data only. OOS_PF = second")
    print("  half, never used to choose. Compare to the fixed-universe ladder (backtest_connors.py).")
