"""
backtest_optimize.py — Search for better entries/exits on the walk-forward engine.

Builds on backtest_walkforward (point-in-time monthly universe, so no selection
bias) and adds:

  ENTRY — "start of the bounce" filters so we don't chase a move that already ran:
    - rsi_entry      : RSI(2) crosses back above this (deeper = more oversold)
    - near_low_pct   : signal close must be within this % of the N-bar low
    - max_chase_pct  : skip the fill if the open gapped more than this above the
                       signal close (we wanted the bounce, not the gap)
  EXIT — a profit target on top of the ATR stop / time stop / RSI exit:
    - profit_target  : exit at entry × (1 + pt); None disables it

A 5-round coordinate search tunes one knob per round on the IN-SAMPLE half and
re-checks OUT-OF-SAMPLE every round (OOS is never used to pick — it's the honesty
check). Objective: in-sample profit factor, requiring >= MIN_TRADES so we don't
"win" with a handful of lucky trades.

Usage:  python backtest_optimize.py
"""
import logging

import numpy as np

import config
from backtest_walkforward import prepare, POOL, _active_sets, WF_START, STOP_MULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

MIN_TRADES = 60   # in-sample trade floor for a parameter set to count


class _P:  # lightweight open position
    __slots__ = ("sec", "entry_price", "stop", "bar", "shares", "target")
    def __init__(s, sec, entry_price, stop, bar, shares, target):
        s.sec, s.entry_price, s.stop, s.bar, s.shares, s.target = sec, entry_price, stop, bar, shares, target


def run_opt(prep, lo, hi, active_sets, p):
    """Run the engine over [lo,hi) with entry/exit params `p`. Returns a stats dict."""
    slip = config.SLIPPAGE_PCT
    o, h, l, c = prep["o"], prep["h"], prep["l"], prep["c"]
    rsi2, atr14 = prep["rsi2"], prep["atr14"]
    lb = p["near_low_lb"]
    rebal_bars = sorted(active_sets)

    equity = peak = 100_000.0
    max_dd = 0.0
    positions, pending, pnls = {}, [], []
    cur_active, ri = set(), -1

    from collections import defaultdict
    for i in range(lo, hi):
        while ri + 1 < len(rebal_bars) and rebal_bars[ri + 1] <= i:
            ri += 1; cur_active = active_sets[rebal_bars[ri]]

        # 1. fill pending entries at today's open
        if pending:
            added = defaultdict(int)
            for sym, sig_rsi, sig_atr, sig_close in sorted(pending, key=lambda x: x[1]):
                if len(positions) >= config.MAX_POSITIONS: break
                if sym in positions: continue
                sec = POOL[sym]
                if (sum(1 for q in positions.values() if q.sec == sec) + added[sec]) >= config.MAX_PER_SECTOR:
                    continue
                open_px = o[sym][i]
                if np.isnan(open_px) or open_px <= 0 or np.isnan(sig_atr): continue
                if open_px > sig_close * (1 + p["max_chase_pct"]): continue          # don't chase the gap
                entry_px = open_px * (1 + slip)
                stop_dist = STOP_MULT * sig_atr
                if stop_dist <= 0: continue
                shares = int(equity * config.RISK_PER_TRADE / stop_dist)
                shares = min(shares, int(equity * config.MAX_POSITION_PCT / entry_px))
                if shares <= 0: continue
                target = entry_px * (1 + p["profit_target"]) if p["profit_target"] else None
                positions[sym] = _P(sec, entry_px, entry_px - stop_dist, i, shares, target)
                added[sec] += 1
        pending = []

        # 2. exits — priority: stop, profit target, time stop, RSI exit
        for sym in list(positions.keys()):
            pos = positions[sym]
            low_px, high_px, open_px = l[sym][i], h[sym][i], o[sym][i]
            prev_rsi, prev2_rsi = rsi2[sym][i - 1], rsi2[sym][i - 2]
            if np.isnan(low_px) or np.isnan(open_px): continue
            exit_px = None
            if low_px <= pos.stop:
                exit_px = min(pos.stop, open_px)                                     # gap-through
            elif pos.target is not None and not np.isnan(high_px) and high_px >= pos.target:
                exit_px = open_px if open_px >= pos.target else pos.target           # gap-up in our favor
            elif (i - pos.bar) >= config.MAX_HOLD_DAYS:
                exit_px = open_px
            elif (not np.isnan(prev_rsi) and not np.isnan(prev2_rsi)
                  and prev2_rsi >= config.RSI_EXIT_THRESHOLD and prev_rsi < config.RSI_EXIT_THRESHOLD):
                exit_px = open_px
            if exit_px is not None:
                exit_px *= (1 - slip)
                pnl = (exit_px - pos.entry_price) * pos.shares
                equity += pnl; pnls.append(pnl)
                del positions[sym]

        # 3. scan new signals among the active universe — start-of-bounce filters
        if len(positions) < config.MAX_POSITIONS:
            sig = []
            for sym in cur_active:
                if sym in positions: continue
                cur_rsi, prev_rsi = rsi2[sym][i], rsi2[sym][i - 1]
                cc, aa = c[sym][i], atr14[sym][i]
                if np.isnan(cur_rsi) or np.isnan(prev_rsi) or np.isnan(aa) or np.isnan(cc): continue
                if not (prev_rsi <= p["rsi_entry"] and cur_rsi > p["rsi_entry"]): continue
                seg_low = l[sym][max(0, i - lb + 1):i + 1]
                rl = np.nanmin(seg_low) if np.any(~np.isnan(seg_low)) else np.nan
                if np.isnan(rl) or cc > rl * (1 + p["near_low_pct"]): continue       # still near the low?
                sig.append((sym, cur_rsi, aa, cc))
            pending = sig

        # 4. MTM + drawdown
        mtm = equity + sum((c[s][i] - pos.entry_price) * pos.shares
                           for s, pos in positions.items() if not np.isnan(c[s][i]))
        peak = max(peak, mtm); max_dd = max(max_dd, (peak - mtm) / peak * 100)

    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else float("inf")
    yrs = (prep["dates"][hi - 1] - prep["dates"][lo]).days / 365.25
    cagr = ((equity / 100_000.0) ** (1 / yrs) - 1) * 100 if yrs > 0 and equity > 0 else 0
    return {"pf": round(pf, 2), "cagr": round(cagr, 2), "dd": round(max_dd, 1),
            "win": round(len(wins) / len(pnls) * 100, 1) if pnls else 0, "trades": len(pnls)}


def _eval(prep, sets_is, sets_oos, lo_is, hi_is, lo_oos, hi_oos, p):
    return run_opt(prep, lo_is, hi_is, sets_is, p), run_opt(prep, lo_oos, hi_oos, sets_oos, p)


def _best(results):
    """Pick by in-sample PF, requiring MIN_TRADES; fall back to most trades."""
    ok = [(val, s_is, s_oos) for val, s_is, s_oos in results if s_is["trades"] >= MIN_TRADES]
    pool = ok if ok else results
    return max(pool, key=lambda r: (r[1]["pf"], r[1]["cagr"]))


if __name__ == "__main__":
    if config.USE_TREND_FILTER or config.USE_REGIME_FILTER:
        logger.warning("Weekly gate ON — daily-only engine; results approximate.")

    prep = prepare(12)
    n = prep["n"]; mid = (WF_START + n) // 2
    # Universe screen is independent of entry/exit params — compute once per half.
    sets_is  = _active_sets(prep, WF_START, mid, 21, 30)
    sets_oos = _active_sets(prep, mid, n, 21, 30)
    ev = lambda p: _eval(prep, sets_is, sets_oos, WF_START, mid, mid, n, p)

    # Baseline: current strategy (no profit target, no bounce filters).
    p = {"rsi_entry": 10.0, "near_low_pct": 0.50, "near_low_lb": 10,
         "max_chase_pct": 0.50, "profit_target": None}
    s_is, s_oos = ev(p)
    print(f"\n{'='*92}\n  ENTRY/EXIT SEARCH — 5 rounds (tune on in-sample, validate out-of-sample)\n{'='*92}")
    print(f"  {'Round / choice':<34}{'IS_PF':>7}{'IS_CAGR':>9}{'IS_DD':>7}{'IS_Trd':>7}"
          f"{'OOS_PF':>8}{'OOS_CAGR':>10}{'OOS_Trd':>8}")
    print("  " + "-" * 88)
    def show(label, s_is, s_oos):
        print(f"  {label:<34}{s_is['pf']:>7.2f}{s_is['cagr']:>+8.2f}%{s_is['dd']:>6.1f}%{s_is['trades']:>7}"
              f"{s_oos['pf']:>8.2f}{s_oos['cagr']:>+9.2f}%{s_oos['trades']:>8}")
    show("baseline (no target/filters)", s_is, s_oos)

    # Round 1 — profit target
    R1 = [0.02, 0.03, 0.05, 0.08, None]
    res = [({**p, "profit_target": pt}, *ev({**p, "profit_target": pt})) for pt in R1]
    best, s_is, s_oos = _best(res); p = best
    show(f"R1 target={p['profit_target']}", s_is, s_oos)

    # Round 2 — near-the-low (start of bounce)
    R2 = [0.02, 0.04, 0.07, 0.50]
    res = [({**p, "near_low_pct": v}, *ev({**p, "near_low_pct": v})) for v in R2]
    best, s_is, s_oos = _best(res); p = best
    show(f"R2 near_low={p['near_low_pct']}", s_is, s_oos)

    # Round 3 — RSI entry depth
    R3 = [5.0, 8.0, 10.0, 15.0]
    res = [({**p, "rsi_entry": v}, *ev({**p, "rsi_entry": v})) for v in R3]
    best, s_is, s_oos = _best(res); p = best
    show(f"R3 rsi_entry={p['rsi_entry']}", s_is, s_oos)

    # Round 4 — anti-chase gap filter
    R4 = [0.005, 0.01, 0.02, 0.50]
    res = [({**p, "max_chase_pct": v}, *ev({**p, "max_chase_pct": v})) for v in R4]
    best, s_is, s_oos = _best(res); p = best
    show(f"R4 max_chase={p['max_chase_pct']}", s_is, s_oos)

    # Round 5 — re-tune the profit target with the refined entry
    R5 = [0.015, 0.02, 0.03, 0.05, None]
    res = [({**p, "profit_target": pt}, *ev({**p, "profit_target": pt})) for pt in R5]
    best, s_is, s_oos = _best(res); p = best
    show(f"R5 target={p['profit_target']}", s_is, s_oos)

    print(f"{'='*92}")
    print(f"  FINAL params: rsi_entry={p['rsi_entry']} near_low={p['near_low_pct']} "
          f"max_chase={p['max_chase_pct']} target={p['profit_target']}")
    print(f"{'='*92}\n")
