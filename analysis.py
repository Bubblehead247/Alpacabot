"""
analysis.py — Robustness & sensitivity testing for the Mean Reversion strategy.

Reuses the production backtest engine (backtest.py) so results are apples-to-apples
with the live rules. Price history is fetched ONCE per symbol and cached to disk;
every parameter variant then re-runs the fast numpy engine on the cached bars.

Two studies:

  SENSITIVITY  — One-at-a-time (OAT) parameter sweeps around the live baseline.
                 A robust parameter shows a smooth plateau, not a lone spike.
                 (overfit edges collapse the moment you nudge the knob.)

  ROBUSTNESS   — Does the edge survive conditions we didn't fit to?
                   • In-sample vs out-of-sample (time split)
                   • Market-regime buckets (bull / COVID / 2022 bear / recovery)
                   • Leave-one-symbol-out (is the edge concentrated in one ETF?)
                   • Transaction-cost / slippage haircut
                   • Monte-Carlo trade-order shuffle (drawdown distribution)

Usage:
    python analysis.py                  # full study, 12yr, config.SYMBOLS
    python analysis.py --years 12
    python analysis.py --refresh        # ignore the bar cache and refetch

Output: console tables + CSVs (sensitivity_*.csv, robustness_*.csv).
"""

import argparse
import pickle
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import backtest as bt

CACHE = Path(__file__).resolve().parent / "_bars_cache.pkl"


# ── Data ────────────────────────────────────────────────────────────────────

def load_all_bars(symbols, years, refresh=False):
    """Fetch every symbol once (12yr daily). Cache to a pickle keyed by request."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365 + 60)
    key   = (tuple(symbols), start.date().isoformat(), end.date().isoformat())

    if CACHE.exists() and not refresh:
        cached = pickle.loads(CACHE.read_bytes())
        if cached.get("key") == key:
            print(f"Loaded {len(cached['bars'])} symbols from cache "
                  f"({key[1]} → {key[2]}).")
            return cached["bars"]

    print(f"Fetching {len(symbols)} symbols from yfinance ({start.date()} → {end.date()})...")
    bars = {}
    for sym in symbols:
        try:
            df = bt.fetch_history(sym, start, end, source="yfinance")
        except Exception as e:
            print(f"  {sym}: fetch failed — {e}")
            continue
        if len(df) < 1050:
            print(f"  {sym}: only {len(df)} bars (<1050) — skipped.")
            continue
        bars[sym] = df
        print(f"  {sym}: {len(df)} bars")

    CACHE.write_bytes(pickle.dumps({"key": key, "bars": bars}))
    print(f"Cached → {CACHE.name}")
    return bars


# ── Engine driver with parameter overrides ───────────────────────────────────

# Every tunable the engine / precompute reads from config at runtime.
_OVERRIDABLE = (
    "RSI_PERIOD", "RSI_ENTRY_THRESHOLD", "RSI_EXIT_THRESHOLD",
    "SMA_DAILY", "SMA_WEEKLY_FAST", "SMA_WEEKLY_SLOW",
    "ATR_PERIOD", "VOLUME_MA_PERIOD", "VOLUME_SPIKE_MULT", "USE_VOLUME_FILTER",
    "RISK_PER_TRADE", "MAX_HOLD_DAYS", "SLIPPAGE_PCT",
    "USE_REGIME_FILTER", "REGIME_ADX_PERIOD", "REGIME_ADX_MIN", "REGIME_ADX_MAX",
)


def run_universe(bars, stop_mult, overrides=None, equity=100_000.0):
    """
    Run the production engine across all symbols at one parameter set.
    Returns (aggregate_metrics_dict, pooled_trades_list).

    config is monkeypatched for the duration and restored afterward, so callers
    can sweep freely without side effects.
    """
    overrides = overrides or {}
    saved = {k: getattr(config, k) for k in _OVERRIDABLE}
    for k, v in overrides.items():
        setattr(config, k, v)

    results, pooled = [], []
    try:
        for sym, df in bars.items():
            data = bt.precompute_symbol(sym, df)
            r = bt.backtest_symbol(data, stop_mult, equity)
            results.append(r)
            pooled.extend(r.trades)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)

    return aggregate(results, pooled, equity), pooled


def aggregate(results, trades, equity=100_000.0):
    """Portfolio-level summary. Each symbol is an independent equal-capital book,
    so headline figures are equal-weight means across symbols; trade-quality
    figures (PF, win rate) are pooled across every trade."""
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers  = [t for t in trades if t.pnl_dollars <= 0]
    gross_w = sum(t.pnl_dollars for t in winners)
    gross_l = abs(sum(t.pnl_dollars for t in losers))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "symbols":       len(results),
        "trades":        len(trades),
        "win_rate":      round(len(winners) / len(trades) * 100, 1) if trades else 0.0,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        "avg_trade_pct": round(mean([t.pnl_pct for t in trades]), 3),
        "mean_cagr":     round(mean([r.cagr_pct for r in results]), 2),
        "mean_return":   round(mean([r.total_return_pct for r in results]), 2),
        "mean_maxdd":    round(mean([r.max_drawdown_pct for r in results]), 2),
        # equal-weight book: total dollars made across the N independent books
        "total_pnl":     round(sum(t.pnl_dollars for t in trades), 2),
    }


def fmt_row(label, m, width=22):
    return (f"  {label:<{width}}"
            f"{m['trades']:>7}{m['win_rate']:>8.1f}{m['profit_factor']:>8.2f}"
            f"{m['avg_trade_pct']:>9.3f}{m['mean_cagr']:>9.2f}"
            f"{m['mean_return']:>10.2f}{m['mean_maxdd']:>9.2f}")


HDR = (f"  {'variant':<22}{'trades':>7}{'win%':>8}{'PF':>8}"
       f"{'avgT%':>9}{'CAGR%':>9}{'ret%':>10}{'maxDD%':>9}")


# ── Sensitivity ───────────────────────────────────────────────────────────────

def sensitivity_study(bars, base_stop):
    """OAT sweeps. Stop multiplier is the engine's stop_mult arg; everything else
    is a config override."""
    print("\n" + "=" * 96)
    print(f"  SENSITIVITY  —  one-at-a-time sweeps  (baseline: stop={base_stop}×, "
          f"RSI in/out={config.RSI_ENTRY_THRESHOLD:.0f}/{config.RSI_EXIT_THRESHOLD:.0f}, "
          f"SMA_d={config.SMA_DAILY}, hold={config.MAX_HOLD_DAYS})")
    print("=" * 96)

    sweeps = {
        "stop_mult (×ATR)":  ("__stop__",            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
        "RSI_ENTRY_THRESH":  ("RSI_ENTRY_THRESHOLD", [5, 8, 10, 12, 15, 20]),
        "RSI_EXIT_THRESH":   ("RSI_EXIT_THRESHOLD",  [60, 65, 70, 75, 80, 90]),
        "RSI_PERIOD":        ("RSI_PERIOD",          [2, 3, 4, 5]),
        "SMA_DAILY":         ("SMA_DAILY",           [20, 30, 50, 100, 150, 200]),
        "ATR_PERIOD":        ("ATR_PERIOD",          [7, 10, 14, 20, 30]),
        "MAX_HOLD_DAYS":     ("MAX_HOLD_DAYS",       [3, 5, 7, 10, 14, 21]),
        "RISK_PER_TRADE":    ("RISK_PER_TRADE",      [0.005, 0.01, 0.02, 0.03]),
    }

    rows = []
    for title, (param, values) in sweeps.items():
        print(f"\n  ► {title}")
        print(HDR)
        for v in values:
            if param == "__stop__":
                m, _ = run_universe(bars, v)
            else:
                m, _ = run_universe(bars, base_stop, overrides={param: v})
            tag = f"{v}"
            marker = "  ← baseline" if _is_baseline(param, v, base_stop) else ""
            print(fmt_row(f"{tag}{marker}", m))
            rows.append({"parameter": title, "value": v, **m})
    _write_csv("sensitivity_results.csv", rows)
    return rows


def _is_baseline(param, v, base_stop):
    baseline = {
        "__stop__": base_stop,
        "RSI_ENTRY_THRESHOLD": config.RSI_ENTRY_THRESHOLD,
        "RSI_EXIT_THRESHOLD":  config.RSI_EXIT_THRESHOLD,
        "RSI_PERIOD":          config.RSI_PERIOD,
        "SMA_DAILY":           config.SMA_DAILY,
        "ATR_PERIOD":          config.ATR_PERIOD,
        "MAX_HOLD_DAYS":       config.MAX_HOLD_DAYS,
        "RISK_PER_TRADE":      config.RISK_PER_TRADE,
    }
    return param in baseline and abs(float(baseline[param]) - float(v)) < 1e-9


# ── Robustness ────────────────────────────────────────────────────────────────

def _bucket_metrics(trades, label, window_years=None):
    """Trade-quality metrics for a subset of trades (bucketed by entry date)."""
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers  = [t for t in trades if t.pnl_dollars <= 0]
    gw = sum(t.pnl_dollars for t in winners)
    gl = abs(sum(t.pnl_dollars for t in losers))
    n  = len(trades)
    return {
        "bucket":        label,
        "trades":        n,
        "win_rate":      round(len(winners) / n * 100, 1) if n else 0.0,
        "profit_factor": round(gw / gl, 2) if gl > 0 else (float("inf") if gw > 0 else 0.0),
        "avg_trade_pct": round(sum(t.pnl_pct for t in trades) / n, 3) if n else 0.0,
        "total_pnl":     round(sum(t.pnl_dollars for t in trades), 2),
    }


def robustness_study(bars, base_stop):
    print("\n" + "=" * 96)
    print(f"  ROBUSTNESS  —  baseline stop={base_stop}×ATR, live params")
    print("=" * 96)

    base_m, trades = run_universe(bars, base_stop)
    print("\n  Baseline (full sample):")
    print(HDR)
    print(fmt_row("full", base_m))

    rows = []

    # 1. In-sample / out-of-sample split by entry date --------------------------
    dates = sorted(datetime.fromisoformat(t.entry_date) for t in trades)
    mid = dates[len(dates) // 2].date().isoformat()
    is_tr  = [t for t in trades if t.entry_date <  mid]
    oos_tr = [t for t in trades if t.entry_date >= mid]
    print(f"\n  ► In-sample / Out-of-sample  (split @ {mid})")
    print(f"  {'bucket':<22}{'trades':>7}{'win%':>8}{'PF':>8}{'avgT%':>9}{'totalPnl':>12}")
    for lbl, sub in (("in-sample", is_tr), ("out-of-sample", oos_tr)):
        b = _bucket_metrics(sub, lbl)
        print(f"  {b['bucket']:<22}{b['trades']:>7}{b['win_rate']:>8.1f}"
              f"{b['profit_factor']:>8.2f}{b['avg_trade_pct']:>9.3f}{b['total_pnl']:>12,.0f}")
        rows.append({"study": "in_out_sample", **b})

    # 2. Market regimes ---------------------------------------------------------
    regimes = [
        ("2018 chop",        "2018-01-01", "2019-01-01"),
        ("2019 bull",        "2019-01-01", "2020-02-19"),
        ("COVID crash/snap", "2020-02-19", "2020-12-31"),
        ("2021 bull",        "2021-01-01", "2022-01-01"),
        ("2022 bear",        "2022-01-01", "2023-01-01"),
        ("2023-24 recovery", "2023-01-01", "2025-01-01"),
        ("2025+ recent",     "2025-01-01", "2027-01-01"),
    ]
    print(f"\n  ► Market regimes  (trades bucketed by entry date)")
    print(f"  {'regime':<22}{'trades':>7}{'win%':>8}{'PF':>8}{'avgT%':>9}{'totalPnl':>12}")
    for lbl, s, e in regimes:
        sub = [t for t in trades if s <= t.entry_date < e]
        b = _bucket_metrics(sub, lbl)
        print(f"  {b['bucket']:<22}{b['trades']:>7}{b['win_rate']:>8.1f}"
              f"{b['profit_factor']:>8.2f}{b['avg_trade_pct']:>9.3f}{b['total_pnl']:>12,.0f}")
        rows.append({"study": "regime", **b})

    # 3. Leave-one-symbol-out ---------------------------------------------------
    print(f"\n  ► Leave-one-symbol-out  (does one ETF carry the edge?)")
    print(HDR)
    full_pf = base_m["profit_factor"]
    loo_rows = []
    for drop in bars:
        subset = {k: v for k, v in bars.items() if k != drop}
        m, _ = run_universe(subset, base_stop)
        loo_rows.append((drop, m))
        rows.append({"study": "leave_one_out", "bucket": f"ex-{drop}", **m})
    # show the most impactful drops (largest PF swing vs full)
    loo_rows.sort(key=lambda km: km[1]["profit_factor"])
    for drop, m in loo_rows:
        delta = m["profit_factor"] - full_pf
        print(fmt_row(f"ex-{drop} (ΔPF {delta:+.2f})", m))

    # 4. Transaction-cost / slippage sweep (engine-driven) ----------------------
    # Re-runs the engine at each slippage level so position sizing, stop levels
    # and fills all respond — a true cost curve, not a post-hoc trade haircut.
    print(f"\n  ► Slippage sensitivity  (engine-modeled, per-side)")
    print(HDR)
    for cost_bps in (0, 5, 10, 20, 50):
        m, _ = run_universe(bars, base_stop, overrides={"SLIPPAGE_PCT": cost_bps / 10_000.0})
        mark = "  ← live" if abs(cost_bps / 10_000.0 - config.SLIPPAGE_PCT) < 1e-9 else ""
        print(fmt_row(f"{cost_bps} bps/side{mark}", m))
        rows.append({"study": "slippage", "bucket": f"{cost_bps}bps", **m})

    _write_csv("robustness_buckets.csv", rows)

    # 5. Monte-Carlo trade-order shuffle ----------------------------------------
    monte_carlo(trades, base_m, n_books=base_m["symbols"])


def monte_carlo(trades, base_m, n_books, iters=5000, seed=42):
    """
    Bootstrap-resample the realized trades (sample N with replacement) many times
    and rebuild a combined equity curve. Gives a distribution of BOTH final return
    and max drawdown — i.e. "if the next ~600 trades are drawn from the same
    distribution, what range of outcomes should I expect?"

    (Note: a pure order *shuffle* would leave total return constant — additive
    P&L is order-invariant — so we resample with replacement instead, which
    perturbs the trade mix and reveals real outcome dispersion.)

    Combined book = n_books independent $100k accounts pooled ($100k × n_books).
    """
    rng = random.Random(seed)
    pnls = [t.pnl_dollars for t in trades]
    if not pnls:
        print("\n  ► Monte Carlo: no trades.")
        return

    start_cap = 100_000.0 * n_books
    n = len(pnls)
    finals, maxdds = [], []
    for _ in range(iters):
        seq = [rng.choice(pnls) for _ in range(n)]   # resample with replacement
        eq = start_cap
        peak = start_cap
        mdd = 0.0
        for p in seq:
            eq += p
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd
        finals.append((eq - start_cap) / start_cap * 100)
        maxdds.append(mdd)

    def pct(xs, q):
        return round(statistics.quantiles(xs, n=100)[q - 1], 2)

    print(f"\n  ► Monte Carlo  ({iters} bootstrap resamples, combined ${start_cap:,.0f} book)")
    print(f"    Total return %   median {statistics.median(finals):>7.2f}   "
          f"p5 {pct(finals,5):>7.2f}   p95 {pct(finals,95):>7.2f}   "
          f"worst {min(finals):>7.2f}")
    print(f"    Max drawdown %   median {statistics.median(maxdds):>7.2f}   "
          f"p95 {pct(maxdds,95):>7.2f}   p99 {pct(maxdds,99):>7.2f}   "
          f"worst {max(maxdds):>7.2f}")
    pos = sum(1 for f in finals if f > 0) / len(finals) * 100
    print(f"    Profitable paths {pos:.1f}%")

    # Bootstrap the per-trade edge: is mean trade return distinguishable from 0?
    pct_returns = [t.pnl_pct for t in trades]
    boot_means = []
    for _ in range(iters):
        sample = [rng.choice(pct_returns) for _ in range(len(pct_returns))]
        boot_means.append(sum(sample) / len(sample))
    lo = round(statistics.quantiles(boot_means, n=100)[2], 3)   # p2.5-ish (p3)
    hi = round(statistics.quantiles(boot_means, n=100)[96], 3)  # p97
    print(f"    Bootstrap mean trade %: {statistics.mean(boot_means):.3f}  "
          f"(95% CI ≈ [{lo}, {hi}])  "
          f"{'edge > 0 ✓' if lo > 0 else 'CI includes 0 ✗'}")


# ── Regime study (step 1: measure before filtering) ───────────────────────────

def _weekly_regime_daily(df, er_period, adx_period):
    """ADX + Efficiency Ratio on W-FRI weekly bars, forward-filled onto the daily
    index (same alignment the weekly SMA gate uses — a daily bar only ever sees
    the most recently completed week, so no lookahead)."""
    from indicators import efficiency_ratio, adx as adx_ind
    wk = pd.DataFrame({
        "high":  df["high"].resample("W-FRI").max(),
        "low":   df["low"].resample("W-FRI").min(),
        "close": df["close"].resample("W-FRI").last(),
    }).dropna()
    er = efficiency_ratio(wk["close"], er_period).reindex(df.index, method="ffill")
    ax = adx_ind(wk["high"], wk["low"], wk["close"], adx_period).reindex(df.index, method="ffill")
    return er.to_numpy(dtype=float), ax.to_numpy(dtype=float)


def _tag_trades_with_regime(bars, trades, er_period=10, adx_period=14):
    """
    Attach DAILY and WEEKLY trend regime (Efficiency Ratio + ADX) to each trade,
    measured as of the SIGNAL bar — the trading day BEFORE entry. (Entry fills at
    the next open, so the decision is made on the prior close; using that bar
    avoids lookahead.)

    Returns list of (trade, regime_dict); trades whose signal-bar regime is NaN
    (warmup) are dropped. regime_dict keys: ER_d, ADX_d, ER_w, ADX_w.
    """
    from indicators import efficiency_ratio, adx as adx_ind

    cache = {}
    for sym, df in bars.items():
        er_d = efficiency_ratio(df["close"], er_period).to_numpy(dtype=float)
        ax_d = adx_ind(df["high"], df["low"], df["close"], adx_period).to_numpy(dtype=float)
        er_w, ax_w = _weekly_regime_daily(df, er_period, adx_period)
        pos = {ts.date().isoformat(): i for i, ts in enumerate(df.index)}
        cache[sym] = (er_d, ax_d, er_w, ax_w, pos)

    tagged = []
    for t in trades:
        er_d, ax_d, er_w, ax_w, pos = cache[t.symbol]
        p = pos.get(t.entry_date)
        if p is None or p == 0:
            continue
        s = p - 1                       # signal bar = day before entry
        reg = {"ER_d": er_d[s], "ADX_d": ax_d[s], "ER_w": er_w[s], "ADX_w": ax_w[s]}
        if any(np.isnan(v) for v in reg.values()):
            continue
        tagged.append((t, reg))
    return tagged


def _trip(trades):
    """Compact 'n/PF/avgT%' cell for a bucket."""
    b = _bucket_metrics(trades, "")
    pf = "inf" if b["profit_factor"] == float("inf") else f"{b['profit_factor']:.2f}"
    return f"{b['trades']}/{pf}/{b['avg_trade_pct']:+.3f}"


def _validate_table(title, buckets, mid, study, rows):
    """buckets: list of (label, [trades]). Prints full / in-sample / OOS triplets
    so we can see whether a bucket's edge holds in BOTH halves of history."""
    print(f"\n  ► {title}")
    print(f"  {'bucket':<22}{'full n/PF/avgT':>20}{'in-sample':>20}{'out-of-sample':>20}")
    for label, sub in buckets:
        isamp = [t for t in sub if t.entry_date <  mid]
        oos   = [t for t in sub if t.entry_date >= mid]
        print(f"  {label:<22}{_trip(sub):>20}{_trip(isamp):>20}{_trip(oos):>20}")
        for tag, grp in (("full", sub), ("in_sample", isamp), ("oos", oos)):
            b = _bucket_metrics(grp, label)
            rows.append({"study": study, "split": tag, **b})


def regime_study(bars, base_stop, er_period=10, adx_period=14):
    """
    Step 1: does the edge live in a particular trend regime — and does it hold
    OUT of sample? Bucket realized trades by DAILY and WEEKLY trend strength
    (measured at the signal bar), and split each bucket in/out of sample.
    We MEASURE here — no rule change yet.
    """
    print("\n" + "=" * 96)
    print(f"  REGIME STUDY  —  daily & weekly trend strength at the signal bar, "
          f"validated in/out of sample")
    print(f"  (ER period={er_period}, ADX period={adx_period}, stop={base_stop}×)")
    print("=" * 96)

    _, trades = run_universe(bars, base_stop)
    tagged = _tag_trades_with_regime(bars, trades, er_period, adx_period)
    ds = sorted(t.entry_date for t, _ in tagged)
    mid = ds[len(ds) // 2]
    print(f"\n  {len(tagged)} of {len(trades)} trades tagged (rest dropped to warmup NaN).")
    print(f"  In/out-of-sample split @ {mid}  "
          f"(IS: {sum(d < mid for d in ds)} trades, OOS: {sum(d >= mid for d in ds)}).")
    print(f"  Read each cell as  n/PF/avgT% .  A robust bucket wins in BOTH halves.")

    rows = []
    for tf, tflabel in (("d", "DAILY"), ("w", "WEEKLY")):
        # ADX textbook thresholds — the actionable, comparable cut --------------
        b_lo, b_mid, b_hi = [], [], []
        for t, reg in tagged:
            a = reg[f"ADX_{tf}"]
            bucket = b_lo if a < 20 else b_mid if a < 25 else b_hi
            bucket.append(t)
        _validate_table(
            f"{tflabel} ADX threshold",
            [("ADX <20 (no trend)", b_lo), ("ADX 20–25 (transit.)", b_mid),
             ("ADX ≥25 (trending)", b_hi)],
            mid, f"ADX_{tf}_threshold", rows)

        # Efficiency Ratio terciles — fixed full-sample cuts (comparable halves) -
        er_vals = [reg[f"ER_{tf}"] for _, reg in tagged]
        c_lo, c_hi = np.nanpercentile(er_vals, [33, 67])
        g1, g2, g3 = [], [], []
        for t, reg in tagged:
            e = reg[f"ER_{tf}"]
            (g1 if e <= c_lo else g2 if e <= c_hi else g3).append(t)
        _validate_table(
            f"{tflabel} ER tercile  (cuts {c_lo:.2f} / {c_hi:.2f})",
            [("ER low (chop)", g1), ("ER mid", g2), ("ER high (trend)", g3)],
            mid, f"ER_{tf}_tercile", rows)

    _write_csv("regime_buckets.csv", rows)


def filter_comparison(bars, base_stop):
    """
    Measure the weekly-ADX regime filter end-to-end: an OFF-vs-ON headline at live
    slippage, then the FULL robustness+slippage suite run on the FILTERED strategy
    (so its in/out-of-sample, regime, leave-one-out, slippage and Monte-Carlo all
    reflect the filter). The decision should rest on these numbers, not a bucket.
    """
    print("\n" + "=" * 96)
    print(f"  REGIME FILTER — net effect  (weekly ADX in "
          f"[{config.REGIME_ADX_MIN:.0f}, {config.REGIME_ADX_MAX:.0f}), "
          f"slippage {config.SLIPPAGE_PCT * 10000:.0f} bps/side, stop {base_stop}×)")
    print("=" * 96)

    off_m, off_tr = run_universe(bars, base_stop)
    on_m,  on_tr  = run_universe(bars, base_stop, overrides={"USE_REGIME_FILTER": True})

    print("\n  Headline (full sample, live slippage):")
    print(HDR)
    print(fmt_row("filter OFF (baseline)", off_m))
    print(fmt_row("filter ON  (ADX band)", on_m))
    kept = on_m["trades"] / off_m["trades"] * 100 if off_m["trades"] else 0
    print(f"\n  Filter keeps {on_m['trades']}/{off_m['trades']} trades ({kept:.0f}%).")

    print("\n  ── Full robustness+slippage suite on the FILTERED strategy ──")
    saved = config.USE_REGIME_FILTER
    config.USE_REGIME_FILTER = True
    try:
        robustness_study(bars, base_stop)
    finally:
        config.USE_REGIME_FILTER = saved


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_csv(name, rows):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n  wrote → {name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Robustness & sensitivity study")
    ap.add_argument("--years", type=int, default=12)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--stop", type=float, default=config.ACTIVE_STOP_MULT,
                    help="Baseline stop multiplier (default: config.ACTIVE_STOP_MULT)")
    ap.add_argument("--refresh", action="store_true", help="Refetch bars, ignore cache")
    ap.add_argument("--only", choices=["sensitivity", "robustness", "regime", "filter"], default=None)
    args = ap.parse_args()

    # Quiet the engine's per-symbol INFO logging — we run it hundreds of times.
    import logging
    logging.getLogger("backtest").setLevel(logging.WARNING)

    symbols = args.symbols or config.SYMBOLS
    bars = load_all_bars(symbols, args.years, refresh=args.refresh)
    if not bars:
        print("No bars loaded.")
        return

    if args.only in (None, "sensitivity"):
        sensitivity_study(bars, args.stop)
    if args.only in (None, "robustness"):
        robustness_study(bars, args.stop)
    if args.only in (None, "regime"):
        regime_study(bars, args.stop)
    if args.only == "filter":
        filter_comparison(bars, args.stop)


if __name__ == "__main__":
    main()
