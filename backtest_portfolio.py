"""
backtest_portfolio.py — Portfolio-level backtest with shared capital + caps.

Unlike backtest.py (each symbol in its own $100k sandbox), this shares ONE
equity pool across config.SYMBOLS and enforces the live risk caps:
  - config.MAX_POSITIONS    total open positions
  - config.MAX_PER_SECTOR   open positions per sector (sectors.py)
  - config.MAX_POSITION_PCT notional cap per position (no leverage)

Stop modes:
  ATR   : fixed hard stop at entry − mult × ATR(14)       (the live stop)
  Trail : trailing stop at peak_high × (1 − pct), ratchets up only

Entry  = current daily config: RSI(2) crosses back ABOVE RSI_ENTRY_THRESHOLD,
plus the daily SMA50 gate only when config.USE_TREND_FILTER is on. Each bar,
yesterday's signals fill at today's open; most oversold names win scarce slots.

No-lookahead guarantees (audited):
  - Entry signals are read at close[i] and filled at open[i+1].
  - A trailing stop is ratcheted with high[i] only AFTER the bar's exit check,
    so the stop tested on bar i reflects highs through i−1 only.
  - RSI exit uses rsi[i−1]/rsi[i−2] and fills at open[i]; ATR sizing uses the
    signal bar's atr. All indicators are causal (rolling).
  - Stops fill at the stop level, or at the OPEN if the bar gapped through it
    (no optimistic gap fills).

Modes:
  python backtest_portfolio.py             # out-of-sample validation (default)
  python backtest_portfolio.py --compare   # single-period, all stop variants

Output (gitignored): backtest_portfolio_{trades,summary,by_symbol}.csv
"""
import argparse
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import config
import sectors
from indicators import sma, rsi, atr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

START_BAR = config.SMA_DAILY + 5   # warmup for daily SMA50 / ATR / RSI


@dataclass
class _Pos:
    symbol:      str
    sector:      str
    entry_date:  str
    entry_price: float
    stop_price:  float
    entry_bar:   int
    shares:      int
    peak:        float        # highest high since entry (trailing stops)
    trail_pct:   float        # 0.0 for a fixed ATR stop


@dataclass
class Trade:
    symbol:       str
    sector:       str
    stop_variant: str
    entry_date:   str
    entry_price:  float
    exit_date:    str
    exit_price:   float
    shares:       int
    exit_reason:  str
    stop_price:   float
    pnl_dollars:  float = 0.0
    pnl_pct:      float = 0.0
    hold_days:    int   = 0


# ── Prepare data + indicators once ────────────────────────────────────────────

def prepare(symbols: list[str], years: int) -> dict:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365 + 100)
    logger.info(f"Downloading {len(symbols)} symbols | {start.date()} → {end.date()}")
    raw = yf.download(symbols, start=start.date().isoformat(), end=end.date().isoformat(),
                      interval="1d", auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        opens, highs, lows, closes = raw["Open"], raw["High"], raw["Low"], raw["Close"]
    else:
        s = symbols[0]
        rn = lambda col: raw[[col]].rename(columns={col: s})
        opens, highs, lows, closes = rn("Open"), rn("High"), rn("Low"), rn("Close")

    idx = closes.index
    sma50, rsi2, atr14, tradeable = {}, {}, {}, []
    warmup_min = config.SMA_DAILY + 10
    for sym in symbols:
        if sym not in closes.columns:
            logger.warning(f"{sym}: not returned by yfinance — skipping.")
            continue
        c = closes[sym]
        if c.notna().sum() < warmup_min:
            logger.warning(f"{sym}: only {c.notna().sum()} valid bars — skipping.")
            continue
        h, l = highs[sym].reindex(idx), lows[sym].reindex(idx)
        sma50[sym] = sma(c, config.SMA_DAILY).reindex(idx).values
        rsi2[sym]  = rsi(c, config.RSI_PERIOD).reindex(idx).values
        atr14[sym] = atr(h, l, c, config.ATR_PERIOD).reindex(idx).values
        tradeable.append(sym)
    logger.info(f"Data: {closes.shape[0]} days × {closes.shape[1]} symbols | "
                f"{len(tradeable)} tradeable.")
    return {
        "dates": idx, "n": len(idx), "tradeable": tradeable,
        "o": {s: opens[s].reindex(idx).values  for s in tradeable},
        "h": {s: highs[s].reindex(idx).values  for s in tradeable},
        "l": {s: lows[s].reindex(idx).values   for s in tradeable},
        "c": {s: closes[s].reindex(idx).values for s in tradeable},
        "sma50": sma50, "rsi2": rsi2, "atr14": atr14,
        "sec": {s: sectors.sector_of(s) for s in tradeable},
    }


# ── Engine: run over bar range [lo, hi) ───────────────────────────────────────

def run_engine(prep, stop_kind, stop_param, variant, lo, hi,
               max_positions, max_per_sector, initial_equity):
    slip = config.SLIPPAGE_PCT
    o, h, l, c = prep["o"], prep["h"], prep["l"], prep["c"]
    sma50, rsi2, atr14 = prep["sma50"], prep["rsi2"], prep["atr14"]
    sec_of, tradeable, dates = prep["sec"], prep["tradeable"], prep["dates"]

    equity = peak_eq = initial_equity
    max_dd = 0.0
    eq_curve: list[float]      = []
    positions: dict[str, _Pos] = {}
    pending: list[tuple]       = []
    trades: list[Trade]        = []

    def sector_load(sec, extra):
        return sum(1 for p in positions.values() if p.sector == sec) + extra.get(sec, 0)

    for i in range(lo, hi):
        date_str = str(dates[i].date())

        # 1. Fill yesterday's signals at today's open (most oversold first)
        if pending:
            added: dict[str, int] = defaultdict(int)
            for sym, sig_rsi, sig_atr in sorted(pending, key=lambda x: x[1]):
                if len(positions) >= max_positions:
                    break
                if sym in positions:
                    continue
                sec = sec_of[sym]
                if sector_load(sec, added) >= max_per_sector:
                    continue
                open_px = o[sym][i]
                if np.isnan(open_px) or open_px <= 0:
                    continue
                entry_px = open_px * (1 + slip)
                if stop_kind == "atr":
                    if np.isnan(sig_atr):
                        continue
                    stop_dist = stop_param * sig_atr
                else:
                    stop_dist = entry_px * stop_param
                if stop_dist <= 0:
                    continue
                shares = int(equity * config.RISK_PER_TRADE / stop_dist)
                shares = min(shares, int(equity * config.MAX_POSITION_PCT / entry_px))
                if shares <= 0:
                    continue
                positions[sym] = _Pos(sym, sec, date_str, entry_px, entry_px - stop_dist, i,
                                      shares, peak=entry_px,
                                      trail_pct=stop_param if stop_kind == "trail" else 0.0)
                added[sec] += 1
        pending = []

        # 2. Exits (stop tested here reflects prior-bar highs only)
        for sym in list(positions.keys()):
            pos       = positions[sym]
            days_held = i - pos.entry_bar
            low_px, high_px, open_px = l[sym][i], h[sym][i], o[sym][i]
            prev_rsi, prev2_rsi = rsi2[sym][i - 1], rsi2[sym][i - 2]
            if np.isnan(low_px) or np.isnan(open_px):
                continue

            exit_px = exit_reason = None
            if low_px <= pos.stop_price:                                  # hard / trailing stop
                # Gap-through: if the bar opened below the stop, fill at the open.
                exit_px = min(pos.stop_price, open_px)
                exit_reason = "stop_loss"
            elif days_held >= config.MAX_HOLD_DAYS:                       # time stop
                exit_px, exit_reason = open_px, "time_stop"
            elif (not np.isnan(prev_rsi) and not np.isnan(prev2_rsi)      # RSI crossback below 70
                  and prev2_rsi >= config.RSI_EXIT_THRESHOLD
                  and prev_rsi  <  config.RSI_EXIT_THRESHOLD):
                exit_px, exit_reason = open_px, "rsi_exit"

            if exit_px is not None:
                exit_px *= (1 - slip)
                pnl_d = (exit_px - pos.entry_price) * pos.shares
                equity += pnl_d
                trades.append(Trade(
                    symbol=sym, sector=pos.sector, stop_variant=variant,
                    entry_date=pos.entry_date, entry_price=round(pos.entry_price, 4),
                    exit_date=date_str, exit_price=round(exit_px, 4), shares=pos.shares,
                    exit_reason=exit_reason, stop_price=round(pos.stop_price, 4),
                    pnl_dollars=round(pnl_d, 2),
                    pnl_pct=round((exit_px - pos.entry_price) / pos.entry_price * 100, 2),
                    hold_days=days_held,
                ))
                del positions[sym]
            elif pos.trail_pct and not np.isnan(high_px):
                # Survived the bar — ratchet the trailing stop up for the NEXT bar.
                pos.peak = max(pos.peak, high_px)
                pos.stop_price = max(pos.stop_price, pos.peak * (1 - pos.trail_pct))

        # 3. Scan for new signals at this close → fill next open
        if len(positions) < max_positions:
            sig = []
            for sym in tradeable:
                if sym in positions:
                    continue
                cur_rsi, prev_rsi = rsi2[sym][i], rsi2[sym][i - 1]
                cc, ss, aa = c[sym][i], sma50[sym][i], atr14[sym][i]
                if np.isnan(cur_rsi) or np.isnan(prev_rsi) or np.isnan(aa) or np.isnan(cc):
                    continue
                crossback = prev_rsi <= config.RSI_ENTRY_THRESHOLD and cur_rsi > config.RSI_ENTRY_THRESHOLD
                trend_ok  = (not np.isnan(ss) and cc > ss) if config.USE_TREND_FILTER else True
                if crossback and trend_ok:
                    sig.append((sym, cur_rsi, aa))
            pending = sig

        # 4. Mark-to-market + drawdown
        mtm = equity + sum((c[s][i] - p.entry_price) * p.shares
                           for s, p in positions.items() if not np.isnan(c[s][i]))
        eq_curve.append(mtm)
        peak_eq = max(peak_eq, mtm)
        max_dd  = max(max_dd, (peak_eq - mtm) / peak_eq * 100)

    return trades, _summarize(trades, equity, initial_equity, dates, lo, hi, variant, max_dd)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _summarize(trades, equity, init_eq, dates, lo, hi, variant, max_dd):
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers  = [t for t in trades if t.pnl_dollars <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    gw = sum(t.pnl_dollars for t in winners); gl = abs(sum(t.pnl_dollars for t in losers))
    pf = gw / gl if gl > 0 else float("inf")
    tot_ret = (equity - init_eq) / init_eq * 100
    n_years = (dates[hi - 1] - dates[lo]).days / 365.25
    cagr = ((equity / init_eq) ** (1 / n_years) - 1) * 100 if n_years > 0 and equity > 0 else 0
    avg_hold = sum(t.hold_days for t in trades) / len(trades) if trades else 0
    return {
        "variant": variant, "start": str(dates[lo].date()), "end": str(dates[hi - 1].date()),
        "total_trades": len(trades), "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2), "cagr_pct": round(cagr, 2),
        "total_return_pct": round(tot_ret, 2), "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(avg_hold, 1), "final_equity": round(equity, 2),
    }


def per_symbol_stats(trades):
    groups = defaultdict(list)
    for t in trades:
        groups[(t.symbol, t.sector, t.stop_variant)].append(t)
    rows = []
    for (sym, sec, variant), ts in groups.items():
        winners = [t for t in ts if t.pnl_dollars > 0]
        rows.append({
            "symbol": sym, "sector": sec, "stop_variant": variant, "total_trades": len(ts),
            "win_rate_pct": round(len(winners) / len(ts) * 100, 1) if ts else 0,
            "total_pnl": round(sum(t.pnl_dollars for t in ts), 2),
        })
    return sorted(rows, key=lambda r: r["total_pnl"], reverse=True)


def _save(rows, filename):
    if not rows:
        return
    dicts = [r if isinstance(r, dict) else asdict(r) for r in rows]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dicts[0].keys())
        w.writeheader(); w.writerows(dicts)
    logger.info(f"Saved → {filename} ({len(rows):,} rows)")


def _row(label, s):
    return (f"  {label:<10}{s['variant']:<10}{s['start']}→{s['end']}"
            f"{s['total_return_pct']:>+8.1f}%{s['cagr_pct']:>+7.2f}%{s['max_drawdown_pct']:>7.1f}%"
            f"{s['win_rate_pct']:>7.1f}{s['profit_factor']:>6.2f}{s['total_trades']:>7,}")


def _header():
    print(f"  {'Period':<10}{'Variant':<10}{'Window':<23}{'Return':>9}{'CAGR':>7}"
          f"{'MaxDD':>8}{'Win%':>7}{'PF':>6}{'Trades':>7}")
    print("  " + "-" * 92)


# ── Drivers ───────────────────────────────────────────────────────────────────

VARIANTS = [
    ("atr",   config.STOP_MULT_B, f"ATR-{config.STOP_MULT_B}x"),
    ("trail", 0.01,               "Trail-1%"),
    ("trail", 0.02,               "Trail-2%"),
]


def validate_oos(prep, args):
    """Split the window in half: train on the first half, test on the second."""
    n   = prep["n"]
    mid = (START_BAR + n) // 2
    print(f"\n{'='*96}\n  OUT-OF-SAMPLE VALIDATION  ({config.MAX_POSITIONS} pos / "
          f"{config.MAX_PER_SECTOR} per sector / {config.MAX_POSITION_PCT:.0%} cap)\n{'='*96}")
    _header()
    all_trades, all_sum = [], []
    for kind, param, label in VARIANTS:
        kw = dict(max_positions=args.max_positions, max_per_sector=args.max_per_sector,
                  initial_equity=args.equity)
        t_is, s_is = run_engine(prep, kind, param, label, START_BAR, mid, **kw)
        t_oos, s_oos = run_engine(prep, kind, param, label, mid, n, **kw)
        print(_row("In-samp", s_is))
        print(_row("Out-samp", s_oos))
        print("  " + "·" * 92)
        all_trades += t_is + t_oos
        all_sum += [{"period": "in_sample", **s_is}, {"period": "out_sample", **s_oos}]
    print(f"{'='*96}\n")
    return all_trades, all_sum


def compare_full(prep, args):
    n = prep["n"]
    full = [("atr", config.STOP_MULT_A, f"ATR-{config.STOP_MULT_A}x")] + VARIANTS + \
           [("trail", 0.05, "Trail-5%")]
    print(f"\n{'='*96}\n  FULL-PERIOD COMPARISON\n{'='*96}")
    _header()
    all_trades, all_sum = [], []
    for kind, param, label in full:
        t, s = run_engine(prep, kind, param, label, START_BAR, n,
                          max_positions=args.max_positions, max_per_sector=args.max_per_sector,
                          initial_equity=args.equity)
        print(_row("Full", s))
        all_trades += t; all_sum.append(s)
    print(f"{'='*96}\n")
    return all_trades, all_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio-level MeansRev backtest")
    parser.add_argument("--years",          type=int,   default=12)
    parser.add_argument("--max-positions",  type=int,   default=config.MAX_POSITIONS)
    parser.add_argument("--max-per-sector", type=int,   default=config.MAX_PER_SECTOR)
    parser.add_argument("--equity",         type=float, default=100_000.0)
    parser.add_argument("--compare", action="store_true",
                        help="Full-period comparison of all stop variants (default: OOS validation).")
    args = parser.parse_args()

    if config.USE_TREND_FILTER or config.USE_REGIME_FILTER:
        logger.warning("Weekly gate is ON — daily-only engine; results approximate. Use backtest.py.")

    prep = prepare(config.SYMBOLS, args.years)
    all_trades, all_sum = compare_full(prep, args) if args.compare else validate_oos(prep, args)

    _save(all_trades,                   "backtest_portfolio_trades.csv")
    _save(all_sum,                      "backtest_portfolio_summary.csv")
    _save(per_symbol_stats(all_trades), "backtest_portfolio_by_symbol.csv")
    print("✅ Done. → backtest_portfolio_{trades,summary,by_symbol}.csv")
