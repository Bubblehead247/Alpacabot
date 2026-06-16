"""
backtest_portfolio.py — Portfolio-level backtest with shared capital + caps.

Unlike backtest.py (which runs each symbol in its own $100k sandbox), this
shares ONE equity pool across config.SYMBOLS and enforces the live risk caps:
  - config.MAX_POSITIONS    total open positions
  - config.MAX_PER_SECTOR   open positions per sector (see sectors.py)
  - config.MAX_POSITION_PCT notional cap per position (no leverage)

So it answers the real question: "what would the *account* have done?"

Stop modes (compared side by side by default):
  ATR   : fixed hard stop at entry − mult × ATR(14)         (the live stop)
  Trail : trailing stop at peak_high × (1 − pct), ratcheting up only
Both modes keep the other exits unchanged (time stop, RSI(2) crossback below 70).

Entry = current daily config: RSI(2) crosses back ABOVE RSI_ENTRY_THRESHOLD,
plus the daily SMA50 gate only when config.USE_TREND_FILTER is on. Each bar,
yesterday's signals fill at today's open; most oversold names win scarce slots.
Per-side slippage (config.SLIPPAGE_PCT) applied. CAGR is calendar-basis.

NOTE: daily-only engine — the weekly trend/regime gates are not modelled.

Usage:
    python backtest_portfolio.py                 # ATR 1.5/2.5 + trail 1/2/5%
    python backtest_portfolio.py --years 8

Output (gitignored):
    backtest_portfolio_trades.csv
    backtest_portfolio_summary.csv
    backtest_portfolio_by_symbol.csv
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


@dataclass
class _Pos:
    symbol:      str
    sector:      str
    entry_date:  str
    entry_price: float
    stop_price:  float
    entry_bar:   int
    shares:      int
    peak:        float        # highest high since entry (for trailing stops)
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


# ── Data + indicators ─────────────────────────────────────────────────────────

def _load_data(symbols: list[str], years: int):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365 + 100)
    logger.info(f"Downloading {len(symbols)} symbols | {start.date()} → {end.date()}")
    raw = yf.download(symbols, start=start.date().isoformat(), end=end.date().isoformat(),
                      interval="1d", auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        opens, highs, lows, closes = raw["Open"], raw["High"], raw["Low"], raw["Close"]
    else:  # single-symbol fallback
        s = symbols[0]
        rename = lambda col: raw[[col]].rename(columns={col: s})
        opens, highs, lows, closes = rename("Open"), rename("High"), rename("Low"), rename("Close")
    logger.info(f"Data: {closes.shape[0]} trading days × {closes.shape[1]} symbols")
    return opens, highs, lows, closes


def _build_indicators(opens, highs, lows, closes, symbols):
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
    logger.info(f"Indicators ready: {len(tradeable)} / {len(symbols)} symbols tradeable.")
    return sma50, rsi2, atr14, tradeable


# ── Portfolio engine ──────────────────────────────────────────────────────────

def run(symbols, stop_kind, stop_param, variant, max_positions, max_per_sector,
        years, initial_equity):
    """stop_kind: 'atr' (stop_param = ATR multiple) or 'trail' (stop_param = pct)."""
    slip = config.SLIPPAGE_PCT
    logger.info(f"\nPortfolio backtest | {variant} | MaxPos {max_positions} | "
                f"MaxPerSector {max_per_sector} | NotionalCap {config.MAX_POSITION_PCT:.0%}")

    opens, highs, lows, closes = _load_data(symbols, years)
    sma50, rsi2, atr14, tradeable = _build_indicators(opens, highs, lows, closes, symbols)

    o_arr = {s: opens[s].reindex(closes.index).values  for s in tradeable}
    h_arr = {s: highs[s].reindex(closes.index).values  for s in tradeable}
    l_arr = {s: lows[s].reindex(closes.index).values   for s in tradeable}
    c_arr = {s: closes[s].reindex(closes.index).values for s in tradeable}
    sec_of = {s: sectors.sector_of(s) for s in tradeable}

    dates      = closes.index
    n          = len(dates)
    equity     = initial_equity
    peak_eq    = initial_equity
    max_dd     = 0.0
    eq_curve: list[float]      = []
    positions: dict[str, _Pos] = {}
    pending: list[tuple]       = []   # (symbol, signal_rsi, signal_atr)
    all_trades: list[Trade]    = []
    start_bar  = config.SMA_DAILY + 5

    def sector_load(sec, extra):
        return sum(1 for p in positions.values() if p.sector == sec) + extra.get(sec, 0)

    for i in range(start_bar, n):
        date_str = str(dates[i].date())

        # 1. Execute yesterday's pending entries at today's open (most oversold first)
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
                open_px = o_arr[sym][i]
                if np.isnan(open_px) or open_px <= 0:
                    continue
                entry_px = open_px * (1 + slip)
                if stop_kind == "atr":
                    if np.isnan(sig_atr):
                        continue
                    stop_dist = stop_param * sig_atr
                else:  # trail
                    stop_dist = entry_px * stop_param
                stop_px = entry_px - stop_dist
                if stop_dist <= 0:
                    continue
                shares = int(equity * config.RISK_PER_TRADE / stop_dist)
                shares = min(shares, int(equity * config.MAX_POSITION_PCT / entry_px))  # notional cap
                if shares <= 0:
                    continue
                positions[sym] = _Pos(sym, sec, date_str, entry_px, stop_px, i, shares,
                                      peak=entry_px,
                                      trail_pct=stop_param if stop_kind == "trail" else 0.0)
                added[sec] += 1
        pending = []

        # 2. Check exits for open positions
        for sym in list(positions.keys()):
            pos       = positions[sym]
            days_held = i - pos.entry_bar
            low_px    = l_arr[sym][i]
            high_px   = h_arr[sym][i]
            open_px   = o_arr[sym][i]
            prev_rsi  = rsi2[sym][i - 1]
            prev2_rsi = rsi2[sym][i - 2]
            if np.isnan(low_px) or np.isnan(open_px):
                continue

            # The stop level for this bar reflects highs through the PRIOR bar
            # (set at the end of the last iteration) — no same-bar lookahead.
            exit_px = exit_reason = None
            if low_px <= pos.stop_price:                                  # hard / trailing stop
                exit_px, exit_reason = pos.stop_price, "stop_loss"
            elif days_held >= config.MAX_HOLD_DAYS:                       # time stop
                exit_px, exit_reason = open_px, "time_stop"
            elif (not np.isnan(prev_rsi) and not np.isnan(prev2_rsi)      # RSI crossback below 70
                  and prev2_rsi >= config.RSI_EXIT_THRESHOLD
                  and prev_rsi  <  config.RSI_EXIT_THRESHOLD):
                exit_px, exit_reason = open_px, "rsi_exit"

            if exit_px is not None:
                exit_px *= (1 - slip)
                pnl_d = (exit_px - pos.entry_price) * pos.shares
                pnl_p = (exit_px - pos.entry_price) / pos.entry_price * 100
                equity += pnl_d
                all_trades.append(Trade(
                    symbol=sym, sector=pos.sector, stop_variant=variant,
                    entry_date=pos.entry_date, entry_price=round(pos.entry_price, 4),
                    exit_date=date_str, exit_price=round(exit_px, 4), shares=pos.shares,
                    exit_reason=exit_reason, stop_price=round(pos.stop_price, 4),
                    pnl_dollars=round(pnl_d, 2), pnl_pct=round(pnl_p, 2), hold_days=days_held,
                ))
                del positions[sym]
            elif pos.trail_pct and not np.isnan(high_px):
                # Survived the bar — ratchet the trailing stop up for the NEXT bar.
                pos.peak = max(pos.peak, high_px)
                pos.stop_price = max(pos.stop_price, pos.peak * (1 - pos.trail_pct))

        # 3. Scan for new entry signals (RSI crossback above 10), fill at next open
        if len(positions) < max_positions:
            new_signals = []
            for sym in tradeable:
                if sym in positions:
                    continue
                cur_rsi, prev_rsi = rsi2[sym][i], rsi2[sym][i - 1]
                c, s, a = c_arr[sym][i], sma50[sym][i], atr14[sym][i]
                if np.isnan(cur_rsi) or np.isnan(prev_rsi) or np.isnan(a) or np.isnan(c):
                    continue
                crossback = prev_rsi <= config.RSI_ENTRY_THRESHOLD and cur_rsi > config.RSI_ENTRY_THRESHOLD
                trend_ok  = (not np.isnan(s) and c > s) if config.USE_TREND_FILTER else True
                if crossback and trend_ok:
                    new_signals.append((sym, cur_rsi, a))
            pending = new_signals

        # 4. Mark-to-market + drawdown
        mtm = equity + sum((c_arr[s][i] - p.entry_price) * p.shares
                           for s, p in positions.items() if not np.isnan(c_arr[s][i]))
        eq_curve.append(mtm)
        peak_eq = max(peak_eq, mtm)
        max_dd  = max(max_dd, (peak_eq - mtm) / peak_eq * 100)

    return all_trades, _summarize(all_trades, eq_curve, equity, initial_equity,
                                  dates, start_bar, variant, max_dd)


# ── Stats + reporting ─────────────────────────────────────────────────────────

def _summarize(trades, eq_curve, equity, init_eq, dates, start_bar, variant, max_dd):
    winners = [t for t in trades if t.pnl_dollars > 0]
    losers  = [t for t in trades if t.pnl_dollars <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_win  = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl_pct for t in losers)  / len(losers)  if losers  else 0
    gw = sum(t.pnl_dollars for t in winners); gl = abs(sum(t.pnl_dollars for t in losers))
    pf = gw / gl if gl > 0 else float("inf")
    tot_ret = (equity - init_eq) / init_eq * 100
    n_years = (dates[-1] - dates[start_bar]).days / 365.25
    cagr = ((equity / init_eq) ** (1 / n_years) - 1) * 100 if n_years > 0 and equity > 0 else 0
    avg_hold = sum(t.hold_days for t in trades) / len(trades) if trades else 0
    ex = {r: sum(1 for t in trades if t.exit_reason == r) for r in ("rsi_exit", "stop_loss", "time_stop")}
    return {
        "variant": variant, "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1), "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2), "profit_factor": round(pf, 2),
        "cagr_pct": round(cagr, 2), "total_return_pct": round(tot_ret, 2),
        "max_drawdown_pct": round(max_dd, 2), "avg_hold_days": round(avg_hold, 1),
        "rsi_exits": ex["rsi_exit"], "stop_exits": ex["stop_loss"], "time_exits": ex["time_stop"],
        "final_equity": round(equity, 2),
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
            "avg_hold_days": round(sum(t.hold_days for t in ts) / len(ts), 1) if ts else 0,
        })
    return sorted(rows, key=lambda r: r["total_pnl"], reverse=True)


def _save(rows, filename):
    if not rows:
        return
    dicts = [r if isinstance(r, dict) else asdict(r) for r in rows]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dicts[0].keys())
        w.writeheader()
        w.writerows(dicts)
    logger.info(f"Saved → {filename} ({len(rows):,} rows)")


def _print_comparison(summaries):
    print(f"\n{'='*86}\n  STOP-METHOD COMPARISON  ({config.MAX_POSITIONS} pos / {config.MAX_PER_SECTOR} "
          f"per sector / {config.MAX_POSITION_PCT:.0%} notional cap)\n{'='*86}")
    print(f"  {'Variant':<12}{'Return':>9}{'CAGR':>8}{'MaxDD':>8}{'Win%':>7}{'PF':>6}{'Trades':>8}{'Hold':>6}")
    print("  " + "-" * 82)
    for s in summaries:
        print(f"  {s['variant']:<12}{s['total_return_pct']:>+8.1f}%{s['cagr_pct']:>+7.2f}%"
              f"{s['max_drawdown_pct']:>7.1f}%{s['win_rate_pct']:>7.1f}{s['profit_factor']:>6.2f}"
              f"{s['total_trades']:>8,}{s['avg_hold_days']:>6.1f}")
    print(f"{'='*86}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio-level MeansRev backtest")
    parser.add_argument("--years",          type=int,   default=12)
    parser.add_argument("--max-positions",  type=int,   default=config.MAX_POSITIONS)
    parser.add_argument("--max-per-sector", type=int,   default=config.MAX_PER_SECTOR)
    parser.add_argument("--equity",         type=float, default=100_000.0)
    args = parser.parse_args()

    if config.USE_TREND_FILTER or config.USE_REGIME_FILTER:
        logger.warning("Weekly gate is ON — daily-only engine; results approximate. Use backtest.py.")

    # (stop_kind, stop_param, label)
    VARIANTS = [
        ("atr",   config.STOP_MULT_A, f"ATR-{config.STOP_MULT_A}x"),
        ("atr",   config.STOP_MULT_B, f"ATR-{config.STOP_MULT_B}x"),
        ("trail", 0.01,               "Trail-1%"),
        ("trail", 0.02,               "Trail-2%"),
        ("trail", 0.05,               "Trail-5%"),
    ]

    all_trades, all_summaries = [], []
    for kind, param, label in VARIANTS:
        trades, summary = run(
            symbols=config.SYMBOLS, stop_kind=kind, stop_param=param, variant=label,
            max_positions=args.max_positions, max_per_sector=args.max_per_sector,
            years=args.years, initial_equity=args.equity,
        )
        all_trades.extend(trades)
        all_summaries.append(summary)

    _print_comparison(all_summaries)
    _save(all_trades,                   "backtest_portfolio_trades.csv")
    _save(all_summaries,                "backtest_portfolio_summary.csv")
    _save(per_symbol_stats(all_trades), "backtest_portfolio_by_symbol.csv")
    print("✅ Done. → backtest_portfolio_{trades,summary,by_symbol}.csv")
