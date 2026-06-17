"""
backtest_walkforward.py — Point-in-time (walk-forward) screening backtest.

Fixes the selection bias in backtest_portfolio.py. There, the 30-name universe
was chosen by screening for "sideways" in mid-2026 and then backtested back to
2014 — look-ahead: most of those names were strong uptrends over the test
window, not range-bound at the time of each trade.

Here the universe is rebuilt every month from a broad candidate pool using ONLY
trailing data as of that date: rank by a sideways score (low trailing ADX + price
hugging its SMA), keep the top K, and allow entries only in that active set.
Existing positions are held to their normal exit even if they drop out.

Everything else matches the live daily strategy: RSI(2) crossback entry, ATR(2.5×)
stop with gap-through fills, time stop, RSI exit, the MAX_POSITIONS /
MAX_PER_SECTOR / MAX_POSITION_PCT caps, and per-side slippage.

Selection-bias recheck: every entry records the name's TRAILING ADX and range
score at the time it was traded, so we can confirm the traded names were genuinely
sideways then (not chosen with future data).

Residual caveat: the candidate POOL below is long-lived liquid names that exist
today, so mild survivorship remains — but the dominant bias (sideways selected
with end-of-period data) is removed.

Usage:
    python backtest_walkforward.py            # full period + in/out-of-sample split
Output (gitignored): backtest_wf_{trades,summary}.csv
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
from indicators import sma, rsi, atr, adx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

REBAL_DAYS    = 21                       # monthly rebalance
ACTIVE_TOP_K  = 30                       # names kept in the active universe
RANGE_WINDOW  = 60                       # trailing bars for the range metric
RANGE_BAND    = 0.05                     # ±5% of SMA50 counts as "near the mean"
WF_START      = config.SMA_DAILY + RANGE_WINDOW + 5
STOP_MULT     = config.STOP_MULT_B       # ATR 2.5× (the validated stop)

# Candidate pool: long-lived liquid US stocks + ETFs, with GICS sector for the cap.
# (Sector ETFs → their sector; broad/bond/intl ETFs → "Broad/Index".)
POOL: dict[str, str] = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","ORCL":"Technology",
    "CRM":"Technology","ADBE":"Technology","CSCO":"Technology","INTC":"Technology",
    "AMD":"Technology","QCOM":"Technology","TXN":"Technology","IBM":"Technology",
    "AVGO":"Technology","UBER":"Technology",
    # Financials
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","C":"Financials","AXP":"Financials","SPGI":"Financials",
    "V":"Financials","MA":"Financials","SCHW":"Financials","BLK":"Financials","USB":"Financials",
    # Health Care
    "JNJ":"Health Care","UNH":"Health Care","PFE":"Health Care","MRK":"Health Care",
    "ABBV":"Health Care","AMGN":"Health Care","LLY":"Health Care","TMO":"Health Care",
    "ABT":"Health Care","MDT":"Health Care","BMY":"Health Care","GILD":"Health Care","IDXX":"Health Care",
    # Consumer Discretionary
    "AMZN":"Consumer Discretionary","HD":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "NKE":"Consumer Discretionary","SBUX":"Consumer Discretionary","TGT":"Consumer Discretionary",
    "LOW":"Consumer Discretionary","DIS":"Consumer Discretionary","ABNB":"Consumer Discretionary",
    # Consumer Staples
    "COST":"Consumer Staples","WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples",
    "PEP":"Consumer Staples","MDLZ":"Consumer Staples","CL":"Consumer Staples",
    # Industrials
    "BA":"Industrials","CAT":"Industrials","GE":"Industrials","HON":"Industrials",
    "UPS":"Industrials","RTX":"Industrials","LMT":"Industrials","DE":"Industrials",
    "MMM":"Industrials","TDG":"Industrials","EMR":"Industrials","ETN":"Industrials",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "OXY":"Energy","FANG":"Energy","PSX":"Energy",
    # Materials
    "LIN":"Materials","SHW":"Materials","APD":"Materials","ECL":"Materials","FCX":"Materials","NEM":"Materials",
    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    # Communication Services
    "T":"Communication Services","VZ":"Communication Services","CMCSA":"Communication Services",
    "NFLX":"Communication Services","GOOGL":"Communication Services","META":"Communication Services",
    # Sector ETFs
    "XLE":"Energy","XLF":"Financials","XLK":"Technology","XLV":"Health Care","XLI":"Industrials",
    "XLY":"Consumer Discretionary","XLP":"Consumer Staples","XLB":"Materials","XLU":"Utilities",
    "XLRE":"Real Estate","XLC":"Communication Services","XBI":"Health Care","XOP":"Energy","XRT":"Consumer Discretionary",
    # Broad / bond / international ETFs
    "SPY":"Broad/Index","QQQ":"Broad/Index","IWM":"Broad/Index","DIA":"Broad/Index","MDY":"Broad/Index",
    "RSP":"Broad/Index","IJH":"Broad/Index","EFA":"Broad/Index","IEFA":"Broad/Index","VGK":"Broad/Index",
    "EZU":"Broad/Index","EEM":"Broad/Index","TLT":"Broad/Index","IEF":"Broad/Index","LQD":"Broad/Index",
    "HYG":"Broad/Index","EMB":"Broad/Index","VCLT":"Broad/Index","AGG":"Broad/Index","BND":"Broad/Index","GLD":"Broad/Index",
}


@dataclass
class _Pos:
    symbol: str; sector: str; entry_date: str; entry_price: float
    stop_price: float; entry_bar: int; shares: int; entry_adx: float; entry_range: float


@dataclass
class Trade:
    symbol: str; sector: str; entry_date: str; entry_price: float
    exit_date: str; exit_price: float; shares: int; exit_reason: str
    pnl_dollars: float; pnl_pct: float; hold_days: int
    entry_adx: float; entry_range: float


# ── Data + indicators ─────────────────────────────────────────────────────────

def prepare(years: int) -> dict:
    syms = list(POOL)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365 + 150)
    logger.info(f"Downloading {len(syms)} candidate symbols | {start.date()} → {end.date()}")
    raw = yf.download(syms, start=start.date().isoformat(), end=end.date().isoformat(),
                      interval="1d", auto_adjust=True, progress=False, threads=True)
    O, H, L, C, Vv = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]
    idx = C.index
    o=h=l=c=v=None
    o,h,l,c,v,rsi2,sma50,atr14,adx14,tradeable = {},{},{},{},{},{},{},{},{},[]
    for s in syms:
        if s not in C.columns or C[s].notna().sum() < config.SMA_DAILY + 30:
            continue
        cc=C[s]; hh=H[s].reindex(idx); ll=L[s].reindex(idx); vv=Vv[s].reindex(idx)
        o[s]=O[s].reindex(idx).values; h[s]=hh.values; l[s]=ll.values
        c[s]=cc.reindex(idx).values;   v[s]=vv.values
        rsi2[s]=rsi(cc,config.RSI_PERIOD).reindex(idx).values
        sma50[s]=sma(cc,config.SMA_DAILY).reindex(idx).values
        atr14[s]=atr(hh,ll,cc,config.ATR_PERIOD).reindex(idx).values
        adx14[s]=adx(hh,ll,cc,config.REGIME_ADX_PERIOD).reindex(idx).values
        tradeable.append(s)
    logger.info(f"Pool: {len(tradeable)}/{len(syms)} usable over {len(idx)} days.")
    return dict(dates=idx,n=len(idx),tradeable=tradeable,o=o,h=h,l=l,c=c,v=v,
                rsi2=rsi2,sma50=sma50,atr14=atr14,adx14=adx14)


# ── Point-in-time sideways screen ─────────────────────────────────────────────

def _sideways_score(prep, s, r):
    """Score at bar r using ONLY data through r. Returns (score, adx) or None."""
    adxv = prep["adx14"][s][r]; smav = prep["sma50"][s][r]
    if np.isnan(adxv) or np.isnan(smav):
        return None
    lo = max(0, r - RANGE_WINDOW + 1)
    seg_c = prep["c"][s][lo:r + 1]; seg_s = prep["sma50"][s][lo:r + 1]
    seg_v = prep["v"][s][lo:r + 1]
    valid = ~np.isnan(seg_s) & ~np.isnan(seg_c)
    if valid.sum() < RANGE_WINDOW // 2:
        return None
    # Liquidity floor: trailing avg dollar volume.
    dollar_vol = np.nanmean(seg_c[valid] * seg_v[valid])
    if dollar_vol < config.SCREEN_MIN_DOLLAR_VOLUME:
        return None
    within = np.abs(seg_c[valid] - seg_s[valid]) / seg_s[valid] <= RANGE_BAND
    range_frac = within.mean()
    trend = max(0.0, 1.0 - adxv / 40.0)
    score = 0.6 * trend + 0.4 * range_frac
    return score, adxv


def _active_sets(prep, lo, hi):
    """Build {rebalance_bar: set(active symbols)} over [lo, hi)."""
    out = {}
    for r in range(lo, hi, REBAL_DAYS):
        scored = []
        for s in prep["tradeable"]:
            res = _sideways_score(prep, s, r)
            if res is not None:
                scored.append((s, res[0]))
        scored.sort(key=lambda x: -x[1])
        out[r] = {s for s, _ in scored[:ACTIVE_TOP_K]}
    return out


# ── Engine ────────────────────────────────────────────────────────────────────

def run_engine(prep, lo, hi, variant, initial_equity):
    slip = config.SLIPPAGE_PCT
    o,h,l,c = prep["o"],prep["h"],prep["l"],prep["c"]
    rsi2,sma50,atr14,adx14 = prep["rsi2"],prep["sma50"],prep["atr14"],prep["adx14"]
    dates,tradeable = prep["dates"],prep["tradeable"]

    active_sets = _active_sets(prep, lo, hi)
    rebal_bars  = sorted(active_sets)

    equity = peak_eq = initial_equity
    max_dd = 0.0
    eq_curve=[]; positions={}; pending=[]; trades=[]

    def sector_load(sec, extra):
        return sum(1 for p in positions.values() if p.sector == sec) + extra.get(sec, 0)

    cur_active = set()
    ri = -1
    for i in range(lo, hi):
        while ri + 1 < len(rebal_bars) and rebal_bars[ri + 1] <= i:
            ri += 1
            cur_active = active_sets[rebal_bars[ri]]
        date_str = str(dates[i].date())

        # 1. Fill pending entries at today's open
        if pending:
            added=defaultdict(int)
            for sym, sig_rsi, sig_atr, sig_adx, sig_rng in sorted(pending, key=lambda x: x[1]):
                if len(positions) >= config.MAX_POSITIONS: break
                if sym in positions: continue
                sec = POOL[sym]
                if sector_load(sec, added) >= config.MAX_PER_SECTOR: continue
                open_px = o[sym][i]
                if np.isnan(open_px) or open_px <= 0 or np.isnan(sig_atr): continue
                entry_px = open_px * (1 + slip)
                stop_dist = STOP_MULT * sig_atr
                if stop_dist <= 0: continue
                shares = int(equity * config.RISK_PER_TRADE / stop_dist)
                shares = min(shares, int(equity * config.MAX_POSITION_PCT / entry_px))
                if shares <= 0: continue
                positions[sym] = _Pos(sym, sec, date_str, entry_px, entry_px-stop_dist, i,
                                      shares, sig_adx, sig_rng)
                added[sec]+=1
        pending=[]

        # 2. Exits
        for sym in list(positions.keys()):
            pos=positions[sym]; days_held=i-pos.entry_bar
            low_px,open_px=l[sym][i],o[sym][i]
            prev_rsi,prev2_rsi=rsi2[sym][i-1],rsi2[sym][i-2]
            if np.isnan(low_px) or np.isnan(open_px): continue
            exit_px=exit_reason=None
            if low_px <= pos.stop_price:
                exit_px=min(pos.stop_price, open_px); exit_reason="stop_loss"   # gap-through
            elif days_held >= config.MAX_HOLD_DAYS:
                exit_px,exit_reason=open_px,"time_stop"
            elif (not np.isnan(prev_rsi) and not np.isnan(prev2_rsi)
                  and prev2_rsi>=config.RSI_EXIT_THRESHOLD and prev_rsi<config.RSI_EXIT_THRESHOLD):
                exit_px,exit_reason=open_px,"rsi_exit"
            if exit_px is not None:
                exit_px*=(1-slip)
                pnl_d=(exit_px-pos.entry_price)*pos.shares; equity+=pnl_d
                trades.append(Trade(sym,pos.sector,pos.entry_date,round(pos.entry_price,4),
                    date_str,round(exit_px,4),pos.shares,exit_reason,round(pnl_d,2),
                    round((exit_px-pos.entry_price)/pos.entry_price*100,2),days_held,
                    round(pos.entry_adx,1),round(pos.entry_range,2)))
                del positions[sym]

        # 3. Scan signals among the CURRENT active universe only
        if len(positions) < config.MAX_POSITIONS:
            sig=[]
            for sym in cur_active:
                if sym in positions or sym not in tradeable: continue
                cur_rsi,prev_rsi=rsi2[sym][i],rsi2[sym][i-1]
                cc,aa,ax=c[sym][i],atr14[sym][i],adx14[sym][i]
                if np.isnan(cur_rsi) or np.isnan(prev_rsi) or np.isnan(aa) or np.isnan(cc): continue
                if prev_rsi<=config.RSI_ENTRY_THRESHOLD and cur_rsi>config.RSI_ENTRY_THRESHOLD:
                    res=_sideways_score(prep,sym,i)
                    rng = res[0] if res else 0.0
                    sig.append((sym,cur_rsi,aa, ax if not np.isnan(ax) else 0.0, rng))
            pending=sig

        # 4. MTM + drawdown
        mtm=equity+sum((c[s][i]-p.entry_price)*p.shares for s,p in positions.items()
                       if not np.isnan(c[s][i]))
        eq_curve.append(mtm); peak_eq=max(peak_eq,mtm)
        max_dd=max(max_dd,(peak_eq-mtm)/peak_eq*100)

    return trades, _summarize(trades, equity, initial_equity, dates, lo, hi, variant, max_dd)


def _summarize(trades, equity, init_eq, dates, lo, hi, variant, max_dd):
    winners=[t for t in trades if t.pnl_dollars>0]; losers=[t for t in trades if t.pnl_dollars<=0]
    wr=len(winners)/len(trades)*100 if trades else 0
    gw=sum(t.pnl_dollars for t in winners); gl=abs(sum(t.pnl_dollars for t in losers))
    pf=gw/gl if gl>0 else float("inf")
    tot=(equity-init_eq)/init_eq*100
    yrs=(dates[hi-1]-dates[lo]).days/365.25
    cagr=((equity/init_eq)**(1/yrs)-1)*100 if yrs>0 and equity>0 else 0
    adx_at_entry=np.mean([t.entry_adx for t in trades]) if trades else 0
    pct_low_adx=100*np.mean([t.entry_adx<25 for t in trades]) if trades else 0
    return dict(variant=variant,start=str(dates[lo].date()),end=str(dates[hi-1].date()),
        total_trades=len(trades),win_rate_pct=round(wr,1),profit_factor=round(pf,2),
        cagr_pct=round(cagr,2),total_return_pct=round(tot,2),max_drawdown_pct=round(max_dd,2),
        avg_entry_adx=round(adx_at_entry,1),pct_entries_adx_below_25=round(pct_low_adx,1),
        final_equity=round(equity,2))


def _save(rows, fn):
    if not rows: return
    dd=[r if isinstance(r,dict) else asdict(r) for r in rows]
    with open(fn,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=dd[0].keys()); w.writeheader(); w.writerows(dd)
    logger.info(f"Saved → {fn} ({len(rows):,} rows)")


def _print(label, s):
    print(f"  {label:<10}{s['start']}→{s['end']}{s['total_return_pct']:>+8.1f}%"
          f"{s['cagr_pct']:>+7.2f}%{s['max_drawdown_pct']:>7.1f}%{s['win_rate_pct']:>7.1f}"
          f"{s['profit_factor']:>6.2f}{s['total_trades']:>7,}{s['avg_entry_adx']:>9.1f}"
          f"{s['pct_entries_adx_below_25']:>8.0f}%")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description="Walk-forward point-in-time screening backtest")
    ap.add_argument("--years",type=int,default=12)
    ap.add_argument("--equity",type=float,default=100_000.0)
    args=ap.parse_args()

    if config.USE_TREND_FILTER or config.USE_REGIME_FILTER:
        logger.warning("Weekly gate ON — daily-only engine; results approximate.")

    prep=prepare(args.years)
    n=prep["n"]; mid=(WF_START+n)//2
    print(f"\n{'='*104}\n  WALK-FORWARD (point-in-time screen, top {ACTIVE_TOP_K}, monthly) | "
          f"ATR-{STOP_MULT}x | {config.MAX_POSITIONS}pos/{config.MAX_PER_SECTOR}sector/"
          f"{config.MAX_POSITION_PCT:.0%}cap\n{'='*104}")
    print(f"  {'Period':<10}{'Window':<24}{'Return':>9}{'CAGR':>7}{'MaxDD':>8}{'Win%':>7}"
          f"{'PF':>6}{'Trades':>7}{'EntryADX':>9}{'<25':>8}")
    print("  "+"-"*100)
    t_full,s_full=run_engine(prep,WF_START,n,"Full",args.equity)
    t_is,s_is=run_engine(prep,WF_START,mid,"In-sample",args.equity)
    t_oos,s_oos=run_engine(prep,mid,n,"Out-sample",args.equity)
    _print("Full",s_full); _print("In-samp",s_is); _print("Out-samp",s_oos)
    print(f"{'='*104}\n")

    _save(t_full,"backtest_wf_trades.csv")
    _save([s_full,s_is,s_oos],"backtest_wf_summary.csv")
    print("✅ Done. → backtest_wf_{trades,summary}.csv")
