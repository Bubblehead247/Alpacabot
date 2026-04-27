"""
backtest.py — Historical backtester for the Mean Reversion Strategy.

Tests both stop variants (1.5× and 2.5× ATR) independently on the same
historical data so you can compare their performance side by side.

Data source: Alpaca historical daily bars (same feed the live bot uses).

Usage:
    python backtest.py                        # Default: all symbols, 5 years
    python backtest.py --symbols SPY QQQ      # Specific symbols
    python backtest.py --years 3              # Shorter lookback
    python backtest.py --start 2018-01-01     # Custom start date

Output:
    - Console summary per symbol + stop variant
    - backtest_trades.csv  — every trade with entry/exit details
    - backtest_summary.csv — aggregated stats per symbol/variant
"""

import argparse
import csv
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config
from indicators import sma, rsi, atr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

_data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:         str
    stop_variant:   str          # "1.5x" or "2.5x"
    entry_date:     str
    entry_price:    float
    exit_date:      str
    exit_price:     float
    shares:         int
    exit_reason:    str          # "rsi_exit", "stop_loss", "time_stop"
    stop_price:     float
    pnl_dollars:    float = 0.0
    pnl_pct:        float = 0.0
    hold_days:      int   = 0


@dataclass
class BacktestResult:
    symbol:          str
    stop_variant:    str
    start_date:      str
    end_date:        str
    initial_equity:  float
    final_equity:    float
    total_trades:    int   = 0
    winning_trades:  int   = 0
    losing_trades:   int   = 0
    win_rate:        float = 0.0
    avg_win_pct:     float = 0.0
    avg_loss_pct:    float = 0.0
    profit_factor:   float = 0.0
    max_drawdown_pct:float = 0.0
    cagr_pct:        float = 0.0
    total_return_pct:float = 0.0
    avg_hold_days:   float = 0.0
    rsi_exits:       int   = 0
    stop_exits:      int   = 0
    time_exits:      int   = 0
    trades:          list  = field(default_factory=list)


# ── Historical Data ───────────────────────────────────────────────────────────

def fetch_history_alpaca(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily bars from Alpaca (free tier = IEX, ~5yr history)."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = _data_client.get_stock_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")

    return bars.sort_index()


def fetch_history_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch split/dividend-adjusted daily bars from Yahoo Finance (decades of history)."""
    df = yf.download(
        symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def fetch_history(symbol: str, start: datetime, end: datetime, source: str = "yfinance") -> pd.DataFrame:
    if source == "alpaca":
        return fetch_history_alpaca(symbol, start, end)
    return fetch_history_yfinance(symbol, start, end)


# ── Core Backtest Engine ──────────────────────────────────────────────────────

def backtest_symbol(
    symbol: str,
    bars: pd.DataFrame,
    stop_mult: float,
    initial_equity: float = 100_000.0,
) -> BacktestResult:
    """
    Simulate the strategy on a single symbol with a given stop multiplier.

    Rules simulated:
      Entry:    Close > SMA(200) AND RSI(2) <= 10 → buy next open (approximated as next close)
      Exit 1:   RSI(2) >= 70 at close → sell next open
      Exit 2:   Intraday low touches stop price → stopped out at stop price (worst case)
      Exit 3:   Position held >= MAX_HOLD_DAYS → sell at close
    """
    variant_label = f"{stop_mult}x"
    logger.info(f"Backtesting {symbol} | Stop: {stop_mult}×ATR | Bars: {len(bars)}")

    close  = bars["close"]
    high   = bars["high"]
    low    = bars["low"]
    opens  = bars["open"]
    dates  = bars.index

    sma200 = sma(close, config.SMA_PERIOD)
    rsi2   = rsi(close, config.RSI_PERIOD)
    atr14  = atr(high, low, close, config.ATR_PERIOD)

    equity        = initial_equity
    equity_curve  = []
    trades        = []
    in_position   = False
    entry_price   = 0.0
    entry_date    = None
    entry_idx     = 0
    shares        = 0
    stop_price    = 0.0
    peak_equity   = initial_equity
    max_drawdown  = 0.0

    for i in range(config.SMA_PERIOD + 5, len(bars)):
        date_str   = str(dates[i].date())
        day_open   = float(opens.iloc[i])
        day_close  = float(close.iloc[i])
        day_low    = float(low.iloc[i])
        day_high   = float(high.iloc[i])
        cur_rsi    = float(rsi2.iloc[i])
        cur_sma    = float(sma200.iloc[i])
        cur_atr    = float(atr14.iloc[i])

        # ── Check Exits First (if in position) ──────────────────────────────
        if in_position:
            days_held  = i - entry_idx
            exit_price = None
            exit_reason = None

            # Priority 1: Hard stop — did intraday low breach stop price?
            if day_low <= stop_price:
                exit_price  = stop_price   # Assume filled at stop (conservative)
                exit_reason = "stop_loss"

            # Priority 2: Time stop
            elif days_held >= config.MAX_HOLD_DAYS:
                exit_price  = day_open     # Exit at open on day 7
                exit_reason = "time_stop"

            # Priority 3: RSI exit (signal from prior close, exit at today's open)
            elif float(rsi2.iloc[i - 1]) >= config.RSI_EXIT_THRESHOLD:
                exit_price  = day_open
                exit_reason = "rsi_exit"

            if exit_price is not None:
                pnl_dollars = (exit_price - entry_price) * shares
                pnl_pct     = ((exit_price - entry_price) / entry_price) * 100
                equity      += pnl_dollars

                trade = Trade(
                    symbol=symbol,
                    stop_variant=variant_label,
                    entry_date=str(entry_date),
                    entry_price=round(entry_price, 2),
                    exit_date=date_str,
                    exit_price=round(exit_price, 2),
                    shares=shares,
                    exit_reason=exit_reason,
                    stop_price=round(stop_price, 2),
                    pnl_dollars=round(pnl_dollars, 2),
                    pnl_pct=round(pnl_pct, 2),
                    hold_days=days_held,
                )
                trades.append(trade)
                in_position = False

        # ── Check Entry (only if flat) ───────────────────────────────────────
        if not in_position:
            above_sma = day_close > cur_sma
            oversold  = cur_rsi   <= config.RSI_ENTRY_THRESHOLD

            if above_sma and oversold and not pd.isna(cur_atr):
                # Enter at next bar's open — use next day's open if available
                if i + 1 < len(bars):
                    entry_price  = float(opens.iloc[i + 1])
                    entry_date   = dates[i + 1].date()
                    entry_idx    = i + 1
                else:
                    continue

                est_stop     = entry_price - (stop_mult * cur_atr)
                stop_dist    = entry_price - est_stop

                if stop_dist <= 0:
                    continue

                dollar_risk  = equity * config.RISK_PER_TRADE
                shares       = int(dollar_risk / stop_dist)

                if shares <= 0:
                    continue

                stop_price   = round(est_stop, 2)
                in_position  = True

        # ── Track Equity Curve + Drawdown ────────────────────────────────────
        # Mark-to-market if in position
        current_value = equity
        if in_position:
            current_value = equity + ((day_close - entry_price) * shares)

        equity_curve.append(current_value)

        if current_value > peak_equity:
            peak_equity = current_value

        drawdown = (peak_equity - current_value) / peak_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # ── Aggregate Stats ───────────────────────────────────────────────────────
    winners    = [t for t in trades if t.pnl_dollars > 0]
    losers     = [t for t in trades if t.pnl_dollars <= 0]
    win_rate   = len(winners) / len(trades) * 100 if trades else 0

    avg_win    = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
    avg_loss   = sum(t.pnl_pct for t in losers)  / len(losers)  if losers  else 0

    gross_wins  = sum(t.pnl_dollars for t in winners)
    gross_loss  = abs(sum(t.pnl_dollars for t in losers))
    pf          = gross_wins / gross_loss if gross_loss > 0 else float("inf")

    total_return = (equity - initial_equity) / initial_equity * 100

    # CAGR — annualized from actual number of trading days
    n_bars  = len(bars) - config.SMA_PERIOD - 5
    n_years = n_bars / 252
    cagr    = ((equity / initial_equity) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    avg_hold = sum(t.hold_days for t in trades) / len(trades) if trades else 0

    return BacktestResult(
        symbol=symbol,
        stop_variant=variant_label,
        start_date=str(bars.index[config.SMA_PERIOD + 5].date()),
        end_date=str(bars.index[-1].date()),
        initial_equity=initial_equity,
        final_equity=round(equity, 2),
        total_trades=len(trades),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=round(win_rate, 1),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        profit_factor=round(pf, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        cagr_pct=round(cagr, 2),
        total_return_pct=round(total_return, 2),
        avg_hold_days=round(avg_hold, 1),
        rsi_exits=sum(1 for t in trades if t.exit_reason == "rsi_exit"),
        stop_exits=sum(1 for t in trades if t.exit_reason == "stop_loss"),
        time_exits=sum(1 for t in trades if t.exit_reason == "time_stop"),
        trades=trades,
    )


# ── Output ────────────────────────────────────────────────────────────────────

def print_result(r: BacktestResult):
    print(f"\n{'=' * 60}")
    print(f"  {r.symbol}  |  Stop: {r.stop_variant} ATR  |  {r.start_date} → {r.end_date}")
    print(f"{'=' * 60}")
    print(f"  Equity:       ${r.initial_equity:>10,.2f}  →  ${r.final_equity:>10,.2f}")
    print(f"  Total Return: {r.total_return_pct:>+.1f}%   |   CAGR: {r.cagr_pct:>+.1f}%")
    print(f"  Max Drawdown: {r.max_drawdown_pct:.1f}%")
    print(f"{'─' * 60}")
    print(f"  Trades:       {r.total_trades}  |  Win Rate: {r.win_rate:.1f}%")
    print(f"  Avg Win:      +{r.avg_win_pct:.2f}%   |   Avg Loss: {r.avg_loss_pct:.2f}%")
    print(f"  Profit Factor:{r.profit_factor:.2f}")
    print(f"  Avg Hold:     {r.avg_hold_days:.1f} days")
    print(f"{'─' * 60}")
    print(f"  Exit Reasons: RSI={r.rsi_exits}  Stop={r.stop_exits}  Time={r.time_exits}")
    print(f"{'=' * 60}\n")


def save_trades_csv(all_trades: list[Trade], filename: str = "backtest_trades.csv"):
    if not all_trades:
        logger.warning("No trades to save.")
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(all_trades[0]).keys())
        writer.writeheader()
        writer.writerows([asdict(t) for t in all_trades])
    logger.info(f"Trades saved → {filename} ({len(all_trades)} rows)")


def save_summary_csv(results: list[BacktestResult], filename: str = "backtest_summary.csv"):
    rows = []
    for r in results:
        rows.append({
            "symbol":          r.symbol,
            "stop_variant":    r.stop_variant,
            "start_date":      r.start_date,
            "end_date":        r.end_date,
            "total_trades":    r.total_trades,
            "win_rate_pct":    r.win_rate,
            "avg_win_pct":     r.avg_win_pct,
            "avg_loss_pct":    r.avg_loss_pct,
            "profit_factor":   r.profit_factor,
            "cagr_pct":        r.cagr_pct,
            "total_return_pct":r.total_return_pct,
            "max_drawdown_pct":r.max_drawdown_pct,
            "avg_hold_days":   r.avg_hold_days,
            "rsi_exits":       r.rsi_exits,
            "stop_exits":      r.stop_exits,
            "time_exits":      r.time_exits,
            "final_equity":    r.final_equity,
        })
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary saved → {filename}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_backtest(
    symbols: list[str] = None,
    years: int = 5,
    start_date: str = None,
    initial_equity: float = 100_000.0,
    source: str = "yfinance",
):
    symbols = symbols or config.SYMBOLS

    end   = datetime.now(timezone.utc)
    if start_date:
        start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=years * 365 + 60)  # +60 buffer for SMA warmup

    logger.info(f"Backtest | Source: {source} | Symbols: {symbols} | {start.date()} → {end.date()}")

    all_results = []
    all_trades  = []

    for symbol in symbols:
        logger.info(f"Fetching history for {symbol}...")
        try:
            bars = fetch_history(symbol, start, end, source=source)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue

        if len(bars) < config.SMA_PERIOD + 20:
            logger.warning(f"{symbol}: Not enough historical data ({len(bars)} bars). Skipping.")
            continue

        for stop_mult in [config.STOP_MULT_A, config.STOP_MULT_B]:
            result = backtest_symbol(symbol, bars, stop_mult, initial_equity)
            print_result(result)
            all_results.append(result)
            all_trades.extend(result.trades)

    if all_results:
        save_trades_csv(all_trades)
        save_summary_csv(all_results)
        print(f"\n✅ Backtest complete. {len(all_trades)} total trades across {len(all_results)} symbol/variant combos.")
        print(f"   → backtest_trades.csv\n   → backtest_summary.csv")
    else:
        print("No results generated. Check your API keys and symbol list.")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean Reversion Strategy Backtester")
    parser.add_argument("--symbols", nargs="+", default=None,     help="Symbols to test (default: config.SYMBOLS)")
    parser.add_argument("--years",   type=int,   default=5,        help="Years of history (default: 5)")
    parser.add_argument("--start",   type=str,   default=None,     help="Start date YYYY-MM-DD (overrides --years)")
    parser.add_argument("--equity",  type=float, default=100_000,  help="Starting equity (default: 100000)")
    parser.add_argument("--source",  choices=["yfinance", "alpaca"], default="yfinance", help="Data source (default: yfinance)")
    args = parser.parse_args()

    run_backtest(
        symbols=args.symbols,
        years=args.years,
        start_date=args.start,
        initial_equity=args.equity,
        source=args.source,
    )
