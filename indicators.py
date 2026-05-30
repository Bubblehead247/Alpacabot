"""
indicators.py — Pure pandas implementations of RSI, SMA, and ATR.
No external TA libraries needed — keeps dependencies minimal and calculations transparent.
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """
    Wilder's RSI using EWM smoothing (alpha = 1/period).
    RSI(2) is extremely sensitive — values below 10 represent severe short-term oversold conditions.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero on flat price series
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (span-based EWM)."""
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's EWM smoothing.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio — how 'trending' vs 'choppy' price is, in [0, 1].

    ER = |net change over N| / sum(|daily change|) over N.
    Near 1 = a clean directional trend; near 0 = sideways/noisy chop.
    Mean-reversion theory says we want LOW ER (range-bound) markets.
    """
    net_change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(window=period, min_periods=period).sum()
    return net_change / volatility.replace(0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index — trend STRENGTH (direction-agnostic), Wilder smoothing.

    ADX < ~20  → no trend / range-bound (favorable for mean reversion)
    ADX > ~25  → trending (unfavorable)
    """
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = ((up_move > down_move) & (up_move > 0)) * up_move.clip(lower=0)
    minus_dm  = ((down_move > up_move) & (down_move > 0)) * down_move.clip(lower=0)

    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha    = 1 / period
    atr_     = true_range.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()  / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
