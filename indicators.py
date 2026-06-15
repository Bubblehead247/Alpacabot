"""indicators.py — re-exports the shared quantcore indicator library.

The implementations used to live here as pure-pandas functions. They now live in
the installed ``quantcore`` package so this bot, its backtester, and every other
project compute the exact same numbers and can never silently drift.

This module is a thin shim: existing imports such as
``from indicators import sma, rsi, atr, adx`` keep working unchanged. The only
behavior change from the old local code is the RSI zero-loss edge case — a
straight run-up (average loss = 0) now reads RSI = 100 instead of NaN, which is
the deliberate, project-wide convention.
"""

from quantcore.indicators import (  # noqa: F401  (re-exported for callers)
    adx,
    atr,
    efficiency_ratio,
    ema,
    rsi,
    sma,
    volume_ratio,
)

__all__ = ["sma", "ema", "rsi", "atr", "adx", "efficiency_ratio", "volume_ratio"]
