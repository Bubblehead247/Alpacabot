"""
config.py — Central configuration for the Mean Reversion Bot.
All tunable parameters live here. No need to touch other files for adjustments.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load credentials from alpaca.env in the same directory as this file
load_dotenv(Path(__file__).resolve().parent / "alpaca.env")

# ── Alpaca Credentials ────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY", "YOUR_KEY_HERE")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_HERE")
PAPER      = True   # Set False when going live with real money

# ── Trading Universe (Rulebook v1 — 16 ETFs) ──────────────────────────────────
SYMBOLS = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    # SPDR sector (XL family)
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]

# ── Indicator Parameters ──────────────────────────────────────────────────────
RSI_PERIOD          = 2
RSI_ENTRY_THRESHOLD = 10.0   # Entry fires when RSI(2) crosses BACK ABOVE this
RSI_EXIT_THRESHOLD  = 70.0   # Exit fires when RSI(2) crosses back below this
SMA_DAILY           = 50     # Daily SMA — price must be above this to enter
SMA_WEEKLY_FAST     = 50     # Weekly SMA — trend gate (1-year)
SMA_WEEKLY_SLOW     = 200    # Weekly SMA — trend gate (4-year)
ATR_PERIOD          = 14
VOLUME_MA_PERIOD    = 20     # Period for volume moving average
VOLUME_SPIKE_MULT   = 1.5    # When the filter is ON: volume must exceed this × 20-day average
# Volume-spike entry confirmation. Backtest (2010–2026, 16 ETFs) showed that
# requiring a volume spike on the exact RSI-cross-above day removes ~90% of
# signals and cuts ~12yr return from +51% to +11%. Left OFF by default; flip to
# True to re-test. Volume is still computed and logged either way.
USE_VOLUME_FILTER   = False
LOOKBACK_DAYS       = 100    # Daily bars (SMA50=50 + ATR14 + volume20 + buffer)
WEEKLY_LOOKBACK_WEEKS = 220  # Weekly bars (SMA200=200 + buffer)

# ── Risk Management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.01        # 1% of account equity risked per trade

# ── Limit Order Slippage Controls ─────────────────────────────────────────────
ENTRY_LIMIT_PCT = 0.005      # Max we'll pay above prior close for a LOO buy (0.5%)
EXIT_LIMIT_PCT  = 0.005      # Min we'll accept below prior close for a LOO sell (0.5%)

# ── Backtest Slippage Model ───────────────────────────────────────────────────
# Per-side slippage the BACKTESTER applies to model bid/ask spread + market
# impact on market-on-open and stop fills. Buys fill SLIPPAGE_PCT above the
# reference price, sells fill SLIPPAGE_PCT below it (a round trip costs ~2×).
# 0.0005 = 5 bps/side. Alpaca is commission-free, so spread/impact is the
# dominant friction. Set to 0.0 for a frictionless backtest.
# NOTE: only consumed by backtest.py — does not affect live order placement.
SLIPPAGE_PCT = 0.0005

# ── Stop Loss Variants ────────────────────────────────────────────────────────
# Both are tracked in logs. Only ACTIVE_STOP_MULT is actually executed.
# Toggle between 1.5 and 2.5 to compare performance in paper trading.
STOP_MULT_A      = 1.5
STOP_MULT_B      = 2.5
ACTIVE_STOP_MULT = 2.5       # ← Change to 1.5 to test the tighter stop variant

# ── Regime Filter (experimental — backtest measurement only) ──────────────────
# Gate entries on weekly trend STRENGTH. Validation (analysis.py) found the edge
# is positive in BOTH in/out-of-sample halves only in a moderate-trend band:
# dead-sideways (ADX<20) and runaway trends (ADX>=25) both underperformed.
# Enter only when weekly ADX is in [MIN, MAX). Enforced in BOTH backtest.py and
# scanner.py (live). Validation: filtered edge is positive in both in/out-of-
# sample halves and ~3× lower drawdown, but keeps only ~26% of signals and its
# bootstrap CI still includes zero — promising, not statistically proven.
USE_REGIME_FILTER = True
REGIME_ADX_PERIOD = 14
REGIME_ADX_MIN    = 20.0
REGIME_ADX_MAX    = 25.0

# ── ntfy.sh Push Alerts ───────────────────────────────────────────────────────
NTFY_TOPIC = "MeansRevRSI"  # ntfy.sh topic — subscribe to this in the ntfy app

# ── Exit Rules ────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS = 7            # Time stop: force exit if trade is still open after 7 days
MAX_POSITIONS = 5            # Max concurrent positions (5% total risk cap at 1% each)

# ── Scheduling (Central Time — CST is ET minus 1 hour) ───────────────────────
SCAN_TIME         = "15:30"  # Post-close scan       (4:30 PM ET)
EXECUTE_TIME      = "08:25"  # Pre-open order submit (9:25 AM ET)
FILL_CONFIRM_TIME = "08:45"  # Fill confirm + stops  (9:45 AM ET)
