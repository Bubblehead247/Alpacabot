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

# ── Stop Loss Variants ────────────────────────────────────────────────────────
# Both are tracked in logs. Only ACTIVE_STOP_MULT is actually executed.
# Toggle between 1.5 and 2.5 to compare performance in paper trading.
STOP_MULT_A      = 1.5
STOP_MULT_B      = 2.5
ACTIVE_STOP_MULT = 2.5       # ← Change to 1.5 to test the tighter stop variant

# ── ntfy.sh Push Alerts ───────────────────────────────────────────────────────
NTFY_TOPIC = "MeansRevRSI"  # ntfy.sh topic — subscribe to this in the ntfy app

# ── Exit Rules ────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS = 7            # Time stop: force exit if trade is still open after 7 days
MAX_POSITIONS = 5            # Max concurrent positions (5% total risk cap at 1% each)

# ── Scheduling (Central Time — CST is ET minus 1 hour) ───────────────────────
SCAN_TIME         = "15:30"  # Post-close scan       (4:30 PM ET)
EXECUTE_TIME      = "08:25"  # Pre-open order submit (9:25 AM ET)
FILL_CONFIRM_TIME = "08:45"  # Fill confirm + stops  (9:45 AM ET)
