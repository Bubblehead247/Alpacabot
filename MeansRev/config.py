"""
config.py — Central configuration for the Mean Reversion Bot.
All tunable parameters live here. No need to touch other files for adjustments.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load credentials from ../alpaca.env (parent folder) if present
load_dotenv(Path(__file__).resolve().parent.parent / "alpaca.env")

# ── Alpaca Credentials ────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY", "YOUR_KEY_HERE")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_HERE")
PAPER      = True   # Set False when going live with real money

# ── Trading Universe ──────────────────────────────────────────────────────────
SYMBOLS = ["SPY", "QQQ", "IWM"]

# ── Indicator Parameters ──────────────────────────────────────────────────────
RSI_PERIOD          = 2
RSI_ENTRY_THRESHOLD = 10.0   # RSI(2) must be <= this to trigger entry
RSI_EXIT_THRESHOLD  = 70.0   # RSI(2) >= this triggers exit next open
SMA_PERIOD          = 200
ATR_PERIOD          = 14
LOOKBACK_DAYS       = 260    # Enough bars for SMA(200) + buffer

# ── Risk Management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.02        # 2% of account equity risked per trade

# ── Stop Loss Variants ────────────────────────────────────────────────────────
# Both are tracked in logs. Only ACTIVE_STOP_MULT is actually executed.
# Toggle between 1.5 and 2.5 to compare performance in paper trading.
STOP_MULT_A        = 1.5
STOP_MULT_B        = 2.5
ACTIVE_STOP_MULT   = 2.5     # ← Change to 1.5 to test the tighter stop variant

# ── Exit Rules ────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS = 7            # Time stop: force exit if trade is still open after 7 days

# ── Scheduling (Central Time — CST is ET minus 1 hour) ─────────────────────────────────────────────────
SCAN_TIME          = "15:30"  # Post-close scan       (4:30 PM ET)
EXECUTE_TIME       = "08:25"  # Pre-open order submit (9:25 AM ET)
FILL_CONFIRM_TIME  = "08:45"  # Fill confirm + stops  (9:45 AM ET)
