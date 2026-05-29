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
SYMBOLS = [
    # Broad US equity
    "SPY", "QQQ", "IWM",
    # US sectors
    "XLF", "XLE", "XLV", "XLU", "XLI", "XLB", "XLRE",
    # Financials (regional banks)
    "KRE",
    # International equity
    "EEM", "EFA", "EWZ", "EWJ", "FXI",
    # Commodities / alternatives
    "GDX", "SLV",
]

# ── Indicator Parameters ──────────────────────────────────────────────────────
RSI_PERIOD          = 2
RSI_ENTRY_THRESHOLD = 10.0   # RSI(2) must be <= this to trigger entry
RSI_EXIT_THRESHOLD  = 70.0   # RSI(2) >= this triggers exit next open
SMA_PERIOD          = 200
EMA_PERIOD          = 50
ATR_PERIOD          = 14
LOOKBACK_DAYS       = 260    # Enough bars for SMA(200) + buffer

# ── Risk Management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.01        # 1% of account equity risked per trade

# ── Limit Order Slippage Controls ─────────────────────────────────────────────
ENTRY_LIMIT_PCT = 0.005      # max we'll pay above prior close for a LOO buy (0.5%)
EXIT_LIMIT_PCT  = 0.005      # min we'll accept below prior close for a LOO sell (0.5%)

# ── Stop Loss Variants ────────────────────────────────────────────────────────
# Both are tracked in logs. Only ACTIVE_STOP_MULT is actually executed.
# Toggle between 1.5 and 2.5 to compare performance in paper trading.
STOP_MULT_A        = 1.5
STOP_MULT_B        = 2.5
ACTIVE_STOP_MULT   = 2.5     # ← Change to 1.5 to test the tighter stop variant

# ── ntfy.sh Push Alerts ───────────────────────────────────────────────────────
# Subscribe to this topic in the ntfy app on your phone to receive alerts.
# Pick a unique name — anyone who knows it can subscribe (public service).
NTFY_TOPIC = "MeansRevRSI"  # ntfy.sh topic — subscribe to this in the ntfy app

# ── Exit Rules ────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS  = 7           # Time stop: force exit if trade is still open after 7 days
MAX_POSITIONS  = 5           # Max concurrent open positions (total risk cap = MAX_POSITIONS × RISK_PER_TRADE)

# ── Scheduling (Central Time — CST is ET minus 1 hour) ─────────────────────────────────────────────────
SCAN_TIME          = "15:30"  # Post-close scan       (4:30 PM ET)
EXECUTE_TIME       = "08:25"  # Pre-open order submit (9:25 AM ET)
FILL_CONFIRM_TIME  = "08:45"  # Fill confirm + stops  (9:45 AM ET)
