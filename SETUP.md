# Setup Guide — Claude Code + Mean Reversion Bot on Windows (PowerShell)

---

## Part 1: Install Claude Code

### Step 1 — Install Git for Windows (required)
Claude Code's native Windows installer depends on Git Bash under the hood.

1. Download from: https://git-scm.com/download/win
2. Run the installer — use all default options
3. The critical default: "Add Git to PATH" is pre-checked. Leave it checked.

### Step 2 — Install Claude Code via PowerShell

Open PowerShell **as Administrator** (right-click → "Run as administrator"), then run:

```powershell
irm https://claude.ai/install.ps1 | iex
```

This is the official one-line native installer. No Node.js required.

### Step 3 — Fix PATH if "claude" is not recognized

This is the most common issue on Windows. After the installer runs, if you open a new
PowerShell window and `claude --version` gives "not recognized", do this:

```powershell
# Add Claude Code to your PowerShell PATH permanently
$claudePath = "$env:USERPROFILE\.local\bin"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($currentPath -notlike "*$claudePath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$claudePath", "User")
    Write-Host "PATH updated. Close and reopen PowerShell, then run: claude --version"
} else {
    Write-Host "PATH already contains Claude. Try: claude --version"
}
```

Close PowerShell completely, reopen it, and verify:

```powershell
claude --version
```

### Step 4 — Authenticate
Run `claude` and follow the login prompts. You'll need a Claude Pro, Max, Team,
or Enterprise account — the free plan does not include Claude Code access.

---

## Part 2: Install Python and the Bot Dependencies

### Step 1 — Install Python (if not already installed)

Download Python 3.11+ from https://www.python.org/downloads/
During install: ✅ Check "Add Python to PATH" before clicking Install.

Verify in a new PowerShell window:
```powershell
python --version
pip --version
```

### Step 2 — Set up the bot

```powershell
# Navigate to wherever you saved the bot files
# Example — adjust this path to match where you actually saved the folder:
cd C:\Users\YourUsername\Documents\mean_reversion_bot

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Set your Alpaca API keys as environment variables

```powershell
# Set for current session only (use this to test):
$env:ALPACA_API_KEY    = "your_alpaca_key_here"
$env:ALPACA_SECRET_KEY = "your_alpaca_secret_here"

# Set permanently (recommended for production):
[Environment]::SetEnvironmentVariable("ALPACA_API_KEY",    "your_alpaca_key_here",    "User")
[Environment]::SetEnvironmentVariable("ALPACA_SECRET_KEY", "your_alpaca_secret_here", "User")
```

After setting permanently, close and reopen PowerShell for the variables to take effect.

---

## Part 3: Use Claude Code to Work on the Bot

Navigate to the bot folder and launch Claude Code:

```powershell
cd C:\Users\YourUsername\Documents\mean_reversion_bot
claude
```

Claude Code reads your entire codebase and you can talk to it naturally. Examples:

```
> explain how the position sizing works in risk.py
> run the backtest and show me the results
> the scanner is throwing an error, can you debug it?
> add a feature that sends me a text when a trade is placed
> what would I need to change to add TQQQ to the universe?
```

Claude Code can read, edit, and run files directly — you approve each change before it executes.

---

## Part 4: Run the Backtest

```powershell
# Default: SPY, QQQ, IWM — last 5 years
python backtest.py

# Custom date range
python backtest.py --start 2020-01-01

# Specific symbols only
python backtest.py --symbols SPY QQQ

# Different lookback
python backtest.py --years 3
```

Output files will appear in the same folder:
- `backtest_trades.csv`  — every trade with entry/exit/P&L details
- `backtest_summary.csv` — aggregated stats per symbol and stop variant

---

## Part 5: Run the Live Bot

**Use Python 3.14.** The shared `quantcore` package is installed for 3.14 only,
so launch the bot with the `py -3.14` selector — plain `python` may resolve to a
different version (e.g. 3.13) that lacks `quantcore` and will crash on startup.
Confirm first:

```powershell
py -3.14 -c "import quantcore, scanner; print('live import chain OK')"
```

```powershell
# Make sure you're in the bot directory with keys set, then:
py -3.14 main.py
```

The bot runs continuously. Logs go to both `bot.log` and the terminal.

To run it in the background (so it keeps running after you close PowerShell):

```powershell
Start-Process py -ArgumentList "-3.14","main.py" -WindowStyle Hidden
```

To see if it's running:
```powershell
Get-Process python
```

To stop it:
```powershell
Stop-Process -Name python
```

---

## Switching Stop Variants

To test Stop A (1.5×ATR), open `config.py` and change:
```python
ACTIVE_STOP_MULT = 1.5
```

To test Stop B (2.5×ATR):
```python
ACTIVE_STOP_MULT = 2.5
```

Then re-run the backtest or restart the live bot. Both stop prices are always
logged regardless of which one is active — so your comparison data accumulates automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `claude` not recognized | Run the PATH fix in Part 1 Step 3 |
| `pip` not found | Reinstall Python with "Add to PATH" checked |
| Alpaca API error 403 | Keys not set correctly — check env variables |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Bot fires at wrong time | You're in CST — times in config.py are already adjusted |
| Backtest has no trades | Try `--start 2018-01-01` for more data coverage |
