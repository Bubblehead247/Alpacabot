# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Run (Windows)
run_bot.bat
# or directly
python strategy1.py
```

The bot runs continuously, checking `schedule.run_pending()` every 30 seconds.

## Environment Setup

Credentials are loaded from `alpaca.env` via `python-dotenv`. Required variables:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
- `ALPACA_API_BASE_URL`
- `EMAIL_FROM`, `EMAIL_TO`, `EMAIL_PASSWORD` (Gmail SMTP)

The `PAPER = True` flag at the top of `strategy1.py` controls paper vs. live trading. Flip to `False` only when ready for live trading.

## Architecture

`strategy1.py` is a single-file trading bot with two sleeves:

**Core sleeve (DCA)** — Monthly buys of fixed-dollar amounts into 6 holdings across 3 themes (Nuclear/SMR, AI/Semiconductors, Crypto-adjacent). Runs on the 1st of each month at 10:00 AM ET.

**Swing sleeve** — Active momentum trading on `SWING_STOCKS` and `SWING_CRYPTO`. Entry signal: RSI > 55 AND volume > 20-day average. Exits via stop loss (−8%), take profit (+15%), or 14-day max hold. Max 3 concurrent swing positions at $50 each.

**Schedule** (all times ET, weekdays only):
- 10:30 AM — `check_swing_entries()`
- 3:45 PM — `check_swing_exits()`
- 4:00 PM — `print_portfolio_summary()` (emails daily summary)

**Key design notes:**
- Core holdings are never touched by swing exit logic — `core_syms` exclusion guards `check_swing_exits()`.
- All orders are notional (fractional) market orders via `place_notional_order()`.
- `get_rsi_and_volume()` fetches 60 days of daily bars from Alpaca's historical API and uses the `ta` library for RSI calculation.
- Logging goes to both `strategy1.log` and stdout.
- Email alerts fire on: swing entry, swing exit, rebalance warning, and daily portfolio summary.
