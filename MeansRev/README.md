# Mean Reversion Bot — SPY / QQQ / IWM

Connors-style RSI(2) mean reversion system targeting oversold pullbacks in bull-market conditions.

## Strategy Logic

**Entry:**
- Close must be above SMA(200) — confirms bull market regime
- RSI(2) must be ≤ 10 — captures maximum rubber-band stretch, not mild dips

**Exit (first condition hit wins):**
- RSI(2) ≥ 70 at close → market sell next open
- Position held ≥ 7 calendar days → time stop, market sell next open
- Hard stop (ATR-based) hit intraday → exchange-level stop order executes immediately

**Risk:**
- 1% of account equity risked per trade (hard max — never exceeded)
- Position size = floor( equity × 0.01 / stop_distance )

**Stop Variants (both logged, one executed):**
- Stop A: 1.5 × ATR(14) below entry
- Stop B: 2.5 × ATR(14) below entry
- Toggle `ACTIVE_STOP_MULT` in config.py to switch which is executed

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

Or edit `config.py` directly (not recommended for production).

### 3. Configure
Edit `config.py` to adjust:
- `ACTIVE_STOP_MULT` — 1.5 or 2.5 (which stop variant to actually trade)
- `PAPER` — True for paper, False for live
- `SYMBOLS` — add/remove tickers
- `RISK_PER_TRADE` — default 0.01 (1%)

### 4. Run
```bash
python main.py
```

The bot runs continuously and logs to both `bot.log` and stdout.

---

## Daily Schedule (Eastern Time)

| Time  | Job                   | Description                                                |
|-------|-----------------------|------------------------------------------------------------|
| 16:30 | Post-close scan       | Evaluate all symbols, queue entry/exit actions             |
| 09:25 | Pre-open execute      | Submit Market OPG orders (fills at opening auction)        |
| 09:45 | Fill confirm + stops  | Verify fills, place GTC hard stop orders on the exchange   |

---

## File Structure

```
mean_reversion_bot/
├── config.py           # All tunable parameters — start here
├── indicators.py       # RSI, SMA, ATR calculations (pure pandas)
├── scanner.py          # Fetches bars, evaluates entry/exit signals
├── risk.py             # Position sizing math
├── position_tracker.py # JSON persistence for open position metadata
├── executor.py         # All Alpaca API interactions
├── main.py             # Scheduler and main loop
├── requirements.txt    # Python dependencies
└── positions.json      # Auto-generated — tracks open positions
```

---

## Comparing Stop Variants

Both stop prices (1.5× and 2.5× ATR) are logged every day in `bot.log` for every position.
To compare, grep the logs:

```bash
grep "Stop_A\|Stop_B" bot.log
```

Run the bot for a full market cycle on each variant before drawing conclusions.
The tighter stop (1.5×) will have more frequent stop-outs but smaller losses per stop.
The wider stop (2.5×) will survive more washouts but take bigger hits when trend is genuinely breaking.

---

## Going Live

1. Set `PAPER = False` in config.py
2. Fund the Alpaca live account
3. Update API keys to live keys
4. Ensure the server running this bot is in US/Eastern timezone (or adjust schedule times)
5. Verify the first few orders manually before leaving it unattended
