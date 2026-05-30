# Mean Reversion Bot — 16-ETF Universe

Connors-style RSI(2) mean reversion system with multi-timeframe trend filtering and volume confirmation.

## Trading Universe

**Broad market (5):** SPY, QQQ, IWM, DIA, MDY

**SPDR Sector / XL family (11):** XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU

---

## Strategy Logic

**Entry (all four must be true on the same day):**
- Weekly gate: close above SMA(50,W) AND SMA(200,W) — structural trend must be intact
- Daily trend: close above SMA(50,D)
- RSI signal: RSI(2) crossed back **above** 10 today (prior bar ≤ 10, current bar > 10) — bounce confirmed, not catching a falling knife
- Volume spike: today's volume > 1.5× the 20-day average — institutional participation required

**Exit (first condition hit wins, in priority order):**
- Hard stop: ATR-based GTC stop order on exchange — executes intraday, no bot required
- Time stop: position held ≥ 7 calendar days → limit sell next open, market fallback by 09:45 ET
- Weekly trend break: close drops below weekly SMA(200) → limit sell next open, market fallback by 09:45 ET
- RSI target: RSI(2) crosses back below 70 (prior bar ≥ 70, current bar < 70) → limit sell next open, market fallback by 09:45 ET

**Order execution:**
- Entries: LOO limit at prior close + 0.5% submitted at 09:25 ET; unfilled limits replaced with DAY market order at 09:45 ET
- Exits: LOO limit at prior close − 0.5% submitted at 09:25 ET; unfilled limits replaced with DAY market sell at 09:45 ET

**Risk:**
- 1% of account equity risked per trade (hard max — never exceeded)
- Maximum 5 concurrent positions (5% total account risk cap)
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

### 2. Set credentials
Create `alpaca.env` in the project directory:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### 3. Configure
Edit `config.py` to adjust:
- `ACTIVE_STOP_MULT` — 1.5 or 2.5 (which stop variant to actually trade)
- `PAPER` — True for paper, False for live
- `SYMBOLS` — add/remove tickers
- `RISK_PER_TRADE` — default 0.01 (1%)
- `MAX_POSITIONS` — default 5 (max concurrent trades)

### 4. Run
```bash
python main.py
```

The bot runs continuously and logs to both `bot.log` and stdout.

---

## Daily Schedule (Eastern Time)

| Time  | Job                  | Description                                                              |
|-------|----------------------|--------------------------------------------------------------------------|
| 16:30 | Post-close scan      | Evaluate all 16 symbols, queue entry/exit actions                        |
| 09:25 | Pre-open execute     | Submit LOO limit orders (entries + exits) for next open                  |
| 09:45 | Fill confirm + stops | Verify fills, place GTC hard stops; replace unfilled limits with market  |

---

## File Structure

```
MeansRev/
├── config.py           # All tunable parameters — start here
├── indicators.py       # RSI, SMA, ATR calculations (pure pandas)
├── scanner.py          # Fetches daily + weekly bars, evaluates all signals
├── risk.py             # Position sizing math
├── position_tracker.py # JSON persistence for open position metadata
├── executor.py         # All Alpaca API interactions and order logic
├── main.py             # Scheduler and main loop
├── trade_log.py        # Appends closed trades to trades.csv
├── notifier.py         # ntfy.sh push alerts
├── requirements.txt    # Python dependencies
├── positions.json      # Auto-generated — tracks open positions
├── trades.csv          # Auto-generated — closed trade history
└── old_code/           # Previous version of all bot files
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
