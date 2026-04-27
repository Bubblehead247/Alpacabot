"""
dashboard.py — Streamlit read-only dashboard for the Mean Reversion bot.

Run with:
    streamlit run dashboard.py

Pure observation — never submits orders, never writes positions.json.
All data comes from Alpaca, the bot's own scanner, the local position tracker,
and bot.log. Reuses the bot's existing functions so the dashboard sees exactly
what the live bot sees.
"""
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

import config
import executor
import position_tracker as pt
import scanner

# Quiet down the bot's own loggers — we only want streamlit's chatter on stdout.
logging.getLogger().setLevel(logging.WARNING)

ET     = ZoneInfo("America/New_York")
LOG_FILE = Path(__file__).resolve().parent / "bot.log"

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeansRev Bot Dashboard",
    page_icon="📈",
    layout="wide",
)


# ── Cached data fetchers ──────────────────────────────────────────────────────
# st.cache_data(ttl=30) means each function is called at most once per 30s per
# unique args. Hitting "Refresh" clears the cache via st.cache_data.clear().
# This keeps the dashboard from hammering Alpaca on every Streamlit rerun.

@st.cache_data(ttl=30)
def fetch_account() -> dict:
    """Cash, equity, day P&L from Alpaca account."""
    acct = executor._client.get_account()
    return {
        "equity":           float(acct.equity),
        "cash":             float(acct.cash),
        "buying_power":     float(acct.buying_power),
        "last_equity":      float(acct.last_equity),       # equity at start of today
        "portfolio_value":  float(acct.portfolio_value),
    }


@st.cache_data(ttl=30)
def fetch_clock() -> dict:
    """Market open/closed + next session times."""
    clock = executor._client.get_clock()
    return {
        "is_open":   clock.is_open,
        "next_open": clock.next_open,
        "next_close": clock.next_close,
        "timestamp": clock.timestamp,
    }


@st.cache_data(ttl=30)
def fetch_is_trading_day() -> bool:
    """Whether the calendar shows a session for today."""
    return executor.is_trading_day_today()


@st.cache_data(ttl=30)
def fetch_alpaca_positions() -> dict:
    """Open positions keyed by symbol — Alpaca side."""
    return {
        sym: {
            "qty":             int(float(p.qty)),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price":   float(p.current_price) if p.current_price else None,
            "market_value":    float(p.market_value),
            "unrealized_pl":   float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        }
        for sym, p in executor.get_alpaca_positions().items()
    }


@st.cache_data(ttl=30)
def fetch_portfolio_history() -> pd.DataFrame:
    """1-month daily equity curve for the chart."""
    try:
        # alpaca-py wraps the underlying REST endpoint. Use raw client method.
        hist = executor._client.get_portfolio_history(
            period="1M",
            timeframe="1D",
        )
        # PortfolioHistory has parallel arrays: timestamp[], equity[]
        ts = pd.to_datetime(hist.timestamp, unit="s", utc=True).tz_convert(ET)
        return pd.DataFrame({"date": ts, "equity": hist.equity}).dropna()
    except Exception as e:
        # Some alpaca-py versions don't expose get_portfolio_history on the
        # trading client — this is a best-effort tile. Empty DF = no chart.
        st.session_state["_portfolio_history_error"] = str(e)
        return pd.DataFrame(columns=["date", "equity"])


@st.cache_data(ttl=30)
def fetch_recent_orders(limit: int = 100) -> list[dict]:
    """Closed (filled or cancelled) orders, newest first — Alpaca side."""
    orders = executor._client.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=limit,
        )
    )
    out = []
    for o in orders:
        # Only include actually-filled orders. Cancellations and rejections
        # would otherwise pollute the trade list with non-events.
        if not o.filled_at or not o.filled_avg_price:
            continue
        out.append({
            "filled_at":    o.filled_at,
            "symbol":       o.symbol,
            "side":         str(o.side).split(".")[-1].lower(),  # "buy"/"sell"
            "qty":          int(float(o.filled_qty)),
            "fill_price":   float(o.filled_avg_price),
            "order_id":     str(o.id),
        })
    return out


@st.cache_data(ttl=30)
def parse_log_for_exit_reasons() -> dict:
    """
    Build {(symbol, date_iso): reason} from bot.log SELL ORDER lines.

    The bot logs lines like:
      📤 SELL ORDER submitted | SPY | 12 shares | Reason=rsi_exit | Order ID=...
    We capture the symbol + date to later enrich Alpaca's filled-orders list
    with the bot's own classification of why each exit happened.
    """
    if not LOG_FILE.exists():
        return {}

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}).*SELL ORDER submitted \| (\S+) .*Reason=(\S+)"
    )
    reasons = {}
    with open(LOG_FILE) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                date_iso, symbol, reason = m.groups()
                reasons[(symbol, date_iso)] = reason
    return reasons


@st.cache_data(ttl=30)
def fetch_scan_snapshot() -> list[dict]:
    """
    Live RSI/SMA/ATR snapshot per symbol — same view the post-close scan uses.
    Calling scanner.evaluate_symbol triggers a fresh Alpaca bars fetch per call.
    """
    out = []
    for sym in config.SYMBOLS:
        try:
            r = scanner.evaluate_symbol(sym)
            out.append(r)
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})
    return out


def read_log_tail(n: int = 50) -> list[str]:
    """Last N lines of bot.log."""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return lines[-n:]


def bot_heartbeat() -> tuple[bool, str]:
    """
    Heuristic liveness check: is bot.log being touched recently?
    Returns (alive_bool, last_modified_human_string).
    Within 10 minutes = alive (the main loop sleeps 30s and most schedules
    log SOMETHING every few minutes once active).
    """
    if not LOG_FILE.exists():
        return False, "bot.log not found"
    mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime, tz=ET)
    age   = datetime.now(ET) - mtime
    alive = age < timedelta(minutes=10)
    return alive, mtime.strftime("%Y-%m-%d %H:%M:%S ET")


# ── Trade pairing (FIFO) ──────────────────────────────────────────────────────

def pair_trades(orders: list[dict], reasons: dict) -> pd.DataFrame:
    """
    Pair BUY orders with subsequent SELL orders FIFO per symbol so we can show
    realized P&L per round trip. Unpaired buys (still-open positions) are
    shown as 'OPEN'. Sells without a matching prior buy are shown as 'SELL'
    with no P&L (shouldn't normally happen).
    """
    # Sort oldest-first so FIFO pairing works.
    orders = sorted(orders, key=lambda o: o["filled_at"])

    open_lots: dict[str, list[dict]] = defaultdict(list)  # symbol -> [buy lots]
    rows = []

    for o in orders:
        sym  = o["symbol"]
        date = o["filled_at"].astimezone(ET).date().isoformat()

        if o["side"] == "buy":
            open_lots[sym].append(o)
            rows.append({
                "exit_date":   "",
                "entry_date":  date,
                "symbol":      sym,
                "qty":         o["qty"],
                "entry_price": o["fill_price"],
                "exit_price":  None,
                "pnl_dollars": None,
                "pnl_pct":     None,
                "reason":      "",
                "status":      "OPEN",
            })
        elif o["side"] == "sell":
            qty_to_close = o["qty"]
            entry_total  = 0.0
            shares_paired = 0
            entry_dates  = []
            # FIFO consume open buy lots
            while qty_to_close > 0 and open_lots[sym]:
                lot = open_lots[sym][0]
                take = min(lot["qty"], qty_to_close)
                entry_total += take * lot["fill_price"]
                shares_paired += take
                entry_dates.append(lot["filled_at"].astimezone(ET).date().isoformat())
                lot["qty"] -= take
                qty_to_close -= take
                if lot["qty"] == 0:
                    open_lots[sym].pop(0)
                    # Mark the OPEN row we previously emitted as CLOSED
                    for row in reversed(rows):
                        if (row["symbol"] == sym
                                and row["status"] == "OPEN"
                                and row["entry_date"] == entry_dates[-1]):
                            row["status"] = "CLOSED"
                            break

            avg_entry = entry_total / shares_paired if shares_paired else 0.0
            pnl_d = (o["fill_price"] - avg_entry) * shares_paired if shares_paired else 0.0
            pnl_p = ((o["fill_price"] - avg_entry) / avg_entry * 100) if avg_entry else 0.0

            rows.append({
                "exit_date":   date,
                "entry_date":  entry_dates[0] if entry_dates else "",
                "symbol":      sym,
                "qty":         shares_paired or o["qty"],
                "entry_price": round(avg_entry, 2) if avg_entry else None,
                "exit_price":  o["fill_price"],
                "pnl_dollars": round(pnl_d, 2) if shares_paired else None,
                "pnl_pct":     round(pnl_p, 2) if shares_paired else None,
                "reason":      reasons.get((sym, date), ""),
                "status":      "CLOSED" if shares_paired else "SELL_NO_MATCH",
            })

    # Newest first for display.
    return pd.DataFrame(rows[::-1])


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("📈 MeansRev Bot Dashboard")

# Manual refresh — clears all @st.cache_data results so the next render hits Alpaca.
top_left, top_right = st.columns([6, 1])
with top_right:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── 1. Top status bar ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    alive, hb_str = bot_heartbeat()
    if alive:
        st.success(f"🟢 Bot running\nLast heartbeat: {hb_str}")
    else:
        st.error(f"🔴 Bot offline\nLast log activity: {hb_str}")

with c2:
    try:
        clock = fetch_clock()
        is_td = fetch_is_trading_day()
        if clock["is_open"]:
            st.success(f"🟢 Market OPEN\nNext close: {clock['next_close'].astimezone(ET).strftime('%H:%M ET')}")
        else:
            label = "🟡 Market closed (trading day)" if is_td else "🔴 Market closed (no session today)"
            st.info(f"{label}\nNext open: {clock['next_open'].astimezone(ET).strftime('%a %Y-%m-%d %H:%M ET')}")
    except Exception as e:
        st.error(f"Clock fetch failed: {e}")

with c3:
    now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    st.metric("Now", now_et)


# ── 2. Account tile ───────────────────────────────────────────────────────────
st.divider()
st.subheader("💰 Account")

try:
    acct = fetch_account()
    day_pl = acct["equity"] - acct["last_equity"]
    day_pl_pct = (day_pl / acct["last_equity"] * 100) if acct["last_equity"] else 0.0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Equity", f"${acct['equity']:,.2f}")
    a2.metric("Cash", f"${acct['cash']:,.2f}")
    a3.metric("Buying power", f"${acct['buying_power']:,.2f}")
    a4.metric("Day P&L", f"${day_pl:+,.2f}", f"{day_pl_pct:+.2f}%")

    # Equity curve — 1-month daily.
    hist = fetch_portfolio_history()
    if not hist.empty:
        st.line_chart(hist.set_index("date")["equity"], height=200)
    else:
        st.caption("Equity curve unavailable (Alpaca portfolio history empty or API mismatch).")
except Exception as e:
    st.error(f"Account fetch failed: {e}")


# ── 3. Open positions ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Open Positions")

try:
    alpaca_pos = fetch_alpaca_positions()
    tracker_pos = pt.all_positions()

    if not alpaca_pos:
        st.info("No open positions.")
    else:
        rows = []
        for sym, ap in alpaca_pos.items():
            tp = tracker_pos.get(sym, {})
            days = pt.days_open(sym) if tp else None
            rows.append({
                "symbol":         sym,
                "qty":            ap["qty"],
                "entry":          round(ap["avg_entry_price"], 2),
                "current":        round(ap["current_price"], 2) if ap["current_price"] else None,
                "unrealized $":   round(ap["unrealized_pl"], 2),
                "unrealized %":   round(ap["unrealized_plpc"] * 100, 2),
                "active_stop":    tp.get("stop_price"),
                "stop_1.5×":      tp.get("stop_price_a"),
                "stop_2.5×":      tp.get("stop_price_b"),
                "days":           f"{days}/{config.MAX_HOLD_DAYS}" if days is not None else "—",
                "entry_date":     tp.get("entry_date", "—"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
except Exception as e:
    st.error(f"Positions fetch failed: {e}")


# ── 4. Today's scan signals (live recompute) ──────────────────────────────────
st.divider()
st.subheader("🎯 Today's Scan Signals")
st.caption("Live recompute of the same signal logic the post-close scan uses (RSI(2) ≤ 10 & Close > SMA(200) for entry, RSI(2) ≥ 70 for exit).")

try:
    snap = fetch_scan_snapshot()
    rows = []
    for r in snap:
        if r.get("error"):
            rows.append({"symbol": r["symbol"], "error": r["error"]})
            continue
        rows.append({
            "symbol":       r["symbol"],
            "close":        r["close"],
            "SMA(200)":     r["sma200"],
            "above_sma":    "✅" if r["above_sma200"] else "❌",
            "RSI(2)":       r["rsi2"],
            "ATR(14)":      r["atr14"],
            "stop_1.5×":    r["stop_a"],
            "stop_2.5×":    r["stop_b"],
            "entry?":       "📥 YES" if r["entry_signal"] else "—",
            "exit?":        "📤 YES" if r["exit_signal"] else "—",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
except Exception as e:
    st.error(f"Scan snapshot failed: {e}")


# ── 5. Recent trades (Alpaca + log enrichment) ────────────────────────────────
st.divider()
st.subheader("📜 Recent Trades")
st.caption("Filled orders from Alpaca paired FIFO; exit reasons from bot.log.")

try:
    orders = fetch_recent_orders(limit=100)
    reasons = parse_log_for_exit_reasons()
    trades_df = pair_trades(orders, reasons)

    if trades_df.empty:
        st.info("No filled orders yet.")
    else:
        # Show only the 30 most recent rows.
        st.dataframe(trades_df.head(30), width="stretch", hide_index=True)

        # Quick aggregate stats over the closed trades only.
        closed = trades_df[trades_df["status"] == "CLOSED"]
        if not closed.empty:
            wins = (closed["pnl_dollars"] > 0).sum()
            total = len(closed)
            win_rate = wins / total * 100
            total_pnl = closed["pnl_dollars"].sum()
            s1, s2, s3 = st.columns(3)
            s1.metric("Closed trades", total)
            s2.metric("Win rate", f"{win_rate:.1f}%")
            s3.metric("Realized P&L", f"${total_pnl:+,.2f}")
except Exception as e:
    st.error(f"Recent trades fetch failed: {e}")


# ── 6. Log tail ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 bot.log (last 50 lines)")

tail = read_log_tail(50)
if not tail:
    st.info("bot.log not yet present (run `python main.py` to start the bot).")
else:
    st.code("".join(tail), language="text")
