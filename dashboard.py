"""
Alpaca Paper Trading Dashboard
================================
Run with:  streamlit run dashboard.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv("alpaca.env")

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
PAPER      = True

CORE_HOLDINGS = {"LEU", "CCJ", "NVDA", "TSM", "COIN", "MSTR"}

THEMES = {
    "Nuclear / SMR":       ["LEU",  "CCJ"],
    "AI / Semiconductors": ["NVDA", "TSM"],
    "Crypto-adjacent":     ["COIN", "MSTR"],
}

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Alpaca Dashboard", page_icon="📈", layout="wide")
st.title("📈 Alpaca Paper Trading Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_data():
    client    = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    account   = client.get_account()
    positions = client.get_all_positions()
    orders    = client.get_orders()
    clock     = client.get_clock()
    return account, positions, orders, clock

try:
    account, positions, orders, clock = load_data()
except Exception as e:
    st.error(f"Failed to connect to Alpaca: {e}")
    st.stop()

# ── Derived values ────────────────────────────────────────────────────────────

portfolio_value = float(account.portfolio_value)
cash            = float(account.cash)
day_pnl         = float(account.equity) - float(account.last_equity)
day_pnl_pct     = day_pnl / float(account.last_equity) * 100 if float(account.last_equity) else 0

# ── Account overview ──────────────────────────────────────────────────────────

market_status = "🟢 Open" if clock.is_open else "🔴 Closed"
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfolio Value", f"${portfolio_value:,.2f}")
col2.metric("Cash", f"${cash:,.2f}")
col3.metric("Day P&L", f"${day_pnl:+,.2f}", f"{day_pnl_pct:+.2f}%")
col4.metric("Buying Power", f"${float(account.buying_power):,.2f}")
col5.metric("Market", market_status)

st.divider()

# ── Positions ─────────────────────────────────────────────────────────────────

left, right = st.columns([2, 1])

with left:
    st.subheader("Positions")
    if not positions:
        st.info("No open positions.")
    else:
        rows = []
        for p in sorted(positions, key=lambda x: x.symbol):
            entry   = float(p.avg_entry_price)
            current = float(p.current_price)
            pnl_pct = (current - entry) / entry * 100
            rows.append({
                "Symbol":    p.symbol,
                "Sleeve":    "Core" if p.symbol in CORE_HOLDINGS else "Swing",
                "Qty":       float(p.qty),
                "Entry $":   entry,
                "Current $": current,
                "Mkt Value": float(p.market_value),
                "P&L $":     float(p.unrealized_pl),
                "P&L %":     pnl_pct,
            })

        df = pd.DataFrame(rows)

        def color_pnl(val):
            color = "green" if val >= 0 else "red"
            return f"color: {color}"

        styled = (
            df.style
            .format({
                "Qty":       "{:.4f}",
                "Entry $":   "${:.2f}",
                "Current $": "${:.2f}",
                "Mkt Value": "${:,.2f}",
                "P&L $":     "${:+,.2f}",
                "P&L %":     "{:+.2f}%",
            })
            .applymap(color_pnl, subset=["P&L $", "P&L %"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Theme allocation ──────────────────────────────────────────────────────────

with right:
    st.subheader("Theme Allocation")
    pos_map = {p.symbol: float(p.market_value) for p in positions}
    invested = sum(pos_map.values())

    if invested == 0:
        st.info("No positions to allocate.")
    else:
        theme_data = []
        for theme, syms in THEMES.items():
            val = sum(pos_map.get(s, 0) for s in syms)
            theme_data.append({"Theme": theme, "Value": val, "Pct": val / invested * 100})

        swing_val = sum(v for s, v in pos_map.items() if s not in CORE_HOLDINGS)
        theme_data.append({"Theme": "Swing", "Value": swing_val, "Pct": swing_val / invested * 100})

        theme_df = pd.DataFrame(theme_data)
        st.dataframe(
            theme_df.style.format({"Value": "${:,.2f}", "Pct": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(theme_df.set_index("Theme")["Pct"])

st.divider()

# ── Recent orders ─────────────────────────────────────────────────────────────

st.subheader("Recent Orders")
if not orders:
    st.info("No recent orders.")
else:
    order_rows = []
    for o in orders[:20]:
        order_rows.append({
            "Time":   o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "",
            "Symbol": o.symbol,
            "Side":   o.side.value.upper(),
            "Type":   o.type.value,
            "Qty":    o.qty or o.notional,
            "Status": o.status.value,
        })
    st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────

st.divider()
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Data cached for 60 seconds. Click Refresh or reload the page to update.")
