"""
notifier.py — ntfy.sh push alert integration for the Mean Reversion Bot.

Two alert tiers:
  ⚠️  WARNING  RSI(2) ≤ 20 — approaching entry zone, heads-up only
  🚨  SIGNAL   RSI(2) ≤ 10 — entry signal fired, trade queued for next open

Alerts are fire-and-forget. If ntfy.sh is unreachable the bot continues normally.
"""
import logging

import requests

import config

logger = logging.getLogger(__name__)


def send_warning(symbol: str, rsi_value: float):
    """Fire a push alert when RSI(2) enters the 10–20 warming zone."""
    try:
        requests.post(
            "https://ntfy.sh",
            json={
                "topic":    config.NTFY_TOPIC,
                "title":    f"⚠️ WARMING UP: {symbol}",
                "message":  (
                    f"{symbol} RSI(2) = {rsi_value:.1f} — approaching entry zone. "
                    f"Watch for ≤10."
                ),
                "priority": 3,
                "tags":     ["chart_with_upwards_trend"],
            },
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"ntfy warning alert failed for {symbol}: {e}")


def send_checkpoint(new_trades: int, target: int):
    """One-time push when the paper-trading evaluation checkpoint is reached."""
    try:
        requests.post(
            "https://ntfy.sh",
            json={
                "topic":    config.NTFY_TOPIC,
                "title":    "\U0001f4ca EVAL CHECKPOINT",
                "message":  (
                    f"{new_trades} closed trades since v1.4 go-live (target {target}). "
                    f"Review execution fidelity: fills/slippage, stops, signal match, "
                    f"win~68%/hold~3-4d, drawdown <25%. Run: python eval_checkpoint.py"
                ),
                "priority": 4,
                "tags":     ["bar_chart"],
            },
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"ntfy checkpoint alert failed: {e}")


def send_signal(symbol: str, rsi_value: float):
    """Fire a high-priority push alert when a live entry signal fires."""
    try:
        requests.post(
            "https://ntfy.sh",
            json={
                "topic":    config.NTFY_TOPIC,
                "title":    f"\U0001f6a8 SIGNAL: {symbol}",
                "message":  (
                    f"{symbol} RSI(2) = {rsi_value:.1f} — ENTRY SIGNAL FIRED. "
                    f"Bot has queued a trade."
                ),
                "priority": 5,
                "tags":     ["rotating_light"],
            },
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"ntfy signal alert failed for {symbol}: {e}")
