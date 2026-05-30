"""
risk.py — Position sizing based on 1% account risk per trade.

Formula:
    shares = floor( (equity × RISK_PER_TRADE) / stop_distance )

Where stop_distance = entry_price - stop_price

This guarantees that if the hard stop is hit, the maximum loss is 1% of account equity.
Fractional cents are always rounded DOWN to ensure we never exceed the 1% threshold.
"""
import logging
import math

import config

logger = logging.getLogger(__name__)


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float = config.RISK_PER_TRADE,
) -> int:
    """
    Returns integer share count to buy.

    Args:
        account_equity: Current portfolio equity in dollars.
        entry_price:    Estimated or actual fill price per share.
        stop_price:     Hard stop price per share.
        risk_fraction:  Fraction of equity to risk (default: 0.01 = 1%).

    Returns:
        Number of whole shares to buy (0 if inputs are invalid).
    """
    stop_distance = entry_price - stop_price

    if stop_distance <= 0:
        logger.error(
            f"Invalid stop distance: {stop_distance:.4f} "
            f"(entry={entry_price}, stop={stop_price}). Returning 0 shares."
        )
        return 0

    dollar_risk = account_equity * risk_fraction
    raw_shares  = dollar_risk / stop_distance
    shares      = math.floor(raw_shares)   # Always floor — never risk more than 1%

    if shares <= 0:
        logger.warning(
            f"Position size calculated as 0. "
            f"Dollar risk=${dollar_risk:.2f}, Stop distance=${stop_distance:.2f}"
        )
        return 0

    position_cost   = shares * entry_price
    actual_risk_usd = shares * stop_distance
    actual_risk_pct = (actual_risk_usd / account_equity) * 100

    logger.info(
        f"Position size | Equity=${account_equity:,.2f} | Entry=${entry_price:.2f} | "
        f"Stop=${stop_price:.2f} | StopDist=${stop_distance:.2f} | "
        f"Shares={shares} | Cost=${position_cost:,.2f} | "
        f"Risk=${actual_risk_usd:.2f} ({actual_risk_pct:.3f}%)"
    )

    return shares
