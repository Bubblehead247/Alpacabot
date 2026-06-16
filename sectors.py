"""
sectors.py — single source of truth for symbol → sector classification.

Used by:
  - screener.py  : tag each candidate with a sector + asset type, and persist
                   the map to sectors.csv (the "hybrid" draft step).
  - main.py      : enforce the per-sector position cap (config.MAX_PER_SECTOR).

Classification rules (first match wins):
  1. Sector / thematic ETFs  → their GICS sector (so they count alongside
     individual stocks in that sector, e.g. XLE + XOM both = Energy).
  2. Broad-market / international ETFs → one shared "Broad/Index" bucket.
  3. Individual stocks → sector from the persisted map (sectors.csv), which the
     screener drafts via yfinance. classify() can also look it up live.
  4. Anything unmapped → "Unknown".

Live code calls sector_of() — it reads only the static maps + sectors.csv and
never touches the network. classify() is the screener-side helper that may call
yfinance.
"""
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BROAD_BUCKET = "Broad/Index"
UNKNOWN      = "Unknown"

# Sector / thematic ETFs → GICS sector
SECTOR_ETF = {
    "XLE": "Energy",          "XOP": "Energy",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLV": "Health Care",     "XBI": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary", "XRT": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# yfinance uses its own sector vocabulary; map it onto the GICS names above so a
# stock and a sector ETF in the same sector share one bucket (e.g. AMGN
# "Healthcare" == XBI "Health Care").
NORMALIZE = {
    "Basic Materials":      "Materials",
    "Consumer Cyclical":    "Consumer Discretionary",
    "Consumer Defensive":   "Consumer Staples",
    "Financial Services":   "Financials",
    "Healthcare":           "Health Care",
    # already-canonical (pass through): Energy, Industrials, Technology,
    # Utilities, Real Estate, Communication Services.
}

# Broad-market / international / style ETFs → one diversified bucket
BROAD_ETF = {
    "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "IJH", "IJR", "VTI", "VOO",
    "EFA", "IEFA", "VGK", "VEA", "VWO", "EEM", "ACWI",
}

CSV_FILE = Path(__file__).resolve().parent / "sectors.csv"


def _load_persisted() -> dict[str, str]:
    m: dict[str, str] = {}
    if CSV_FILE.exists():
        with open(CSV_FILE, newline="") as f:
            for row in csv.DictReader(f):
                m[row["symbol"]] = row["sector"]
    return m


_PERSISTED = _load_persisted()


def sector_of(symbol: str) -> str:
    """Network-free sector lookup for live use (cap enforcement)."""
    if symbol in SECTOR_ETF:
        return SECTOR_ETF[symbol]
    if symbol in BROAD_ETF:
        return BROAD_BUCKET
    return _PERSISTED.get(symbol, UNKNOWN)


def classify(symbol: str) -> tuple[str, str]:
    """
    Return (sector, asset_type) for the screener. Uses the static maps first,
    then falls back to a yfinance lookup. Slow/flaky — call only for the handful
    of names being reported, never the whole survivor set.
    """
    if symbol in SECTOR_ETF:
        return SECTOR_ETF[symbol], "Sector ETF"
    if symbol in BROAD_ETF:
        return BROAD_BUCKET, "ETF"
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        if (info.get("quoteType") or "").upper() == "ETF":
            return BROAD_BUCKET, "ETF"
        sector = info.get("sector") or UNKNOWN
        return NORMALIZE.get(sector, sector), "Stock"
    except Exception as e:
        logger.warning(f"{symbol}: sector lookup failed: {e}")
        return UNKNOWN, "Unknown"


def persist(rows: list[dict]) -> None:
    """Write symbol → sector/type map to sectors.csv (committed, read live)."""
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "sector", "type"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"symbol": r["symbol"], "sector": r["sector"], "type": r["type"]})
    logger.info(f"Wrote {len(rows)} sector rows to {CSV_FILE.name}")
