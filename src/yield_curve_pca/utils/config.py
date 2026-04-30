"""Project-wide constants and configuration.

Single source of truth for things like the default maturity set,
the FRED series identifiers, and any other magic values that
otherwise get duplicated across notebooks.
"""

from __future__ import annotations

# FRED series ID → human-readable maturity label.
# These are Daily Treasury Constant Maturity rates.
DEFAULT_MATURITIES: dict[str, str] = {
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}

# Maturity label → years (used as the x-axis when plotting curves).
MATURITY_YEARS: dict[str, float] = {
    "3M": 0.25,
    "6M": 0.5,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# FRED endpoint for daily-update CSV downloads (no auth required).
FRED_CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Default project date range for the analysis.
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-04-30"

# Annualization factor for daily P&L (US business days per year).
ANNUALIZATION_FACTOR = 252
