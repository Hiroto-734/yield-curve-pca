"""FRED data fetching.

Uses the public ``fredgraph.csv`` endpoint, which requires no API key
and returns a clean two-column CSV (date + value) per series.

We deliberately avoid ``pandas-datareader`` because as of pandas 2.3
it raises a ``deprecate_kwarg`` signature error at import time and the
project has been unmaintained for years.
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import DEFAULT_MATURITIES, FRED_CSV_ENDPOINT


def _build_fred_url(series_id: str, start: str | None, end: str | None) -> str:
    parts = [f"id={series_id}"]
    if start is not None:
        parts.append(f"cosd={start}")
    if end is not None:
        parts.append(f"coed={end}")
    return f"{FRED_CSV_ENDPOINT}?{'&'.join(parts)}"


def fetch_fred_series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Fetch a single FRED series as a date-indexed Series.

    Missing values (US holidays etc.) are returned as NaN.

    Args:
        series_id: FRED series ID (e.g. ``"DGS10"``, ``"CPIAUCSL"``).
        start: ``YYYY-MM-DD``. If ``None``, fetches full history.
        end: ``YYYY-MM-DD``. If ``None``, fetches up to the most recent point.

    Returns:
        ``pd.Series`` indexed by ``observation_date`` (named after ``series_id``).
    """
    url = _build_fred_url(series_id, start, end)
    df = pd.read_csv(url, parse_dates=["observation_date"])
    return df.set_index("observation_date")[series_id]


def fetch_treasury_yields(
    start: str = "2020-01-01",
    end: str | None = None,
    maturities: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch a panel of Constant Maturity Treasury yields from FRED.

    Each maturity is fetched in its own request because FRED's multi-series
    CSV endpoint applies ``cosd``/``coed`` only to the first series, leaving
    every other column with its full history. Calling once per series
    keeps the resulting DataFrame aligned and predictable.

    Args:
        start: First date to include (``YYYY-MM-DD``).
        end: Last date to include (``YYYY-MM-DD``). If ``None``, latest.
        maturities: Mapping of FRED series ID → display label. Defaults to
            ``DEFAULT_MATURITIES`` from ``utils.config``.

    Returns:
        DataFrame with one column per maturity (named by display label),
        indexed by date. Holidays appear as rows full of NaN.
    """
    if maturities is None:
        maturities = DEFAULT_MATURITIES

    series_dict: dict[str, pd.Series] = {}
    for fred_id, label in maturities.items():
        series_dict[label] = fetch_fred_series(fred_id, start, end)

    df = pd.concat(series_dict, axis=1)
    df.columns.name = "maturity"
    df.index.name = "date"
    return df


def fetch_macro_data(
    series_ids: list[str],
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Fetch multiple monthly macro series from FRED into one DataFrame.

    Convenient wrapper around ``fetch_fred_series`` for things like
    CPI (``CPIAUCSL``), Core CPI (``CPILFESL``), or NFP (``PAYEMS``).
    Series are joined on ``observation_date``; mismatched calendars
    surface as NaN rows.

    Args:
        series_ids: List of FRED series IDs.
        start: First date to include.

    Returns:
        DataFrame with one column per series, indexed by observation date.
    """
    series_dict = {sid: fetch_fred_series(sid, start) for sid in series_ids}
    return pd.concat(series_dict, axis=1)
