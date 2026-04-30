"""Treasury yield preprocessing.

Two operations matter:

1. **Cleaning**: drop rows that are entirely NaN (US holidays) and
   then drop any remaining row with even a single NaN, so PCA can
   operate on a fully-populated matrix.
2. **Differencing**: convert daily yield levels (in %) into daily
   *changes* in basis points, which is what we feed into PCA.
"""

from __future__ import annotations

import pandas as pd


def clean_yields(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing yields.

    The raw FRED data has two flavors of missing values:
      - whole-row NaN on US market holidays (no data published for any maturity);
      - sparse NaN where a maturity is unusually thin (e.g. 20Y after its 2020 reissue).

    PCA requires a complete matrix, so the simplest defensible policy is to
    drop both.

    Args:
        raw: DataFrame with one column per maturity.

    Returns:
        DataFrame with no NaN values.
    """
    return raw.dropna(how="all").dropna(how="any")


def to_changes_bp(yields: pd.DataFrame) -> pd.DataFrame:
    """Convert daily yield levels (in %) to first-difference changes in basis points.

    Yield curve PCA is conventionally applied to *changes* (which are
    closer to stationary) rather than levels (which trend). The output is
    in bp because that is the unit traders quote moves in:

        100 bp == 1.00 % == 0.01 (decimal).

    The first row is dropped (it has no prior day to difference against).

    Args:
        yields: DataFrame of yields in percent, indexed by date.

    Returns:
        DataFrame of daily changes in bp, with first row removed.
    """
    return yields.diff().dropna() * 100
