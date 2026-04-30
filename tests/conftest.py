"""Shared pytest fixtures.

We deliberately don't hit the FRED network in tests — instead we
rely on the cached parquet artifacts the notebooks have already
produced and committed. If those aren't present (e.g. the user
cloned without running the notebooks), the affected tests are
skipped rather than failed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def yields_clean(project_root: Path) -> pd.DataFrame:
    """Cleaned UST yields produced by Notebook 01."""
    path = project_root / "data" / "processed" / "ust_yields_clean.parquet"
    if not path.exists():
        pytest.skip(f"Cached fixture missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def changes_bp(project_root: Path) -> pd.DataFrame:
    """Daily yield changes in bp produced by Notebook 02."""
    path = project_root / "data" / "processed" / "ust_yields_changes_bp.parquet"
    if not path.exists():
        pytest.skip(f"Cached fixture missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def pc_scores(project_root: Path) -> pd.DataFrame:
    """PC scores produced by Notebook 03."""
    path = project_root / "data" / "processed" / "pca_scores.parquet"
    if not path.exists():
        pytest.skip(f"Cached fixture missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture
def synthetic_yields() -> pd.DataFrame:
    """Small synthetic yield panel for unit-testing preprocessor logic.

    Includes one all-NaN row (a "holiday") and one partially-missing row.
    """
    return pd.DataFrame(
        {
            "3M":  [1.50, np.nan, 1.55, 1.60, 1.58],
            "2Y":  [1.40, np.nan, 1.42, 1.45, np.nan],
            "10Y": [2.00, np.nan, 2.02, 2.05, 2.08],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="D", name="date"),
    )


@pytest.fixture
def synthetic_pc_score() -> pd.Series:
    """100-day Gaussian noise around zero, deterministic via fixed seed.

    Used by backtest tests so results are reproducible.
    """
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.standard_normal(100) * 5.0,
        index=pd.date_range("2024-01-01", periods=100, freq="B"),
        name="PC2",
    )
