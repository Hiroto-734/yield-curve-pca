"""Tests for ``yield_curve_pca.data.preprocessor``."""

from __future__ import annotations

import pandas as pd

from yield_curve_pca.data.preprocessor import clean_yields, to_changes_bp


def test_clean_yields_removes_all_nan_rows(synthetic_yields: pd.DataFrame) -> None:
    cleaned = clean_yields(synthetic_yields)
    # Holiday row (2024-01-03) and partial-NaN row (2024-01-06) must be gone.
    assert cleaned.isna().sum().sum() == 0
    assert pd.Timestamp("2024-01-03") not in cleaned.index
    assert pd.Timestamp("2024-01-06") not in cleaned.index


def test_clean_yields_preserves_columns(synthetic_yields: pd.DataFrame) -> None:
    cleaned = clean_yields(synthetic_yields)
    assert list(cleaned.columns) == list(synthetic_yields.columns)


def test_to_changes_bp_unit_conversion(synthetic_yields: pd.DataFrame) -> None:
    cleaned = clean_yields(synthetic_yields)
    changes = to_changes_bp(cleaned)
    # 1.55 - 1.50 = 0.05% → 5 bp; preprocessor multiplies by 100.
    assert abs(changes.loc[pd.Timestamp("2024-01-04"), "3M"] - 5.0) < 1e-9


def test_to_changes_bp_drops_first_row(synthetic_yields: pd.DataFrame) -> None:
    cleaned = clean_yields(synthetic_yields)
    changes = to_changes_bp(cleaned)
    # ``diff().dropna()`` removes the first row.
    assert len(changes) == len(cleaned) - 1


def test_to_changes_bp_no_nan_in_output(synthetic_yields: pd.DataFrame) -> None:
    cleaned = clean_yields(synthetic_yields)
    changes = to_changes_bp(cleaned)
    assert changes.isna().sum().sum() == 0
