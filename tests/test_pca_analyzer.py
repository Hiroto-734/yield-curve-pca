"""Tests for ``yield_curve_pca.analysis.pca_analyzer``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_curve_pca.analysis.pca_analyzer import YieldCurvePCA, fit_pca_from_yields


def test_pca_raises_before_fit() -> None:
    pca = YieldCurvePCA(n_components=3)
    with pytest.raises(RuntimeError, match="must be fit"):
        _ = pca.loadings


def test_pca_fit_transform_shapes(changes_bp: pd.DataFrame) -> None:
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    scores = pca.transform(changes_bp)
    assert scores.shape == (len(changes_bp), 3)
    assert list(scores.columns) == ["PC1", "PC2", "PC3"]
    assert pca.loadings.shape == (3, len(changes_bp.columns))


def test_pca_explained_variance_sums_correctly(changes_bp: pd.DataFrame) -> None:
    pca = YieldCurvePCA(n_components=None).fit(changes_bp)
    # All components together must account for ~100% of variance.
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-9
    # Ratios are monotonically non-increasing by construction.
    diffs = np.diff(pca.explained_variance_ratio_.values)
    assert (diffs <= 1e-12).all()


def test_pca_top3_explains_most_variance(changes_bp: pd.DataFrame) -> None:
    """The whole project rests on this: top 3 components > 90% of variance."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    assert pca.explained_variance_ratio_.sum() > 0.90


def test_pc1_is_a_level_factor(changes_bp: pd.DataFrame) -> None:
    """PC1 loadings should all share the same sign (the Level interpretation)."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    pc1 = pca.loadings.loc["PC1"]
    assert (pc1 > 0).all() or (pc1 < 0).all()


def test_pc2_is_a_slope_factor(changes_bp: pd.DataFrame) -> None:
    """PC2 should flip sign across the curve (Slope)."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    pc2 = pca.loadings.loc["PC2"]
    assert pc2.iloc[0] * pc2.iloc[-1] < 0


def test_inverse_transform_reconstructs(changes_bp: pd.DataFrame) -> None:
    pca = YieldCurvePCA(n_components=None).fit(changes_bp)
    scores = pca.transform(changes_bp)
    reconstructed = pca.inverse_transform(scores)
    # With all components, reconstruction should be near-perfect.
    np.testing.assert_allclose(
        reconstructed.values, changes_bp.values, atol=1e-6
    )


def test_save_load_roundtrip(changes_bp: pd.DataFrame, tmp_path: Path) -> None:
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    save_path = tmp_path / "pca.joblib"
    pca.save(save_path)

    restored = YieldCurvePCA.load(save_path)
    pd.testing.assert_frame_equal(pca.loadings, restored.loadings)
    pd.testing.assert_series_equal(
        pca.explained_variance_ratio_,
        restored.explained_variance_ratio_,
    )


def test_fit_pca_from_yields_matches_manual(yields_clean: pd.DataFrame) -> None:
    """Convenience function should produce the same fit as the manual chain."""
    from yield_curve_pca.data.preprocessor import to_changes_bp

    pca_auto, changes_auto = fit_pca_from_yields(yields_clean, n_components=3)
    pca_manual = YieldCurvePCA(n_components=3).fit(to_changes_bp(yields_clean))

    pd.testing.assert_frame_equal(pca_auto.loadings, pca_manual.loadings)
    pd.testing.assert_frame_equal(changes_auto, to_changes_bp(yields_clean))
