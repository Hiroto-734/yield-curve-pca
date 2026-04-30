"""Tests for ``yield_curve_pca.analysis.backtest``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yield_curve_pca.analysis.backtest import (
    backtest,
    classify_regime,
    compute_metrics,
    make_signal,
    regime_to_filter,
)


def test_compute_metrics_zero_pnl() -> None:
    pnl = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
    pos = pd.Series([0, 0, 0, 0, 0])
    m = compute_metrics(pnl, pos)
    assert m["total_pnl_bp"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["hit_rate"] == 0.0


def test_compute_metrics_constant_positive_pnl() -> None:
    """Every active day wins the same amount → hit rate == 1.0, infinite-ish Sharpe."""
    pnl = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    pos = pd.Series([1, 1, 1, 1, 1])
    m = compute_metrics(pnl, pos)
    assert m["total_pnl_bp"] == 5.0
    assert m["hit_rate"] == 1.0
    assert m["max_dd_bp"] == 0.0
    assert m["sharpe"] == 0.0  # std = 0 → we report Sharpe as 0 to avoid inf.


def test_compute_metrics_drawdown_detected() -> None:
    pnl = pd.Series([10.0, -5.0, -3.0, 2.0])  # peak at 10, trough at 2
    pos = pd.Series([1, 1, 1, 1])
    m = compute_metrics(pnl, pos)
    assert m["max_dd_bp"] == -8.0  # 10 - 2


def test_make_signal_no_lookahead(synthetic_pc_score: pd.Series) -> None:
    """Signal at index t must not depend on pc_score values from index ≥ t.

    We verify by perturbing pc_score[k:] for some k and checking
    the signal at index < k stays identical.
    """
    base = make_signal(synthetic_pc_score, window=20, enter_threshold=1.0)

    perturbed = synthetic_pc_score.copy()
    perturbed.iloc[50:] = 999.0  # smash the future
    perturbed_signal = make_signal(perturbed, window=20, enter_threshold=1.0)

    # The first 50 positions must match exactly. Use position[t-1] as the
    # decision input for position[t], so up to and including index 49 nothing
    # has yet seen the perturbed values.
    pd.testing.assert_series_equal(base.iloc[:50], perturbed_signal.iloc[:50])


def test_make_signal_respects_threshold(synthetic_pc_score: pd.Series) -> None:
    """Higher thresholds should produce fewer (or equal) trades."""
    sig_low = make_signal(synthetic_pc_score, window=20, enter_threshold=0.5)
    sig_high = make_signal(synthetic_pc_score, window=20, enter_threshold=2.5)
    assert (sig_high != 0).sum() <= (sig_low != 0).sum()


def test_make_signal_max_hold_days_caps_position(synthetic_pc_score: pd.Series) -> None:
    """No position should be held strictly longer than ``max_hold_days``."""
    max_hold = 5
    sig = make_signal(
        synthetic_pc_score,
        window=10,
        enter_threshold=0.5,
        max_hold_days=max_hold,
    )

    # Run-length encode the position.
    in_position = sig != 0
    runs = (in_position != in_position.shift()).cumsum()[in_position]
    if len(runs) > 0:
        assert runs.value_counts().max() <= max_hold


def test_backtest_returns_aligned_series(synthetic_pc_score: pd.Series) -> None:
    result = backtest(
        "test",
        synthetic_pc_score,
        window=20,
        enter_threshold=1.0,
    )
    assert len(result.daily_pnl) == len(synthetic_pc_score)
    assert len(result.cum_pnl) == len(synthetic_pc_score)
    assert len(result.position) == len(synthetic_pc_score)
    assert result.name == "test"


def test_classify_regime_threshold_behavior(synthetic_pc_score: pd.Series) -> None:
    """A very high threshold → everything is 'ranging'; very low → 'trending'."""
    high_thr = classify_regime(synthetic_pc_score, window=10, threshold=1e6)
    assert (high_thr == "ranging").all()

    low_thr = classify_regime(synthetic_pc_score, window=10, threshold=0)
    # After warmup, every day should be classified as trending (any nonzero sum > 0).
    after_warmup = low_thr.iloc[10:]
    assert (after_warmup == "trending").all()


def test_regime_to_filter_lags(synthetic_pc_score: pd.Series) -> None:
    """The OK/NG filter must be lagged by one day from the regime label."""
    regime = classify_regime(synthetic_pc_score, window=10, threshold=0)
    filt = regime_to_filter(regime)
    assert filt.iloc[0] != filt.iloc[0]  # NaN at the very first row (after shift)
    # After the shift, today's filter reflects yesterday's regime.
    expected = regime.shift(1).map({"trending": "NG", "ranging": "OK"})
    pd.testing.assert_series_equal(filt, expected, check_names=False)


def test_regime_filter_blocks_entry(synthetic_pc_score: pd.Series) -> None:
    """A constant-NG filter must produce zero trades."""
    ng_filter = pd.Series("NG", index=synthetic_pc_score.index)
    sig = make_signal(
        synthetic_pc_score,
        window=10,
        enter_threshold=0.5,
        regime_filter=ng_filter,
    )
    assert (sig != 0).sum() == 0
