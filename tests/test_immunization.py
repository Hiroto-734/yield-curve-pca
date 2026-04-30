"""Tests for ``yield_curve_pca.analysis.immunization``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_curve_pca.analysis.immunization import (
    Bond,
    Portfolio,
    daily_pnl_direct,
    daily_pnl_via_pcs,
    hedged_portfolio,
    pc_exposures,
    solve_hedge,
    walk_forward_hedge,
)
from yield_curve_pca.analysis.pca_analyzer import YieldCurvePCA
from yield_curve_pca.utils.config import MATURITY_YEARS

# --------------------------------------------------------------------- #
# Bond                                                                  #
# --------------------------------------------------------------------- #


def test_bond_rejects_unknown_maturity() -> None:
    with pytest.raises(ValueError, match="Unknown maturity label"):
        Bond(maturity_label="99Y", notional=1_000_000, yield_pct=4.0)


def test_bond_modified_duration_zero_coupon_formula() -> None:
    """``D_mod = T / (1 + y)`` for the zero-coupon approximation."""
    b = Bond(maturity_label="10Y", notional=1_000_000, yield_pct=4.0)
    expected = 10.0 / 1.04
    assert abs(b.modified_duration - expected) < 1e-9


def test_bond_dv01_long_position_is_positive() -> None:
    """Long positions have positive DV01 (they lose money when yields rise)."""
    b = Bond(maturity_label="10Y", notional=1_000_000, yield_pct=4.0)
    # DV01 = $1M × (10/1.04) × 0.0001 ≈ $961.54
    assert b.dv01 > 0
    assert abs(b.dv01 - 1_000_000 * (10 / 1.04) * 0.0001) < 1e-6


def test_bond_dv01_short_position_is_negative() -> None:
    """Short positions get a negative DV01, so +1bp produces a gain."""
    b = Bond(maturity_label="10Y", notional=-1_000_000, yield_pct=4.0)
    assert b.dv01 < 0


def test_bond_dv01_scales_with_maturity() -> None:
    """For the same notional, longer maturity → larger DV01."""
    short = Bond(maturity_label="2Y", notional=1_000_000, yield_pct=4.0)
    long = Bond(maturity_label="30Y", notional=1_000_000, yield_pct=4.0)
    assert long.dv01 > short.dv01


# --------------------------------------------------------------------- #
# Portfolio                                                             #
# --------------------------------------------------------------------- #


def test_portfolio_from_holdings_pulls_yields() -> None:
    yields = pd.Series({m: 4.0 for m in MATURITY_YEARS})
    p = Portfolio.from_holdings({"10Y": 100_000_000}, yields)
    assert len(p.bonds) == 1
    assert p.bonds[0].maturity_label == "10Y"
    assert p.bonds[0].notional == 100_000_000
    assert p.bonds[0].yield_pct == 4.0


def test_portfolio_total_dv01_aggregates_correctly() -> None:
    yields = pd.Series({m: 4.0 for m in MATURITY_YEARS})
    p = Portfolio.from_holdings(
        {"2Y": 50_000_000, "10Y": 50_000_000}, yields
    )
    expected = sum(b.dv01 for b in p.bonds)
    assert abs(p.total_dv01 - expected) < 1e-6


def test_portfolio_dv01_by_maturity_index_matches_config() -> None:
    yields = pd.Series({m: 4.0 for m in MATURITY_YEARS})
    p = Portfolio.from_holdings({"10Y": 100_000_000}, yields)
    s = p.dv01_by_maturity
    assert list(s.index) == list(MATURITY_YEARS)
    assert s["10Y"] > 0
    # Maturities not held should be zero (not NaN).
    assert s.drop("10Y").sum() == 0


def test_portfolio_dv01_by_maturity_aggregates_same_maturity() -> None:
    """Two long positions at the same maturity should sum their DV01s."""
    p = Portfolio(
        bonds=[
            Bond("10Y", 50_000_000, 4.0),
            Bond("10Y", 50_000_000, 4.0),
        ]
    )
    single = Bond("10Y", 100_000_000, 4.0)
    assert abs(p.dv01_by_maturity["10Y"] - single.dv01) < 1e-6


# --------------------------------------------------------------------- #
# PC exposures                                                          #
# --------------------------------------------------------------------- #


def test_pc_exposures_long_portfolio_has_positive_pc1_exposure(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """A long bond portfolio should have positive Level (PC1) exposure.

    PC1 loadings are uniformly positive (Level interpretation), so a
    portfolio that is long DV01 across the curve has a positive dot
    product with PC1.
    """
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    p = Portfolio.from_holdings({"10Y": 100_000_000}, yields_clean.iloc[-1])

    exp = pc_exposures(p, pca)
    assert exp["PC1"] > 0


def test_pc_exposures_indexed_by_pc_names(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    p = Portfolio.from_holdings({"10Y": 100_000_000}, yields_clean.iloc[-1])

    exp = pc_exposures(p, pca)
    assert list(exp.index) == ["PC1", "PC2", "PC3"]


# --------------------------------------------------------------------- #
# P&L: direct vs via-PCs                                                #
# --------------------------------------------------------------------- #


def test_pnl_via_all_pcs_matches_direct_to_numerical_precision(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """When we keep all PCs, the via-PCs P&L should equal the direct P&L."""
    pca = YieldCurvePCA(n_components=None).fit(changes_bp)  # all components
    scores = pca.transform(changes_bp)
    p = Portfolio.from_holdings(
        {"10Y": 100_000_000}, yields_clean.iloc[-1]
    )

    pnl_direct = daily_pnl_direct(p, changes_bp)
    pnl_via_pcs = daily_pnl_via_pcs(p, scores, pca, n_components=None)

    np.testing.assert_allclose(
        pnl_direct.values, pnl_via_pcs.values, atol=1e-6
    )


def test_pnl_via_top3_pcs_explains_most_variance(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """Top 3 PCs should reconstruct the daily P&L to within ~10% RMSE.

    The PCA model explains 96% of yield-change variance with 3 PCs, so
    the residual P&L variance should be small relative to the total.
    """
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    scores = pca.transform(changes_bp)
    p = Portfolio.from_holdings(
        {"10Y": 100_000_000}, yields_clean.iloc[-1]
    )

    pnl_direct = daily_pnl_direct(p, changes_bp)
    pnl_via_pcs = daily_pnl_via_pcs(p, scores, pca, n_components=3)

    residual = pnl_direct - pnl_via_pcs
    explained_ratio = 1 - residual.var() / pnl_direct.var()
    # Top-3 PCs should explain at least 90% of the P&L variance for a
    # vanilla long-bond portfolio.
    assert explained_ratio > 0.90


def test_short_portfolio_pnl_signs_flip() -> None:
    """A short portfolio should produce P&L with the opposite sign of a long one."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    changes = pd.DataFrame(
        {"10Y": [5.0, -3.0, 2.0, 0.0, -1.0]}, index=dates
    )

    long_p = Portfolio(bonds=[Bond("10Y", 1_000_000, 4.0)])
    short_p = Portfolio(bonds=[Bond("10Y", -1_000_000, 4.0)])

    pnl_long = daily_pnl_direct(long_p, changes)
    pnl_short = daily_pnl_direct(short_p, changes)

    np.testing.assert_allclose(pnl_long.values, -pnl_short.values, atol=1e-9)


# --------------------------------------------------------------------- #
# Hedge construction                                                    #
# --------------------------------------------------------------------- #


def test_solve_hedge_dimension_mismatch_raises(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """1 hedge instrument vs 2 PCs to hedge → ValueError."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    p = Portfolio.from_holdings({"30Y": 100_000_000}, yields_clean.iloc[-1])
    with pytest.raises(ValueError, match="must match"):
        solve_hedge(
            p, pca, ["10Y"], yields_clean.iloc[-1], ["PC1", "PC2"]
        )


def test_solve_hedge_neutralizes_targeted_pc_exposures(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """After hedging PC1+PC2+PC3, the combined exposures should all be ≈ 0."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    yields = yields_clean.iloc[-1]
    p = Portfolio.from_holdings({"30Y": 100_000_000}, yields)

    hedge = solve_hedge(
        p, pca, ["2Y", "10Y", "30Y"], yields, ["PC1", "PC2", "PC3"]
    )
    combined = hedged_portfolio(p, hedge, yields)
    new_exp = pc_exposures(combined, pca)

    # Every targeted exposure should be zero to machine precision.
    np.testing.assert_allclose(new_exp.values, 0.0, atol=1e-6)


def test_solve_hedge_long_portfolio_gets_short_30y_in_full_hedge(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """A long-30Y portfolio hedged with 30Y as one instrument should see
    its 30Y hedge come out negative (i.e. short 30Y)."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    yields = yields_clean.iloc[-1]
    p = Portfolio.from_holdings({"30Y": 100_000_000}, yields)

    hedge = solve_hedge(
        p, pca, ["2Y", "10Y", "30Y"], yields, ["PC1", "PC2", "PC3"]
    )
    assert hedge["30Y"] < 0


def test_solve_hedge_partial_neutralization_leaves_other_exposures(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """Hedging only PC1 should leave PC2/PC3 exposures non-zero."""
    pca = YieldCurvePCA(n_components=3).fit(changes_bp)
    yields = yields_clean.iloc[-1]
    p = Portfolio.from_holdings({"30Y": 100_000_000}, yields)

    hedge = solve_hedge(p, pca, ["10Y"], yields, ["PC1"])
    combined = hedged_portfolio(p, hedge, yields)
    new_exp = pc_exposures(combined, pca)

    # PC1 ≈ 0 by construction
    assert abs(new_exp["PC1"]) < 1e-6
    # PC2 and PC3 should remain non-trivial
    assert abs(new_exp["PC2"]) > 1.0
    assert abs(new_exp["PC3"]) > 1.0


def test_hedged_portfolio_combines_bonds(
    yields_clean: pd.DataFrame,
) -> None:
    """``hedged_portfolio`` should produce a Portfolio with original + hedge bonds."""
    yields = yields_clean.iloc[-1]
    p = Portfolio.from_holdings({"30Y": 100_000_000}, yields)
    hedge = {"10Y": -50_000_000}

    combined = hedged_portfolio(p, hedge, yields)
    assert len(combined.bonds) == 2
    # Original bond is still in there
    assert any(
        b.maturity_label == "30Y" and b.notional == 100_000_000
        for b in combined.bonds
    )
    # Hedge bond is added with the right notional
    assert any(
        b.maturity_label == "10Y" and b.notional == -50_000_000
        for b in combined.bonds
    )


# --------------------------------------------------------------------- #
# Walk-forward                                                          #
# --------------------------------------------------------------------- #


def test_walk_forward_returns_expected_keys(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """The walk-forward function returns a dict with the documented keys."""
    out = walk_forward_hedge(
        portfolio_holdings={"30Y": 100_000_000},
        yields=yields_clean,
        changes_bp=changes_bp,
        hedge_maturities=["3M", "5Y", "20Y"],
        pcs_to_hedge=["PC1", "PC2", "PC3"],
        window_days=252,
        rebalance_freq="ME",
    )
    assert set(out) == {"pnl", "hedge_history", "rebalance_dates"}
    assert isinstance(out["pnl"], pd.Series)
    assert isinstance(out["hedge_history"], pd.DataFrame)


def test_walk_forward_no_lookahead(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """Smashing future yield changes must not affect hedges constructed earlier.

    We perturb ``changes_bp`` past a midpoint date and confirm the
    walk-forward hedge ratios at all rebalance dates *before* the
    midpoint are unchanged. This is the canonical look-ahead-bias check.
    """
    common_kwargs = dict(
        portfolio_holdings={"30Y": 100_000_000},
        yields=yields_clean,
        hedge_maturities=["3M", "5Y", "20Y"],
        pcs_to_hedge=["PC1", "PC2", "PC3"],
        window_days=252,
        rebalance_freq="ME",
    )
    base = walk_forward_hedge(changes_bp=changes_bp, **common_kwargs)

    midpoint = changes_bp.index[len(changes_bp) // 2]
    perturbed = changes_bp.copy()
    # Smash future with a different (but still well-conditioned) signal so
    # the post-midpoint PCA fit remains numerically stable.
    rng = np.random.default_rng(42)
    n_post = (perturbed.index > midpoint).sum()
    perturbed.loc[perturbed.index > midpoint, :] = (
        rng.standard_normal((n_post, perturbed.shape[1])) * 100
    )

    perturbed_out = walk_forward_hedge(changes_bp=perturbed, **common_kwargs)

    # Hedges at every rebalance date < midpoint should be identical.
    pre_midpoint = base["hedge_history"].index < midpoint
    pd.testing.assert_frame_equal(
        base["hedge_history"].loc[pre_midpoint],
        perturbed_out["hedge_history"].loc[pre_midpoint],
        check_exact=False,
    )


def test_walk_forward_skips_warmup(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """No P&L should be reported during the warm-up period."""
    out = walk_forward_hedge(
        portfolio_holdings={"30Y": 100_000_000},
        yields=yields_clean,
        changes_bp=changes_bp,
        hedge_maturities=["3M", "5Y", "20Y"],
        pcs_to_hedge=["PC1", "PC2", "PC3"],
        window_days=252,
        rebalance_freq="ME",
    )
    # First P&L observation must come at least window_days into the sample.
    first_pnl_loc = changes_bp.index.get_loc(out["pnl"].index[0])
    assert first_pnl_loc >= 252


def test_walk_forward_monthly_has_more_rebalances_than_quarterly(
    yields_clean: pd.DataFrame, changes_bp: pd.DataFrame
) -> None:
    """Higher rebalance frequency = more rebalance dates in the same period."""
    common = dict(
        portfolio_holdings={"30Y": 100_000_000},
        yields=yields_clean,
        changes_bp=changes_bp,
        hedge_maturities=["3M", "5Y", "20Y"],
        pcs_to_hedge=["PC1", "PC2", "PC3"],
        window_days=252,
    )
    monthly = walk_forward_hedge(rebalance_freq="ME", **common)
    quarterly = walk_forward_hedge(rebalance_freq="QE", **common)
    assert len(monthly["rebalance_dates"]) > len(quarterly["rebalance_dates"])
