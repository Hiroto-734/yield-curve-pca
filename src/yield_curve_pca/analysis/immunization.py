"""PCA-based portfolio immunization.

Bonds and portfolios as the smallest data structures we need, plus the
two pieces of math the rest of the analysis is built on:

1. **DV01 per bond** — the dollar P&L for a 1 bp parallel rise in the
   bond's yield. We use the modified-duration approximation
   ``DV01 ≈ notional × T / (1 + y) × 0.0001`` (zero-coupon-style),
   which is good to single-bp accuracy for the maturities and yield
   levels in this project (3M-30Y, ~0-5%).

2. **PC factor exposure** — the dollar P&L of the portfolio for a
   one-unit move along each principal component. Computed directly
   from the PCA loadings and the per-maturity DV01 vector.

The hedge construction itself (solving a linear system to neutralize
selected exposures) lives in ``solve_hedge`` and ``apply_hedge``;
those round out Phase 2 of the immunization notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..utils.config import MATURITY_YEARS
from .pca_analyzer import YieldCurvePCA

# --------------------------------------------------------------------- #
# Bond and Portfolio                                                    #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Bond:
    """A simplified zero-coupon-style bond.

    We don't model coupon cash flows; for the project's purposes
    (PCA-based factor analysis) the modified-duration approximation
    around the bond's current yield is enough. All quantities are
    dollar-denominated.

    Attributes:
        maturity_label: One of the keys in ``MATURITY_YEARS`` (e.g. "10Y").
        notional: Face value in dollars (positive = long, negative = short).
        yield_pct: Current yield in percent (e.g. 4.0 for 4%).
    """

    maturity_label: str
    notional: float
    yield_pct: float

    def __post_init__(self) -> None:
        if self.maturity_label not in MATURITY_YEARS:
            raise ValueError(
                f"Unknown maturity label '{self.maturity_label}'. "
                f"Must be one of {list(MATURITY_YEARS)}."
            )

    @property
    def maturity_years(self) -> float:
        return MATURITY_YEARS[self.maturity_label]

    @property
    def modified_duration(self) -> float:
        """``D_mod = T / (1 + y)`` for a zero-coupon bond."""
        return self.maturity_years / (1.0 + self.yield_pct / 100.0)

    @property
    def dv01(self) -> float:
        """Dollar P&L for a +1 bp parallel rise in this bond's yield.

        Sign convention: ``DV01 > 0`` for long positions (which lose money
        when yields rise). For a short position, pass a negative ``notional``;
        ``dv01`` will then be negative, and a +1 bp move produces a gain
        (``-DV01 × Δy_bp > 0``).
        """
        return self.notional * self.modified_duration * 0.0001


@dataclass
class Portfolio:
    """A bag of bonds.

    Bonds at the same maturity are aggregated when reporting DV01 by
    maturity, but kept separate in the underlying ``bonds`` list so a
    portfolio that contains both a long and a short of the same maturity
    is still well-defined.
    """

    bonds: list[Bond] = field(default_factory=list)

    # ----- constructors -------------------------------------------------- #

    @classmethod
    def from_holdings(
        cls,
        holdings: dict[str, float],
        yields: pd.Series,
    ) -> Portfolio:
        """Build a portfolio from ``{maturity_label: notional}`` and a yield curve.

        Args:
            holdings: e.g. ``{"30Y": 100_000_000}``.
            yields: Series indexed by maturity label, in percent.
                Typically ``yields_clean.iloc[-1]`` (the latest snapshot).

        Returns:
            ``Portfolio`` with one ``Bond`` per holding entry.
        """
        return cls(
            bonds=[
                Bond(maturity_label=m, notional=n, yield_pct=float(yields[m]))
                for m, n in holdings.items()
            ]
        )

    # ----- aggregate properties ----------------------------------------- #

    @property
    def total_notional(self) -> float:
        return sum(b.notional for b in self.bonds)

    @property
    def total_dv01(self) -> float:
        return sum(b.dv01 for b in self.bonds)

    @property
    def dv01_by_maturity(self) -> pd.Series:
        """Per-maturity DV01, summed over bonds at the same maturity.

        Returns a Series indexed by ``MATURITY_YEARS.keys()`` order, with
        zero entries for maturities not held.
        """
        agg: dict[str, float] = {m: 0.0 for m in MATURITY_YEARS}
        for b in self.bonds:
            agg[b.maturity_label] += b.dv01
        return pd.Series(agg, name="dv01_by_maturity")

    def __repr__(self) -> str:
        n = len(self.bonds)
        nominal = self.total_notional / 1e6
        dv01 = self.total_dv01
        return f"Portfolio({n} bonds, ${nominal:.1f}M notional, DV01=${dv01:,.0f})"


# --------------------------------------------------------------------- #
# Factor exposures                                                      #
# --------------------------------------------------------------------- #


def pc_exposures(portfolio: Portfolio, pca: YieldCurvePCA) -> pd.Series:
    """Portfolio's exposure to each principal component.

    For a portfolio with per-maturity DV01 vector ``d`` and PCA loadings
    matrix ``L`` (rows = PCs, cols = maturities), the dollar P&L from a
    move ``s_k`` along ``PC_k`` is approximately

        P&L_k  ≈  - s_k × (L[k] · d)

    so we call ``L[k] · d`` the *exposure* of the portfolio to ``PC_k``.
    The minus sign is the same convention as DV01 — a long bond portfolio
    has a positive exposure to a positive PC1 move (parallel rise) and
    therefore loses money when that move happens.

    Args:
        portfolio: A ``Portfolio``.
        pca: A fitted ``YieldCurvePCA``.

    Returns:
        Series of dollar exposures, indexed by PC name (``PC1``, ``PC2``, …).
    """
    dv01 = portfolio.dv01_by_maturity.reindex(pca.loadings.columns).fillna(0.0)
    exposures = pca.loadings.values @ dv01.values
    return pd.Series(
        exposures,
        index=pca.loadings.index,
        name="pc_exposure_dollars_per_unit_pc",
    )


# --------------------------------------------------------------------- #
# Daily P&L (two equivalent calculations)                               #
# --------------------------------------------------------------------- #


def daily_pnl_direct(
    portfolio: Portfolio,
    changes_bp: pd.DataFrame,
) -> pd.Series:
    """Daily P&L computed directly from per-maturity yield changes.

    ``P&L_t = -Σᵢ DV01ᵢ × Δyᵢ_t``  (with Δy in bp, DV01 in $/bp)

    This is the ground-truth reference. Compare against
    ``daily_pnl_via_pcs`` to see how much of the variance the kept PCs
    explain.

    Args:
        portfolio: A ``Portfolio``.
        changes_bp: Daily yield changes in bp, one column per maturity.

    Returns:
        Series of daily P&L in dollars, indexed by date.
    """
    dv01 = portfolio.dv01_by_maturity.reindex(changes_bp.columns).fillna(0.0)
    pnl = -(changes_bp.values @ dv01.values)
    return pd.Series(pnl, index=changes_bp.index, name="pnl_direct_dollars")


def daily_pnl_via_pcs(
    portfolio: Portfolio,
    pc_scores: pd.DataFrame,
    pca: YieldCurvePCA,
    n_components: int | None = None,
) -> pd.Series:
    """Daily P&L approximated by projecting onto the top ``n_components`` PCs.

    ``P&L_t  ≈  -Σ_k score_k_t × exposure_k  -  DV01 · μ``

    where ``μ`` is the per-maturity mean that PCA centered out before
    projecting. The mean term is constant per day (it's the drift
    component) but matters for matching ``daily_pnl_direct`` exactly:

    * ``n_components=None`` (use every PC) reproduces the direct P&L to
      numerical precision.
    * ``n_components=3`` matches the direct P&L up to the variance not
      captured by the dropped PCs (~4% in this project).

    Args:
        portfolio: A ``Portfolio``.
        pc_scores: DataFrame of PC scores, one column per PC.
        pca: The fitted ``YieldCurvePCA`` whose loadings produced ``pc_scores``.
        n_components: How many leading PCs to use; ``None`` = all of them.

    Returns:
        Series of daily P&L in dollars, indexed by ``pc_scores.index``.
    """
    exposures = pc_exposures(portfolio, pca)

    pcs = pc_scores.columns if n_components is None else pc_scores.columns[:n_components]
    scores_subset = pc_scores[pcs]
    exposures_subset = exposures.reindex(pcs)

    # Score-driven term (varies by day).
    pnl = -(scores_subset.values @ exposures_subset.values)

    # Constant mean correction. PCA centered the inputs before projecting,
    # so the score-driven term reconstructs (changes_bp - mean), not
    # changes_bp. We add back -DV01 · mean to restore the missing constant.
    dv01 = portfolio.dv01_by_maturity.reindex(pca.loadings.columns).fillna(0.0)
    mean_correction = -float(dv01.values @ pca.mean_.values)
    pnl = pnl + mean_correction

    return pd.Series(pnl, index=pc_scores.index, name="pnl_via_pcs_dollars")
