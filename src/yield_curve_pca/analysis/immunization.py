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

import numpy as np
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
# Hedge construction                                                    #
# --------------------------------------------------------------------- #


def solve_hedge(
    portfolio: Portfolio,
    pca: YieldCurvePCA,
    hedge_maturities: list[str],
    yields: pd.Series,
    pcs_to_hedge: list[str] | None = None,
) -> dict[str, float]:
    """Find hedge notionals that neutralize a chosen subset of PC exposures.

    Sets up a linear system

        E @ n  =  -p_exp

    where ``p_exp`` is the portfolio's exposure to the PCs we want to
    hedge, ``n`` is the (unknown) vector of hedge notionals in dollars,
    and ``E[k, j]`` is the exposure of $1 of hedge instrument ``j`` to
    ``PC_k``. The number of hedge instruments must equal the number of
    PCs to hedge so the system is square; otherwise it would be either
    over- or under-determined.

    Args:
        portfolio: The portfolio to hedge.
        pca: The fitted ``YieldCurvePCA``.
        hedge_maturities: Maturity labels of the hedge instruments
            (must equal in length to ``pcs_to_hedge``).
        yields: Current yield curve, used to price the hedge instruments
            (DV01 depends on yield).
        pcs_to_hedge: Subset of PC names to neutralize. Defaults to all
            of pca's components.

    Returns:
        Dict ``{maturity_label: hedge_notional_dollars}`` suitable for
        feeding into ``Portfolio.from_holdings``. Negative notionals
        mean short positions (the typical case for hedging a long
        bond portfolio).

    Raises:
        ValueError: if the dimensions don't match or the hedge matrix
            is singular (e.g. two hedge instruments with collinear PC
            exposures).
    """
    if pcs_to_hedge is None:
        pcs_to_hedge = list(pca.loadings.index)

    if len(hedge_maturities) != len(pcs_to_hedge):
        raise ValueError(
            f"Number of hedge instruments ({len(hedge_maturities)}) must "
            f"match number of PCs to hedge ({len(pcs_to_hedge)})."
        )

    # Portfolio exposures to the targeted PCs
    p_exp = pc_exposures(portfolio, pca).loc[pcs_to_hedge]

    # exposure_matrix[k, j] = exposure of $1 of hedge instrument j to PC k
    n = len(pcs_to_hedge)
    exposure_matrix = np.zeros((n, n))
    for j, mat in enumerate(hedge_maturities):
        unit_dv01 = Bond(mat, 1.0, float(yields[mat])).dv01
        for k, pc in enumerate(pcs_to_hedge):
            exposure_matrix[k, j] = unit_dv01 * pca.loadings.loc[pc, mat]

    try:
        notionals = np.linalg.solve(exposure_matrix, -p_exp.values)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Hedge matrix is singular — chosen instruments don't span the "
            "targeted PC subspace. Pick instruments at more spread maturities."
        ) from exc

    return dict(zip(hedge_maturities, notionals.tolist(), strict=True))


def hedged_portfolio(
    portfolio: Portfolio,
    hedge: dict[str, float],
    yields: pd.Series,
) -> Portfolio:
    """Return a new ``Portfolio`` = original + hedge bonds.

    Args:
        portfolio: Original portfolio.
        hedge: Output of ``solve_hedge`` (``{maturity_label: notional}``).
        yields: Yield curve to price the hedge bonds at.

    Returns:
        ``Portfolio`` containing the original bonds plus one ``Bond`` per
        hedge entry.
    """
    hedge_bonds = [
        Bond(maturity_label=mat, notional=notional, yield_pct=float(yields[mat]))
        for mat, notional in hedge.items()
    ]
    return Portfolio(bonds=list(portfolio.bonds) + hedge_bonds)


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


def walk_forward_hedge(
    portfolio_holdings: dict[str, float],
    yields: pd.DataFrame,
    changes_bp: pd.DataFrame,
    hedge_maturities: list[str],
    pcs_to_hedge: list[str],
    window_days: int = 252,
    rebalance_freq: str = "ME",
    n_components: int = 3,
) -> dict:
    """Walk-forward immunization with rolling-window PCA fits.

    At each rebalance date ``t``:

    1. Fit a fresh ``YieldCurvePCA`` on the last ``window_days`` rows of
       ``changes_bp`` ending at (and including) ``t`` — *no* future data.
    2. Build the portfolio at ``yields.loc[t]`` and solve hedge notionals
       against the locally-fitted PCA.
    3. Apply that hedge to ``changes_bp`` from ``t + 1`` business day to
       (and including) the next rebalance date. The strict ``>`` here
       prevents counting ``changes_bp[t]`` — which is informationally
       known at close of ``t`` and used for fitting — as out-of-sample
       P&L.

    Skipping rule: rebalance dates where ``loc < window_days - 1``
    (not enough history) are dropped silently.

    Args:
        portfolio_holdings: ``{maturity_label: notional}`` for the
            portfolio to hedge (rebuilt at each rebalance with current yields).
        yields: Daily yield curve panel (one row per business day, in %).
        changes_bp: Daily yield changes in bp, same calendar as ``yields``
            but starting one row later (because of ``.diff()``).
        hedge_maturities: Hedge instrument maturity labels.
        pcs_to_hedge: PC names to neutralize at each rebalance.
        window_days: Trailing rolling-window length (in business days) for
            the PCA fit. Default 252 ≈ 1 year.
        rebalance_freq: pandas resample alias (default ``"ME"`` = month-end).
            Use ``"W"`` for weekly, ``"QE"`` for quarterly, etc.
        n_components: PC count for the local PCA fits.

    Returns:
        Dict with three entries:

        * ``"pnl"`` (``pd.Series``) — out-of-sample daily P&L in dollars,
          indexed by date. Days during warm-up (before the first valid
          rebalance) are excluded.
        * ``"hedge_history"`` (``pd.DataFrame``) — one row per rebalance,
          columns ``hedge_<maturity>`` for each instrument's notional.
        * ``"rebalance_dates"`` (``pd.DatetimeIndex``) — actually-used
          rebalance dates (after warm-up filter).
    """
    # Rebalance schedule snapped to business days in yields.index
    rebal_target = yields.resample(rebalance_freq).last().index
    rebal_dates = pd.DatetimeIndex([d for d in rebal_target if d in yields.index])

    pnl = pd.Series(np.nan, index=changes_bp.index, name="walk_forward_pnl_dollars")
    hedge_records: list[dict] = []
    used_rebal: list[pd.Timestamp] = []

    for i, t in enumerate(rebal_dates):
        if t not in changes_bp.index:
            continue
        loc = changes_bp.index.get_loc(t)
        if loc < window_days - 1:
            continue

        # Past-only training window (window_days rows ending at, and
        # including, t). changes_bp[t] is already in the info set at
        # close of t, so including it is not look-ahead.
        train_data = changes_bp.iloc[loc - window_days + 1 : loc + 1]

        pca_local = YieldCurvePCA(n_components=n_components).fit(train_data)

        current_yields = yields.loc[t]
        portfolio_t = Portfolio.from_holdings(portfolio_holdings, current_yields)
        hedge_t = solve_hedge(
            portfolio_t, pca_local, hedge_maturities, current_yields, pcs_to_hedge
        )
        combined_t = hedged_portfolio(portfolio_t, hedge_t, current_yields)

        # Apply hedge to changes from t+1 (strict) until next rebalance (inclusive)
        next_t = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else changes_bp.index[-1] + pd.Timedelta(days=1)
        )
        period_mask = (changes_bp.index > t) & (changes_bp.index <= next_t)
        period_changes = changes_bp.loc[period_mask]
        if len(period_changes) > 0:
            period_pnl = daily_pnl_direct(combined_t, period_changes)
            pnl.loc[period_changes.index] = period_pnl.values

        hedge_records.append(
            {"rebalance_date": t, **{f"hedge_{m}": n for m, n in hedge_t.items()}}
        )
        used_rebal.append(t)

    hedge_history = (
        pd.DataFrame(hedge_records).set_index("rebalance_date")
        if hedge_records
        else pd.DataFrame()
    )

    return {
        "pnl": pnl.dropna(),
        "hedge_history": hedge_history,
        "rebalance_dates": pd.DatetimeIndex(used_rebal),
    }


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
