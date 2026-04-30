"""Yield-curve PCA wrapper.

A thin layer over ``sklearn.decomposition.PCA`` that:

* preserves the maturity/date labels so loadings and scores stay readable,
* exposes ``loadings``, ``explained_variance_ratio_``, ``cumulative_variance_ratio_``
  as proper pandas objects rather than bare numpy arrays,
* knows how to ``save``/``load`` itself via ``joblib``.

Design choice: we **do not** standardize the input. All maturities are
already in the same unit (bp), and their daily volatilities sit in a
similar range (~3-7 bp). Standardizing would rescale that away and make
the loadings unreadable in bp terms — see Notebook 03 for the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import pandas as pd
from sklearn.decomposition import PCA


class YieldCurvePCA:
    """PCA wrapper specialized for daily yield-change matrices.

    Typical usage::

        pca = YieldCurvePCA(n_components=3).fit(changes_bp)
        scores = pca.transform(changes_bp)
        print(pca.loadings)
        print(pca.explained_variance_ratio_)
    """

    def __init__(self, n_components: int | None = None) -> None:
        """Args:
            n_components: how many principal components to keep. ``None`` keeps
                all of them (i.e. ``min(n_samples, n_features)``).
        """
        self.n_components = n_components
        self._pca: PCA | None = None
        self._columns: pd.Index | None = None

    # ------------------------------------------------------------------ #
    # Fit / transform                                                    #
    # ------------------------------------------------------------------ #

    def fit(self, changes_bp: pd.DataFrame) -> Self:
        """Fit PCA on a ``(N_days, N_maturities)`` matrix of daily bp changes."""
        self._columns = changes_bp.columns
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(changes_bp.values)
        return self

    def transform(self, changes_bp: pd.DataFrame) -> pd.DataFrame:
        """Project new data onto the fitted PCs.

        Returns a DataFrame with one column per PC (``PC1``, ``PC2``, ...) and
        the original date index.
        """
        self._check_fitted()
        scores = self._pca.transform(changes_bp.values)
        return pd.DataFrame(scores, index=changes_bp.index, columns=self._pc_names())

    def fit_transform(self, changes_bp: pd.DataFrame) -> pd.DataFrame:
        """Fit on the data and immediately return the in-sample scores."""
        return self.fit(changes_bp).transform(changes_bp)

    def inverse_transform(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct the (approximate) input from PC scores."""
        self._check_fitted()
        reconstructed = self._pca.inverse_transform(scores.values)
        return pd.DataFrame(reconstructed, index=scores.index, columns=self._columns)

    # ------------------------------------------------------------------ #
    # Properties: pretty pandas views over the underlying numpy arrays   #
    # ------------------------------------------------------------------ #

    @property
    def loadings(self) -> pd.DataFrame:
        """Eigenvector matrix as a DataFrame: ``rows = PCs, cols = maturities``."""
        self._check_fitted()
        return pd.DataFrame(
            self._pca.components_,
            index=self._pc_names(),
            columns=self._columns,
        )

    @property
    def explained_variance_ratio_(self) -> pd.Series:
        """Per-component variance fraction (sums to ≤1)."""
        self._check_fitted()
        return pd.Series(
            self._pca.explained_variance_ratio_,
            index=self._pc_names(),
            name="explained_variance_ratio",
        )

    @property
    def cumulative_variance_ratio_(self) -> pd.Series:
        """Running total of ``explained_variance_ratio_``."""
        return self.explained_variance_ratio_.cumsum().rename("cumulative")

    @property
    def explained_variance_(self) -> pd.Series:
        """Per-component eigenvalue (variance, not ratio)."""
        self._check_fitted()
        return pd.Series(
            self._pca.explained_variance_,
            index=self._pc_names(),
            name="explained_variance",
        )

    @property
    def n_components_(self) -> int:
        self._check_fitted()
        return self._pca.n_components_

    @property
    def mean_(self) -> pd.Series:
        """Per-feature mean used for centering (sklearn convention).

        sklearn's ``PCA.transform`` subtracts this before projecting, so any
        downstream code that wants to reconstruct *uncentered* data from
        scores needs to add it back. Used by ``immunization.daily_pnl_via_pcs``
        to keep the mean (drift) component consistent with the direct P&L.
        """
        self._check_fitted()
        return pd.Series(self._pca.mean_, index=self._columns, name="mean")

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Pickle the fitted estimator (and labels) to disk."""
        self._check_fitted()
        joblib.dump({"pca": self._pca, "columns": self._columns}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> YieldCurvePCA:
        """Restore a previously-saved instance."""
        payload = joblib.load(Path(path))
        instance = cls(n_components=payload["pca"].n_components_)
        instance._pca = payload["pca"]
        instance._columns = payload["columns"]
        return instance

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("YieldCurvePCA must be fit before this operation.")

    def _pc_names(self) -> list[str]:
        return [f"PC{i + 1}" for i in range(self._pca.n_components_)]

    def __repr__(self) -> str:
        if self._pca is None:
            return f"YieldCurvePCA(n_components={self.n_components}, fitted=False)"
        evr = self._pca.explained_variance_ratio_
        return (
            f"YieldCurvePCA(n_components={self._pca.n_components_}, "
            f"explained={evr.sum():.1%}, top3={evr[:3].round(3).tolist()})"
        )


def fit_pca_from_yields(
    yields_clean: pd.DataFrame,
    n_components: int | None = None,
) -> tuple[YieldCurvePCA, pd.DataFrame]:
    """Convenience: clean yields → bp changes → fit PCA in one call.

    Args:
        yields_clean: Already-cleaned yields (no NaN, in %).
        n_components: Forwarded to ``YieldCurvePCA``.

    Returns:
        ``(fitted_pca, changes_bp)`` tuple. The DataFrame is returned so
        callers can immediately do downstream things like ``.transform``.
    """
    from ..data.preprocessor import to_changes_bp

    changes_bp = to_changes_bp(yields_clean)
    pca = YieldCurvePCA(n_components=n_components).fit(changes_bp)
    return pca, changes_bp
