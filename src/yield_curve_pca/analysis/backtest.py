"""PC-score-based mean reversion strategy backtest.

The strategy looks at the *cumulative* PC score (which behaves like a
slow-moving level), computes a rolling z-score relative to its recent
history, and bets on reversion when ``|z|`` exceeds a threshold.

A handful of optional refinements layer on top:

* ``trend_filter_threshold``: don't enter if recent momentum is large
  (the strategy bleeds in trending periods — see Notebook 07).
* ``stop_loss_bp`` / ``max_hold_days``: bound the worst-case trade.
* ``regime_filter``: a per-day "OK"/"NG" Series; ``"NG"`` blocks entry.
* asymmetric ``enter_threshold`` vs ``exit_threshold``: enter only on
  strong signals, but exit early once they fade.

All decisions for day ``t`` use only information available at end of
day ``t-1`` (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.config import ANNUALIZATION_FACTOR


@dataclass
class BacktestResult:
    """Container for the output of a single backtest.

    Attributes:
        name: Human-readable label.
        daily_pnl: Per-day P&L in bp (units inherited from the input PC score).
        cum_pnl: Cumulative sum of ``daily_pnl``.
        position: Per-day position (-1, 0, or +1).
        metrics: Summary stats — see ``compute_metrics``.
    """

    name: str
    daily_pnl: pd.Series
    cum_pnl: pd.Series
    position: pd.Series
    metrics: dict


def compute_metrics(daily_pnl: pd.Series, position: pd.Series) -> dict:
    """Compute the standard backtest metrics from a P&L series.

    Returns:
        Dict with ``total_pnl_bp``, annualized ``sharpe``, ``max_dd_bp``,
        ``hit_rate`` (over active days), ``n_trades``, and ``active_pct``.
    """
    valid = daily_pnl.dropna()
    if valid.std() > 0:
        sharpe = (valid.mean() / valid.std()) * np.sqrt(ANNUALIZATION_FACTOR)
    else:
        sharpe = 0.0

    cum = valid.cumsum()
    drawdown = cum - cum.cummax()

    n_active = (position != 0).sum()
    n_total = len(position.dropna())
    pos_changes = max(int((position.diff() != 0).sum()) - 1, 0)

    active_pnl = valid[position.dropna() != 0]
    hit_rate = (active_pnl > 0).mean() if len(active_pnl) > 0 else 0.0

    return {
        "total_pnl_bp": float(cum.iloc[-1]) if not cum.empty else 0.0,
        "sharpe": float(sharpe),
        "max_dd_bp": float(drawdown.min()) if not drawdown.empty else 0.0,
        "hit_rate": float(hit_rate),
        "n_trades": int(pos_changes),
        "active_pct": float(n_active / n_total * 100) if n_total > 0 else 0.0,
    }


def make_signal(
    pc_score: pd.Series,
    window: int = 60,
    enter_threshold: float = 1.5,
    exit_threshold: float | None = None,
    trend_filter_threshold: float | None = None,
    stop_loss_bp: float | None = None,
    max_hold_days: int | None = None,
    regime_filter: pd.Series | None = None,
) -> pd.Series:
    """Generate a per-day position from a PC score, with optional filters.

    The base logic is z-score mean reversion on the *cumulative* PC score:
    a long position bets the cumulative score will rise back toward its
    rolling mean (``z`` was below ``-enter_threshold``); a short position
    is the mirror.

    Args:
        pc_score: Daily PC score series (e.g. ``pc_scores["PC2"]``).
        window: Rolling window (in days) for mean and std of the cumulative
            score.
        enter_threshold: Open a position when the lagged ``|z|`` exceeds this.
        exit_threshold: Close the position when ``|z|`` drops below this.
            Defaults to ``enter_threshold`` (symmetric).
        trend_filter_threshold: Block new entries while ``|sum of last 20 days
            of pc_score|`` exceeds this. ``None`` disables the filter.
        stop_loss_bp: Force-exit when in-trade cumulative P&L drops below
            ``-stop_loss_bp``. ``None`` disables.
        max_hold_days: Force-exit after this many bars. ``None`` disables.
        regime_filter: Per-day Series with values ``"OK"`` or ``"NG"``;
            ``"NG"`` blocks new entries. ``None`` disables.

    Returns:
        Position series aligned to ``pc_score.index``.
    """
    if exit_threshold is None:
        exit_threshold = enter_threshold

    cum = pc_score.cumsum()
    rolling_mean = cum.rolling(window).mean()
    rolling_std = cum.rolling(window).std()
    z = (cum - rolling_mean) / rolling_std
    z_lag = z.shift(1)

    trend_lag = (
        pc_score.rolling(20).sum().shift(1).abs()
        if trend_filter_threshold is not None
        else None
    )

    signal = pd.Series(0.0, index=pc_score.index)
    position = 0.0
    holding_days = 0
    entry_pnl = 0.0

    for i, _ in enumerate(pc_score.index):
        # --- exit checks (only when in a position) -----------------------
        if position != 0:
            holding_days += 1
            entry_pnl += position * pc_score.iloc[i]

            forced_exit = False
            if stop_loss_bp is not None and entry_pnl < -stop_loss_bp:
                forced_exit = True
            elif max_hold_days is not None and holding_days >= max_hold_days:
                forced_exit = True
            elif pd.notna(z_lag.iloc[i]) and abs(z_lag.iloc[i]) < exit_threshold:
                forced_exit = True

            if forced_exit:
                position = 0.0
                holding_days = 0
                entry_pnl = 0.0
                signal.iloc[i] = position
                continue

        # --- entry checks (only when flat) --------------------------------
        if position == 0 and pd.notna(z_lag.iloc[i]):
            if trend_lag is not None and pd.notna(trend_lag.iloc[i]):
                if trend_lag.iloc[i] > trend_filter_threshold:
                    signal.iloc[i] = position
                    continue

            if regime_filter is not None and regime_filter.iloc[i] == "NG":
                signal.iloc[i] = position
                continue

            if z_lag.iloc[i] > enter_threshold:
                position = -1.0  # short PC: bet on reversion downward
            elif z_lag.iloc[i] < -enter_threshold:
                position = +1.0  # long PC: bet on reversion upward
            holding_days = 0
            entry_pnl = 0.0

        signal.iloc[i] = position

    return signal


def backtest(
    name: str,
    pc_score: pd.Series,
    **signal_kwargs,
) -> BacktestResult:
    """Run a single backtest end-to-end and bundle the results.

    Convenience wrapper around ``make_signal`` + ``compute_metrics``.
    """
    sig = make_signal(pc_score, **signal_kwargs)
    pnl = sig * pc_score
    return BacktestResult(
        name=name,
        daily_pnl=pnl,
        cum_pnl=pnl.cumsum(),
        position=sig,
        metrics=compute_metrics(pnl, sig),
    )


def classify_regime(
    pc_score: pd.Series,
    window: int = 60,
    threshold: float = 30.0,
) -> pd.Series:
    """Threshold-based regime classifier.

    The simplest possible regime split:

    * ``"trending"`` when the absolute rolling-window sum of ``pc_score``
      exceeds ``threshold`` (interpretation: cumulative move has been large);
    * ``"ranging"`` otherwise.

    Notebook 07 found this single filter — applied to PC2 with window 60 and
    threshold 30 — was enough to flip the base mean-reversion strategy from
    a negative Sharpe to a (just barely) positive one.

    Args:
        pc_score: Daily PC score series.
        window: Rolling window in days.
        threshold: Cutoff (in same units as cumulative score) above which we
            label the day "trending".

    Returns:
        Series of strings (``"trending"`` / ``"ranging"``) with the same index.
    """
    abs_momentum = pc_score.rolling(window).sum().abs()
    regime = pd.Series("ranging", index=pc_score.index)
    regime[abs_momentum > threshold] = "trending"
    return regime


def regime_to_filter(regime: pd.Series) -> pd.Series:
    """Convert a ``"trending"``/``"ranging"`` series into the OK/NG filter
    expected by ``make_signal``.

    Lags by one day to keep the strategy free of look-ahead.
    """
    return regime.map({"trending": "NG", "ranging": "OK"}).shift(1)
