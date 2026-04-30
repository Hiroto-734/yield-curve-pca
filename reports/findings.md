# Findings — Yield Curve PCA Analysis

**Period covered**: 2020-01-02 to 2026-04-27 (1,580 business days, 10 maturities from 3M to 30Y)

This document is the project's written deliverable — the version you'd hand to an interviewer who asks "tell me about this project." For the underlying analysis, see the numbered notebooks under `notebooks/`.

---

## 1. The factor structure (Notebooks 01-04)

PCA on the daily yield-change matrix recovers Litterman & Scheinkman (1991) on the 2020-2026 sample:

| Component | Variance explained | Human-indicator counterpart | Daily correlation with counterpart |
|-----------|--------------------|------------------------------|------------------------------------|
| **PC1: Level** | 79.8% | 10Y yield | **0.96** |
| **PC2: Slope** | 11.8% | 2s10s spread (10Y − 2Y) | **0.91** |
| **PC3: Curvature** | 4.6% | Butterfly (2×5Y − (2Y+10Y)) | 0.26 (narrow) → 0.59 (best wide) |
| Top 3 cumulative | **96.2%** | | |

**Why PC1 lands at 79.8% rather than the textbook ~85%**: the 2020-2026 window contains the historic 2022-2024 inversion, during which the curve moved more in slope than in level. Pre-COVID samples typically show 85-90% for PC1.

**Why PC3 misses the simple butterfly**: PC3 has its strongest loadings at 3M (+0.70) and 30Y (+0.26), maturities the 2-5-10 butterfly literally doesn't see. No 3-point butterfly can replicate a 10-point loading vector. This is exactly where PCA's optimality buys something a hand-built indicator cannot.

---

## 2. Three case studies (Notebook 05)

The single most useful exercise was decomposing real market events into PC scores and reading them against the textbook expectations.

### 2.1 COVID emergency cut — 2020-03-16

The Sunday-night Fed announcement of a 100bp cut to zero plus $700B QE was, by the textbook, a Bull-Steepener (rates collapse, short more than long). Our PC scores show otherwise: **PC1 = -3.34σ, PC2 = -1.62σ — a Bull-Flattener.**

Reading the daily moves: Δ3M = -4 bp, Δ30Y = -22 bp. The 3M had nowhere left to fall (already pinned at the zero floor), so the entire reaction was concentrated in the long end via the flight-to-safety bid for Treasuries. The textbook label assumes the front end has room to move — context overrode the label.

### 2.2 First 75bp hike since 1994 — 2022-06-15

A 75bp hike on a hike day "should" produce a Bear-Flattener. Our PC scores: **PC1 = -3.44σ, PC2 = +3.11σ — a Bull-Steepener (rates fall on a hike day).**

Reading the daily moves: Δ2Y = -25 bp, Δ30Y = -6 bp. The Wall Street Journal had leaked the 75bp two days earlier (June 13), so by June 15 the 75bp was fully priced. Powell's press conference framed it as "not a regular occurrence," which removed the tail of more-hawkish surprises. Relief buying concentrated where the rate-hike pricing had been most aggressive: the front end. **What matters is (action − expectation), not the action itself.**

### 2.3 First cut, 50bp — 2024-09-18

The end of the hiking cycle. **PC2 = +1.88σ dominates — a Steepener** despite being a cut day. Δ3M = -11 (cut filtering through), but Δ10Y = +5 and Δ30Y = +7 (long end *up*). Powell framed the 50bp as an "insurance cut" against a still-strong economy, which removed the recession premium from the long end. The day marks the structural unwind of the 2022-2024 inversion.

---

## 3. PC2 mean reversion strategy (Notebooks 06-08)

I built a systematic strategy on PC2 because mean reversion is the obvious application of the regime/level interpretation, then evaluated it honestly.

### 3.1 The base strategy and its failure

A standard z-score mean-reversion on cumulative PC2 (60-day window, |z| > 1.5σ entry, symmetric exit) **lost money over 2020-2026**:

| Metric | Value |
|--------|-------|
| Total P&L | **−181 bp** |
| Annualized Sharpe | **−0.44** |
| Max drawdown | −284 bp |
| Hit rate | 48% (coin flip) |
| Trades | 164 |

A 6×6 sensitivity grid over windows {20, 40, 60, 90, 120, 252} and thresholds {1.0, 1.25, 1.5, 1.75, 2.0, 2.5} confirmed this isn't a parameter-choice problem — almost every cell is negative. Comparison against 100 random strategies matched on long/short/flat frequencies put the actual strategy at the left tail (z = −1.09σ).

### 3.2 The diagnosis

Loss attribution found two highly significant predictors of daily P&L (p < 0.001 each):

* **|20-day PC2 trend|** correlates **−0.30** with P&L. The strategy bleeds in trending markets — exactly the textbook failure mode of mean reversion.
* **|z-score|** correlates **−0.55** with P&L. **Bigger signals correlate with worse trades**, the opposite of the textbook expectation. Extreme z-scores arise during sustained moves, not at turning points.

The second finding is what made the asymmetric "enter only at 2.0σ" variant *worse* than the base, not better.

### 3.3 What worked

I tested five filters: asymmetric thresholds, stop-loss, time-based exit, trend filter, and a regime classifier. **Only the regime classifier turned the Sharpe positive.**

The classifier itself is one rule: "ranging" if |60-day cumulative PC2| ≤ 30bp, else "trending." Trade only in ranging regimes.

| Strategy variant | Sharpe |
|-------------------|--------|
| Base | −0.44 |
| Asymmetric (enter 2.0σ / exit 0.5σ) | −0.88 (worst) |
| Stop loss −30bp | −0.16 |
| Time exit 20d | −0.42 |
| Trend filter | −0.61 |
| **Regime classifier** | **+0.06** |
| Combined (all five) | −0.46 (over-filtered, only 9 trades) |

**Bottom line: when to trade is a bigger lever than how to trade.** The regime-only variant trades 17% of days at a 52% hit rate, takes Sharpe to barely positive, and cuts max drawdown by 67%. It's still well below an investable threshold, but the direction is set by data rather than by parameter tuning.

### 3.4 The realistic ceiling

Perfect-foresight projection onto PC2 (knowing tomorrow's score today) gives Sharpe ~16 — unreachable. The useful version is the hit-rate-to-Sharpe map: 50% accuracy → Sharpe 0; 52% → 0.5 (production minimum); 54% → 1.0 (top-systematic territory). The whole game in this style of strategy is finding 1-2 percentage points of edge over a coin flip on a daily signal.

---

## 4. Macro regression (Notebook 09)

Regressing PC scores against approximate CPI / NFP / FOMC surprises gives a clean and honest picture:

* **Release-day effect is real**: PC1 std on release days is 22.9 vs 14.9 on non-release days — a **1.54x ratio**. Same picture for PC2 (1.58x), weaker for PC3 (1.23x), consistent with curvature being driven more by supply/demand than by macro releases.
* **The FOMC dummy is the only feature that survives joint regression**: β = **−7.24 bp on PC1, p < 0.001**. Negative-on-average matches the case studies — over 2020-2026, FOMC days have skewed toward "less hawkish than priced" reactions.
* **CPI and NFP coefficients are not significant in any specification, R² < 1% across the board.** This is a data-quality problem, not a structure problem: my "proxy surprise" is `(actual MoM) − (6-month rolling average MoM)`, which captures deviation from recent trend but not what the market reacts to, which is `(actual) − (consensus forecast)`. With Bloomberg consensus data, comparable studies typically find R² of 5-20%.

The notebook is therefore a quantification of **what is and isn't explainable from public macro data**, rather than a working alpha signal.

---

## 5. PCA-based portfolio immunization (Notebooks 10-11)

The strategy notebooks (06-09) showed PCA can't be predicted off — the directional bet on PC2 lost money. Notebooks 10-11 turn the same factors into a **risk management tool** instead, which is what Litterman & Scheinkman's original paper proposed them for. The contrast is the project's clearest finding.

### 5.1 Setup

A vanilla long-30Y portfolio: $100M notional, modified duration ~30 years, daily P&L std of **$1.66M** (≈ $26M annualized) when unhedged. The hedge variants:

| Variant | Hedge instruments | PCs neutralized |
|---------|--------------------|-----------------|
| Unhedged | — | — |
| PC1 hedge | 10Y | PC1 (Level) |
| PC1+PC2 hedge | 2Y, 10Y | PC1, PC2 |
| PC1+PC2+PC3 hedge | **3M, 5Y, 20Y** | PC1, PC2, PC3 |

The 3-PC hedge deliberately excludes 30Y as an instrument: with the original portfolio also being 30Y, including it lets the linear system collapse to "just short the original," which is correct but uninformative. Choosing instruments at distinct maturities forces a real factor-neutralizing combination.

### 5.2 Headline result: 96% variance reduction

| Variant | Daily std | % of unhedged variance | Variance reduction |
|---------|-----------|------------------------|---------------------|
| Unhedged | $1.66M | 100.00% | — |
| PC1 hedged | $0.64M | 14.99% | **85.01%** |
| PC1+PC2 hedged | $0.53M | 10.29% | **89.71%** |
| **PC1+PC2+PC3 hedged** | **$0.33M** | **3.97%** | **96.03%** |

The residual 3.97% closely matches the PCA's 3.8% unexplained variance (1 − 96.2%) — exactly what the math predicts. Cumulative-P&L plots show the unhedged portfolio drifting through nine-figure swings during the 2022-2024 hiking cycle, while the fully-hedged portfolio sits visibly flat.

### 5.3 The instrument-selection finding

A genuinely surprising side-result: the same "hedge PC1+PC2+PC3" task with different instrument choices gives meaningfully different residual variance:

| 3-PC hedge instruments | PC4-10 residual variance | Var reduction |
|------------------------|---------------------------|----------------|
| {2Y, 5Y, 10Y} | 3.4 × 10¹¹ | 87.8% |
| {3M, 5Y, 10Y} | 2.5 × 10¹¹ | — |
| {3M, 2Y, 10Y} | 3.1 × 10¹¹ | — |
| {2Y, 7Y, 20Y} | 1.2 × 10¹¹ | — |
| **{3M, 5Y, 20Y}** | **1.1 × 10¹¹** | **96.0%** |

Same number of factors hedged, same factors zeroed, but a 3× difference in residual variance depending on which maturities span the hedge. **Picking the hedge basket is a sub-problem in its own right** — what real risk desks optimize over, not just "how many factors to neutralize." A simple first principle that emerged: spread the instruments across the curve (one short, one belly, one long) rather than clustering them in the belly.

### 5.4 Walk-forward validation (Notebook 12)

A natural objection to the 96.03% figure is that the hedge ratios were
constructed using PCA loadings fitted on the *full* 2020-2026 sample,
which technically uses future data. Notebook 12 addresses this directly
by running a true walk-forward backtest:

* At each month-end ``t``, fit a fresh PCA on only the trailing 252
  business days (no future information);
* Use *those* loadings together with current yields at ``t`` to compute
  hedge notionals;
* Apply the hedge from ``t+1`` until the next rebalance.

Results on the same out-of-sample evaluation period (~5 years, 1268
business days):

| Setup | Variance reduction |
|-------|---------------------|
| In-sample (full-period PCA, t=0 hedge held to end) | **96.53%** |
| **Walk-forward (252d window, monthly rebalance)** | **93.27%** |
| Walk-forward (504d window, monthly) | 96.67% |
| Walk-forward (504d window, quarterly) | 96.63% |

The look-ahead premium is **3.26 percentage points** with a 252-day
window — small enough that the in-sample result was largely valid, and
large enough that the honest version (93.27%) is the number to quote.
Either way the conclusion holds: the same PCA factors that couldn't
be predicted off can be hedged off cleanly, even with no future
information.

Why this works: visualizing the loadings re-fit at six different
historical dates shows the Level / Slope / Curvature *shape* is
essentially constant across the sample. PCA loadings drift slowly
because the underlying economic mechanisms (monetary policy
transmission, term premium, supply/demand) are themselves stable —
which is exactly Litterman & Scheinkman's original observation
that has held since 1991.

### 5.5 The asymmetry result

The same PCA model that lost money trying to predict PC2 (Notebook 06: Sharpe -0.38) eliminates 93-96% of variance when used to hedge factor exposures (Notebooks 11-12). Same factors, opposite outcomes — this is the project's headline asymmetry:

- **Prediction needs direction** (which way does PC2 go tomorrow?) → can't be answered → naive strategies lose
- **Hedging needs structure** (how does the curve move when it moves?) → answered by stable loadings → factor-neutralization works, in-sample *and* out-of-sample

This matches the message of Litterman & Scheinkman 1991 — its title is "Common Factors Affecting Bond *Returns*," not "Predicting Bond Returns" — and is the cleanest summary of where PCA earns its keep in fixed income.

---

## 6. Five-minute spoken summary

> I applied PCA to daily changes in the US Treasury yield curve from 2020 through April 2026 — ten maturities from 3M to 30Y. The top three components explain 96.2% of variance and recover the textbook Level / Slope / Curvature factor structure.
>
> Comparing the PCs to the traditional human indicators: PC1 vs Δ10Y correlates at 0.96, PC2 vs Δ2s10s at 0.91, but PC3 vs the 2-5-10 butterfly is only 0.26. That gap is the most interesting finding — Level and Slope can be approximated by simple human indicators, but Curvature is where PCA's full 10-point loading vector actually buys you something a 3-point indicator literally cannot represent.
>
> I then built a PC2 mean-reversion strategy as a development task. The naive version lost money — Sharpe −0.38 — and I diagnosed why with loss attribution: extreme z-scores correlate negatively with forward P&L (r = −0.55), the opposite of the textbook expectation, because extreme z arises during sustained moves rather than at turning points. Adding a single regime filter — trade only when the 60-day cumulative PC2 is within ±30bp of zero — turned the Sharpe positive at +0.06. Still noise-level for live deployment, but it isolates the real lever: when you trade matters more than how.
>
> Three case studies reinforce the broader lesson that surface labels mislead without context: the COVID emergency cut produced a Bull-Flattener instead of a Bull-Steepener (front end pinned at zero, flight-to-safety in the long end); the 75bp hike of June 2022 produced a Bull-Steepener instead of a Bear-Flattener (WSJ leak two days earlier had priced it in); the first cut of September 2024 produced a pure Steepener despite cutting (Powell's "insurance cut" framing removed recession premium from the long end). The market reacts to the difference between action and expectation, not to action alone.
>
> Pivoting from prediction to risk management on the same factors, I built a PCA-based immunization for a $100M long-30Y portfolio. Hedging PC1 alone with a 10Y bond eliminates 85% of the daily P&L variance; adding a 2Y to neutralize PC2 takes it to 90%; adding 3M and 20Y to neutralize PC3 takes it to **96%** — exactly the explained-variance budget that the PCA started with. To address the look-ahead concern in that in-sample result, I added a walk-forward backtest that re-fits the PCA monthly on only past data, and the variance reduction stays at **93.27%** — a 3-point gap that confirms the in-sample number wasn't an artifact of using future information. The same factors that couldn't be predicted off can be hedged off cleanly, in-sample *and* out-of-sample, which is the message Litterman & Scheinkman put in the paper title: "Common Factors Affecting Bond *Returns*," not "Predicting Bond Returns."
>
> The whole project is reproducible — installable package, 50 unit tests passing in CI, every parquet artifact and figure regenerable from the numbered notebooks.

---

## 7. Limitations and what I'd do next

* **Macro regression**: needs Bloomberg/Refinitiv consensus surprise data to do justice to the question.
* **Strategy refinement**: the regime classifier is threshold-based. An HMM or a small classifier using cross-asset features (VIX, credit spreads, equity momentum) is the natural next step. Realistic expectation: maybe 0.1-0.3 of additional Sharpe.
* **Cross-country comparison**: applying the same PCA to German Bunds and looking at PC-level cross-country relationships is the cleanest unfinished thread.
* **Transaction costs are not in the backtest** — adding 0.25-0.5 bp per trade leg would shift every Sharpe by roughly −0.1 to −0.3, and would reduce the walk-forward variance reduction by a few percentage points (monthly rebalance × 4 instruments × ~0.5bp ≈ 2bp/month of friction).
* **Interest rate swap pricer**: the immunization story currently uses cash bonds only. Building a swap pricer (bootstrap discount curve → forward rates → DV01 → PCA factor exposures) would let the same factor-hedging machinery work on derivatives, which is the natural next extension and what real rates desks operate on.

---

## Figures

* `reports/figures/02_monthly_snapshots.png` — six-year curve evolution
* `reports/figures/03_loadings.png` — Level / Slope / Curvature shapes
* `reports/figures/04_pc_vs_human_cumulative.png` — PCA vs traditional indicators
* `reports/figures/05_case_*.png` — three case-study zoom-ins
* `reports/figures/07_strategies_comparison.png` — all six strategy variants on one chart
* `reports/figures/08_hitrate_sharpe.png` — hit-rate-to-Sharpe mapping with industry benchmarks
