# Glossary

A focused glossary of the financial and statistical terms used throughout this project. Definitions are deliberately concise and project-relevant, not exhaustive.

---

## Yield curve and bonds

### **Yield (more precisely, yield to maturity)**
The annualized return implied by buying a bond at today's market price and holding it to maturity. Distinct from the *coupon* (the contractual periodic payment fixed at issuance). When prices move, the yield moves in the opposite direction; the coupon does not.

### **Coupon**
The contractual interest payment a bond makes to its holder, fixed at issuance and quoted as a percentage of face value. Most US Treasury notes and bonds pay semi-annually.

### **Treasury / UST**
Short for "United States Treasury security." The project uses daily Constant Maturity Treasury (CMT) yields from FRED for ten standard maturities.

### **Constant Maturity Treasury (CMT)**
A daily-published yield curve where each point is interpolated to a fixed maturity (e.g., always exactly 10 years to maturity). FRED publishes the CMT series we use; identifiers are `DGS3MO`, `DGS6MO`, `DGS1`, ..., `DGS30`.

### **Yield curve**
The function mapping bond maturity → yield, at a given point in time. Plotted with maturity on the x-axis and yield on the y-axis.

### **Maturity**
Time remaining until a bond's principal is repaid. The project uses ten maturities: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.

### **Inversion**
A state where shorter-dated yields exceed longer-dated yields (e.g., 2Y > 10Y). Historically a leading indicator of US recessions, though the 2022-2024 inversion was driven by Fed inflation-fighting rather than recession pricing.

### **Term premium**
The excess yield demanded by long-dated bond holders as compensation for tying up capital and bearing duration risk. Conceptually: long yield ≈ expected average short rate over the holding period + term premium.

### **Front end / belly / wings**
Trader shorthand for parts of the curve. *Front end* = short maturities (3M-2Y). *Belly* = mid-curve (5Y-7Y). *Wings* = the extremes (front end and 30Y), used especially in the context of butterfly trades.

### **Basis point (bp)**
1 bp = 0.01% = 0.0001. Daily yield changes are typically quoted in bp (e.g., "the 10Y rose 5 bp today"). 100 bp = 1.00%.

---

## Curve dynamics jargon

### **Level / Slope / Curvature**
The three principal components of yield curve changes. Level = parallel shift up/down. Slope = rotation (steepening/flattening). Curvature = bend in the middle relative to the wings.

### **2s10s, 5s30s, 3M2Y**
Slope spread shorthand. `2s10s = 10Y − 2Y`. `5s30s = 30Y − 5Y`. `3M2Y = 2Y − 3M`. Negative means the front end is higher than the back (inversion).

### **Butterfly**
A measure of curvature using three points on the curve. Project default (per spec): `2 × 5Y − (2Y + 10Y)`. Positive when the belly bumps up relative to the line connecting the wings.

### **Bull / Bear (in bonds)**
*Bull* = yields falling (bond prices rising). *Bear* = yields rising (bond prices falling). The opposite convention to equities, because bond prices move inversely to yield.

### **Steepener / Flattener**
*Steepener* = the slope (e.g., 2s10s) widens. *Flattener* = the slope narrows. Combined with bull/bear, gives the four regimes (Bull-Steepener, Bull-Flattener, Bear-Steepener, Bear-Flattener) used in Notebook 02.

### **DV01**
Dollar value of a 1 bp change in yield. The trader-facing measure of duration risk; a portfolio with $50,000 DV01 loses $50,000 for every 1 bp the yield curve rises (parallel shift).

### **Flight to safety**
The market behavior where investors sell risk assets and buy government bonds during a crisis, driving long yields down. Explains why the COVID 2020-03-16 emergency cut produced a Bull-Flattener rather than the textbook Bull-Steepener: the front end was already at zero, so the entire move concentrated in the long end via flight-to-safety bidding.

---

## Statistics and PCA

### **PCA (Principal Component Analysis)**
A linear decomposition that finds the orthogonal directions of maximum variance in a multivariate dataset. Applied to yield-change matrices, it produces the Level / Slope / Curvature factors.

### **Loadings**
The coefficients of each principal component on each input variable; equivalently, the eigenvectors of the covariance matrix. For yield curves, the loadings map each maturity to a weight, and the *shape* of those weights is what we interpret as Level / Slope / Curvature.

### **PC1, PC2, PC3 (PC scores)**
The projections of each daily change vector onto each principal direction. A scalar per day per component, in the same units (bp) as the input.

### **Explained variance ratio**
The fraction of total variance accounted for by each component. Sums to ≤1 across all components.

### **Stationarity**
Loosely: the property that a time series's statistical character doesn't drift over time. Yield levels are non-stationary (trending); yield changes are closer to stationary, which is why PCA is applied to changes, not levels.

### **z-score**
Standardized value: `(current − rolling mean) / rolling std`. Used as the entry signal in the mean-reversion strategy.

### **Sharpe ratio**
Mean daily return ÷ std of daily return, annualized by × √252. The standard risk-adjusted return measure. Industry benchmarks: 0.5 = production minimum, 1.0 = top systematic, 2.0 = legendary (Renaissance Medallion).

### **Hit rate**
Fraction of active days with positive P&L. A 50% hit rate is coin-flipping; 52-54% is what a Sharpe of 0.5-1.0 needs in a daily-signal strategy.

### **Max drawdown**
The largest peak-to-trough decline in the cumulative P&L curve.

---

## Backtest and trading

### **Look-ahead bias**
Using information from the future (relative to the decision point) when constructing a signal. The project's backtest avoids this by lagging the z-score by one day before forming positions, and verifies the absence of look-ahead with a "smash the future" perturbation test.

### **Walk-forward analysis**
A validation method that fits the model on past data, evaluates on the immediately following out-of-sample period, then rolls the window forward. Stricter than "fit on the full sample." This project does *not* use walk-forward — see [Honest Limitations](../README.md#honest-limitations).

### **Mean reversion**
A trading thesis that extreme values revert toward a long-term mean. Implemented in the project as: when the z-score of cumulative PC2 exceeds a threshold, take the opposite-direction position.

### **Trend following**
The opposite thesis: extreme values continue further. Mean reversion strategies tend to lose money in trending markets, which is the failure mode the project diagnoses for naive PC2 mean reversion.

### **Regime classifier**
A model that labels each day with a market state (e.g., "trending" vs "ranging"). The project uses a simple threshold rule on the absolute 60-day cumulative PC2 to decide when to apply the mean-reversion strategy.

### **HMM (Hidden Markov Model)**
A probabilistic state-space model used in finance to infer unobserved regimes. Listed in the project's "next steps" as a more sophisticated alternative to the threshold-based regime classifier currently used.

### **Carry**
The income earned on a bond from holding it (coupon plus roll-down on the curve), independent of price changes. Not used as a signal in this project but referenced as a candidate factor in the development-task discussion.

---

## Macro and events

### **FOMC (Federal Open Market Committee)**
The Federal Reserve committee that sets the US policy rate. Meets eight times per year (plus emergency meetings). Statements are released at 14:00 ET on the second day of each meeting.

### **CPI (Consumer Price Index)**
US headline inflation measure, released monthly by the Bureau of Labor Statistics around the 10th-13th of the month for the prior month's data.

### **NFP (Non-Farm Payrolls)**
US monthly employment change, released by the Bureau of Labor Statistics on the first Friday of each month at 08:30 ET.

### **Surprise**
The difference between an actual macro release and the consensus forecast that preceded it. Markets typically respond to the surprise, not to the level. The project uses a free-data proxy (`actual MoM − 6-month rolling average MoM`) because consensus data requires a Bloomberg / Refinitiv subscription.

### **Litterman & Scheinkman (1991)**
The seminal paper that established Level / Slope / Curvature as the canonical three-factor model for fixed-income yield-curve risk. Full citation: Litterman, R. and Scheinkman, J. (1991). *"Common Factors Affecting Bond Returns."* The Journal of Fixed Income, 1(1), 54-61.

---

[← Back to README](../README.md)
