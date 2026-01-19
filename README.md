# Signal-Extracting-System-Base-on-Modified-and-Adaptive-HRL

This is a pragmatic algorithmic trading system that utilizes a **modified and adaptive Hierarchical Learning framework** to trade crypto assets.  
The system emphasizes **signal extraction, local regime adaptation, and alpha-level decision making**, rather than fragile end-to-end policy learning.

---

## Background

This repository targets **retail-accessible quantitative trading** using **Level-1 crypto market data** and commodity hardware (PC-class machines).

Most high-performance alpha discovery and execution systems:
- Are restricted to hedge funds
- Require large-scale compute
- Rely on centralized, global learning assumptions

This project instead focuses on:
- **Locally adaptive signal extraction**
- **Explicit alpha generation, filtering, and selection**
- **Case-based learning under non-stationary crypto regimes**

While vanilla HRL often fails in crypto markets due to **extreme volatility, regime shifts, and delayed rewards**, this system modifies the mathematical structure of HRL by:
- Redefining quality and relevance functions
- Replacing global policy learning with **local inference**
- Introducing **alpha-level selection and weighting**

---

# Methodology

# 1. Factor Pool Generation (`apg_exp1.py`)

This module generates a **diverse and expressive alpha pool** via recursive symbolic construction and time-series transformations, while enforcing basic financial and dimensional constraints.

## Key Components

### Feature Dimensions
- **PRICE**: returns, price spreads, candle body ratios, high–low bias, transformed returns  
- **VOLUME**: taker ratios, net aggression, trade intensity, volume z-scores, ADV

### Generation Process

1. **Base Feature Construction**
   - Log returns
   - Volume z-scores
   - Price spreads
   - Taker buy/sell ratios
   - Log and signed square-root transforms for scale stabilization

2. **Recursive Alpha Generation**
   - Binary operators: `+ - * /`
   - Semantic constraints (e.g. price + price, volume × price)
   - Controlled recursion depth to avoid expression explosion

3. **Time-Series Operators**
   - Rolling mean, std, skew
   - Time-series deltas
   - Fibonacci windows: 3, 5, 8, 13, 21, 34, 55

### Quality Filtering
- Minimum variability threshold
- Minimum IC threshold against forward returns (IC > 0.015)
- Parallelized batch mining

### Output
- `signal_matrix.npy`: shape `(T, N)`
- `alpha_pool.csv`: alpha formulas and quality statistics

---

# 2. Signal Classification (`cst_main.py`)

## Purpose
Classifies raw alphas into **economically interpretable clusters**, enabling hierarchical control and specialization. Due to some serious problems that i detected from directly using those gross factors into mathematical combination and possible RL and DP, I must cluster those factors into different clusters according to their performances in market microstructure. 

## Processing Pipeline

### Signal Validation
Removes signals with:
- Near-zero variance
- Low cardinality
- Excessive missing values

### Feature Extraction
For each alpha:
- Return correlation
- Volatility correlation
- Spread correlation
- Volume correlation
- Persistence (autocorrelation)
- Turnover

### Cluster Assignment
Seven clusters reflecting real microstructure behaviors:
1. Trend-following momentum  
2. Volatility breakout  
3. Liquidity imbalance  
4. Spread compression  
5. Mean-reversion microstructure  
6. Volume shock response  
7. Noise-sensitive fast decay  

### Output
- `alpha_cluster.csv`: enriched alpha metadata with cluster labels

---

# 3. Adaptive Local Alpha Selection Framework (Core Contribution)

This system **does not learn global actions or portfolio weights directly**.  
Instead, it learns:

> **Which alphas are locally relevant, and how to weight them under the current market state.**

This design choice is intentional and mathematically motivated.

We assign weights to different groups in different market microstructure, we first utilize a manage-worker logic where a manager will discover the market mstructure and liquidity using my original data, it will decide which groups should be used, the reward should be the PnL across a certain really small percentage of the current price to avoid luck and remember to avoid look-ahead bias, so construct the quality function and guide my through the tech frame and math work that involve in a HRL structure. i think the manager is used to detemine which several clusters are we going to put most weights on (like 90%, while rest of the weight are distributed to other clusters to hedge risk, so that 10% shoule be assigned to clusters that has the lowest correlation with clusters we are going to put big weights on), the task we r going to do next is to train the manager using DL, after that we can train intro-cluster specific alphaes
---

## 3.1 State Representation and No Look-Ahead

At decision time \( T \), the state is defined as:

\[
s_T = (x_{T-1}, u_T)
\]

Where:
- \( x_{T-1} \): market features available up to \( T-1 \)
- \( u_T \): contemporaneous position / volume information

This construction **avoids look-ahead bias** while allowing realistic execution conditioning.

---

## 3.2 Local Regime Matching via k-NN

Instead of learning from the entire history, the system performs **case-based inference**.

Define a distance metric:
\[
D(s_T, s_t)
\]

The local neighborhood:
\[
\mathcal{N}_k(s_T) = \text{k nearest past states}
\]

All learning and inference is performed **only on this local subset**, enabling:
- Regime adaptivity
- Robustness to non-stationarity
- Fast reaction to microstructure changes

---

## 3.3 Alpha Relevance Modeling (Selection First)

For each alpha \( \alpha_i \), define a binary relevance variable:
\[
z_{i,T} \in \{0,1\}
\]

Meaning:
- \( z_{i,T} = 1 \): alpha is locally valid
- \( z_{i,T} = 0 \): alpha is ignored

Relevance is determined conditionally:
\[
\mathbb{E}[R_i \mid s_T] > \tau_i
\quad \text{and} \quad
\text{Var}(R_i \mid s_T) < \sigma_i^2
\]

This reframes alpha selection as **conditional hypothesis testing**, not heuristics.

---

## 3.4 Learning Alpha Relevance (ML / DL)

Alpha relevance probabilities:
\[
pi_i(s_T) = \mathbb{P}(z_{i,T}=1 \mid s_T)
\]

Learned via supervised or weakly-supervised models:
- Logistic regression
- Gradient boosting
- Neural networks (shared backbone + alpha embeddings)

Loss:
\[
\mathcal{L}_{\text{select}}
=
-\sum_{t \in \mathcal{N}_k}
\sum_i
w_t
\big[
z_{i,t} \log \pi_i(s_t)
+
(1-z_{i,t}) \log (1-\pi_i(s_t))
\big]
\]

This allows:
- Sparse alpha activation
- Interpretability
- Stable local learning

---

## 3.5 Alpha Weighting (Conditional Allocation)

For selected alphas:
\[
w_{i,T}
=
\arg\max
\left(
\hat{\mu}_i(s_T) w
-
\lambda \hat{\sigma}_i^2(s_T) w^2
\right)
\]

Where local mean and variance are estimated from \( \mathcal{N}_k(s_T) \).

Final portfolio:
\[
p_T = \sum_i z_{i,T} w_{i,T} \alpha_i
\]

---

# 4. Hierarchical Manager–Worker Structure

### Workers
- Operate **within alpha clusters**
- Perform relevance estimation and weighting
- Capture microstructure-level dynamics

### Manager
- Allocates risk and capital across clusters
- Enforces global constraints
- Stabilizes cross-regime behavior

This hierarchy reduces:
- Dimensionality
- Variance
- Representation failure

---


## Summary

This system is best understood as:

> **A locally adaptive, hierarchical, sparse alpha inference engine designed for non-stationary crypto markets.**

It prioritizes:
- Interpretability
- Robustness
- Practical deployability
---

