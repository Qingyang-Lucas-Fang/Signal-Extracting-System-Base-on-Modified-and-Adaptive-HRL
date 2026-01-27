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
---
This design choice is intentional and mathematically motivated. We assign weights to different groups in different market microstructure, we first utilize a manage-worker logic where a manager will discover the market mstructure and liquidity using my original data, it will decide which groups should be used, the reward should be the PnL across a certain really small percentage of the current price to avoid luck and remember to avoid look-ahead bias, so construct the quality function and guide my through the tech frame and math work that involve in a HRL structure. i think the manager is used to detemine which several clusters are we going to put most weights on (like 90%, while rest of the weight are distributed to other clusters to hedge risk, so that 10% shoule be assigned to clusters that has the lowest correlation with clusters we are going to put big weights on), the task we r going to do next is to train the manager using DL, after that we can train intro-cluster specific alphaes

## 3. Local Adaptive Alpha Inference Framework

This section defines the **core mathematical and algorithmic framework** used to transform a large, static alpha pool into a **locally adaptive trading strategy**.  
The system is designed for **non-stationary crypto markets**, where global learning is fragile and regime shifts are frequent.

The key design decision is **not** to learn raw trading actions directly, but to decompose decision-making into:

1. **Which alphas are relevant under the current market condition**
2. **How to weight only those relevant alphas**

This decomposition is structural and mathematically necessary.

---
## 3.1 Problem Formulation

Let:

- **$A = \{\\alpha_1, \\alpha_2, \\dots, \\alpha_N\}$**  
  A large pool of candidate alpha signals generated and filtered upstream.

- **$T$**  
  The current decision timestamp.

The objective is to construct a portfolio signal:

$$
p_T = \sum_{i=1}^{N} w_{i,T} \\, \\alpha_i
$$

subject to the constraints:

- **Exposure constraint**

<p style="text-align: center;">
$$\sum_{i=1}^{N} |w_{i,T}| \\le C$$
</p>

- **Risk constraint**

<p style="text-align: center;">
$$\\mathrm{Risk}(p_T) \\le \\rho$$
</p>

where:
- $w_{i,T}$ is the weight assigned to alpha $\\alpha_i$ at time $T$
- $C$ is the total leverage / exposure budget
- $\\rho$ is the maximum allowed portfolio risk

### Core Question

Should the model:

- Learn **raw actions** and normalize them into weights?  
or  
- Learn **which alphas are valid** and then **assign weights conditionally**?

This system adopts the **second approach**, which is mathematically well-posed and structurally stable.

---

## 3.2 Two Competing Learning Formulations

### 3.2.1 Action Learning (Inferior Formulation)

The model learns a raw action vector:

$$
a_T \\in \\mathbb{R}^N
$$

and normalizes it:

$$
w_T = \\frac{a_T}{\\|a_T\\|}
$$

#### Structural Flaws

- **Implicit feasibility assumption**  
  All alphas are assumed to be valid at all times.

- **Normalization destroys signal geometry**  
  Relative magnitudes after normalization do not correspond to optimal expected returns.

- **Unidentifiable credit assignment**  
  Poor performance may originate from bad alphas, bad scaling, or noise, which are inseparable.

- **Severe instability under local learning**  
  Small data changes induce large portfolio reallocations.

This formulation forces learning in a dense space where **most dimensions should be zero**, without encoding sparsity explicitly.

---

### 3.2.2 Alpha Selection + Weighting (Preferred Formulation)

Introduce a latent relevance indicator:

$$
z_{i,T} \\in \\{0,1\\}
$$

where:
- $z_{i,T} = 1$ indicates alpha $\\alpha_i$ is locally valid at time $T$
- $z_{i,T} = 0$ indicates alpha $\\alpha_i$ is ignored

The portfolio becomes:

$$
p_T = \sum_{i=1}^{N} z_{i,T} \\, w_{i,T} \\, \\alpha_i
$$

This decomposes the problem into:

1. **Alpha relevance inference**
2. **Conditional weight allocation**

This is a fundamental structural decomposition, not a heuristic.

---

## 3.3 State Representation and No Look-Ahead

At decision time $T$, define the state:

$$
s_T = (x_{T-1}, u_T)
$$

where:

- $x_{T-1}$: market features observable up to time $T-1$  
  (returns, volatility, spreads, microstructure features)

- $u_T$: contemporaneous execution information at time $T$  
  (current position, volume constraints, liquidity)

This construction:
- Eliminates look-ahead bias
- Matches real execution conditions
- Conditions decisions on current exposure without leaking future prices

---

## 3.4 Local Regime Conditioning via k-NN

Instead of global learning, the system performs **case-based inference**.

Define a distance metric:

$$
D(s_T, s_t)
$$

The local neighborhood is:

$$
\\mathcal{N}_k(s_T) = \\text{k nearest past states to } s_T
$$

All estimation and learning are performed **only within this neighborhood**.

### Advantages

- Regime adaptivity
- Robustness to non-stationarity
- Fast response to microstructure shifts
- Implicit regularization via locality

---

## 3.5 Local Dataset Construction

From $\\mathcal{N}_k(s_T)$, construct a local dataset:

$$
\\mathcal{D}_T = \\{(s_{t_j}, r_{t_j,i})\\}
$$

where:
- $s_{t_j}$ is a historical state similar to $s_T$
- $r_{t_j,i}$ is the realized return of alpha $\\alpha_i$ at time $t_j$

This dataset is reconstructed at every decision step.

---

## 3.6 Alpha Relevance as a Conditional Random Variable

For each alpha $\\alpha_i$, define the conditional return distribution:

$$
R_i \\mid s_T
$$

Alpha relevance is defined via:

$$
z_{i,T} = 1 \\quad \\text{if}
$$

$$
\\mathbb{E}[R_i \\mid s_T] > \\tau_i
\\quad \\text{and} \\quad
\\mathrm{Var}(R_i \\mid s_T) < \\sigma_i^2
$$

where:
- $\\tau_i$ is a minimum expected return threshold
- $\\sigma_i^2$ is a maximum acceptable variance

This is **conditional hypothesis testing**, not heuristic filtering.

---

## 3.7 Learning Alpha Relevance (ML / DL)

Replace hard thresholds with probabilistic modeling.

### Relevance Probability

$$
\\pi_i(s_T) = \\mathbb{P}(z_{i,T} = 1 \\mid s_T)
$$

Modeled as:

$$
\\pi_i(s_T) = \\sigma\\big(f_\\theta(s_T, \\phi_i)\\big)
$$

where:
- $\\phi_i$ denotes alpha metadata (cluster, turnover, horizon)
- $f_\\theta$ is a shared model (logistic regression, GBDT, neural network)
- $\\sigma(\\cdot)$ is the sigmoid function

This is a **multi-task conditional classification problem**.

### Locally Weighted Loss

<p style="text-align: center;">
$$
\\mathcal{L}_{\\text{select}} =
\sum_{j=1}^{k} \sum_{i=1}^{N}
w_j \\, \\Big[
z_{i,t_j} \\log \\pi_i(s_{t_j})
+ (1 - z_{i,t_j}) \\log(1 - \\pi_i(s_{t_j}))
\\Big]
$$
</p>

Learning is strictly local.

---

## 3.8 Conditional Alpha Weighting

For alphas with $z_{i,T} = 1$, solve:

$$
\\max_{w_i} \\; \\hat{\\mu}_i(s_T) w_i - \\lambda \\hat{\\sigma}_i^2(s_T) w_i^2
$$

with local estimates:

$$
\\hat{\\mu}_i(s_T) = \sum_j w_j r_{t_j,i}
$$

$$
\\hat{\\sigma}_i^2(s_T) = \sum_j w_j (r_{t_j,i} - \\hat{\\mu}_i)^2
$$

This produces stable and interpretable weights.

---

## 3.9 Joint Selection–Allocation Optimization

The full problem:

<p style="text-align: center;">
$$
\\max_{\\{w_i, z_i\\}}
\sum_i z_i \\hat{\\mu}_i w_i
- \\lambda \sum_i z_i^2 \\hat{\\sigma}_i^2 w_i^2
$$
</p>

subject to:

$$
\sum_i |z_i w_i| \\le C
$$

This is a **locally adaptive sparse quadratic program**.

---

## 3.10 Hierarchical Manager–Worker Structure

### Workers (Per Cluster)

For cluster $c$:

$$
A_c = \\{\\alpha_i : i \\in c\\}
$$

Each worker:
- Estimates relevance probabilities
- Computes local return statistics
- Operates at microstructure timescales

### Manager (Across Clusters)

The manager allocates capital across clusters:

$$
\\max_{\\beta} \\sum_c \\beta_c \\mu_c
\\quad \\text{s.t.} \\quad
\\beta^T \\Sigma \\beta \\le \\rho
$$

---

## 3.11 Where Deep Learning Helps

- **Representation learning**
   
$$
z_t = f_\\phi(x_{t-1})
$$

- **Metric learning**

$$
D_\\theta(s_T, s_t)
$$

- **Meta-learning**
  Adaptive selection of $k$, distance kernels, and thresholds.

---

## 3.12 Why This Dominates Action Learning

| Aspect | Learn Actions | Learn Alphas |
|------|--------------|--------------|
| Sparsity | Emergent, unstable | Structural |
| Interpretability | None | High |
| Locality | Weak | Native |
| Overfitting | Severe | Controlled |
| Failure Mode | Explosive | Graceful |

---

**Summary**

This system is a locally adaptive, hierarchical, sparse alpha inference engine designed explicitly for non-stationary crypto markets.
