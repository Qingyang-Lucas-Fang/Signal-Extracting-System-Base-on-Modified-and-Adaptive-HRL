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

- **A = {α₁, α₂, …, αₙ}**  
  A large pool of candidate alpha signals generated and filtered upstream

- **T**  
  The current decision timestamp

The objective is to construct a portfolio signal:

p_T = Σ (i = 1 to N) [ w_i,T · α_i ]


subject to the constraints:

- **Exposure constraint**
Σ |w_i,T| ≤ C


- **Risk constraint**


Risk(p_T) ≤ ρ


where:
- `w_i,T` is the weight assigned to alpha `α_i` at time `T`
- `C` is the total leverage / exposure budget
- `ρ` is the maximum allowed portfolio risk

### Core Question

Should the model:

- Learn **raw actions** and normalize them into weights?  
or  
- Learn **which alphas are valid** and then **assign weights conditionally**?

This system adopts the **second approach**, which is mathematically correct and structurally stable.

---

## 3.2 Two Competing Learning Formulations

### 3.2.1 Action Learning (Inferior Formulation)

The model learns a raw action vector:



a_T ∈ R^N


and normalizes it:



w_T = a_T / ||a_T||


#### Structural Flaws

- **Implicit feasibility assumption**  
  All alphas are assumed to be conditionally valid at all times.

- **Normalization destroys signal geometry**  
  Relative magnitudes after normalization do not correspond to optimal expected returns.

- **Unidentifiable credit assignment**  
  Poor performance can come from:
  - bad alpha
  - bad scaling
  - noise  
  These are indistinguishable.

- **Severe instability under local learning**  
  Small data changes cause large portfolio swings.

This approach forces learning in a space where **most dimensions should be zero**, but the model is never told that explicitly.

---

### 3.2.2 Alpha Selection + Weighting (Preferred Formulation)

Introduce a **latent relevance indicator**:



z_i,T ∈ {0, 1}


where:
- `z_i,T = 1` → alpha `α_i` is locally valid at time `T`
- `z_i,T = 0` → alpha `α_i` is ignored

The portfolio becomes:



p_T = Σ (i = 1 to N) [ z_i,T · w_i,T · α_i ]


This decomposes the problem into two independent questions:

1. **Which alphas apply now?**
2. **How much weight should they receive?**

This is a **fundamental decomposition**, not a modeling trick.

---

## 3.3 State Representation and No Look-Ahead

At decision time `T`, the system constructs a state:



s_T = ( x_{T-1}, u_T )


where:

- `x_{T-1}`  
  Market features available **up to time T−1**  
  (returns, volatility, spreads, microstructure features, etc.)

- `u_T`  
  Contemporaneous execution-related information at time `T`  
  (current position, volume, liquidity constraints)

### Why This Matters

- Prevents **look-ahead bias**
- Matches real execution conditions
- Allows conditioning on current exposure without leaking future prices

---

## 3.4 Local Regime Conditioning via k-NN

Instead of learning from the entire historical dataset, the system performs **case-based inference**.

### Distance-Based Regime Matching

Define a distance function:



D(s_T, s_t)


This measures similarity between the current state and a past state.

The local neighborhood is:



N_k(s_T) = k nearest past states to s_T


All learning and estimation are performed **only on this neighborhood**.

### Benefits

- Regime adaptivity
- Robustness to non-stationarity
- Fast response to microstructure changes
- Natural regularization through locality

---

## 3.5 Local Dataset Construction

From the neighborhood `N_k(s_T)`, construct a local dataset:



D_T = { (s_tj, r_tj,i) }


where:
- `s_tj` is a past state similar to `s_T`
- `r_tj,i` is the realized return of alpha `α_i` at time `t_j`

This dataset is **reconstructed at every decision point**.

---

## 3.6 Alpha Relevance as a Conditional Random Variable

For each alpha `α_i`, define a **local conditional return distribution**:



R_i | s_T


Define relevance using a statistical test:



z_i,T = 1 if:
E[R_i | s_T] > τ_i
AND
Var(R_i | s_T) < σ_i²


where:
- `τ_i` is a minimum expected return threshold
- `σ_i²` is a maximum acceptable variance

This is **conditional hypothesis testing**, not heuristics.

---

## 3.7 Learning Alpha Relevance (ML / DL)

Instead of hard thresholds, relevance is modeled probabilistically.

### Relevance Probability



π_i(s_T) = P(z_i,T = 1 | s_T)


Modeled as:



π_i(s_T) = sigmoid( f_θ( s_T , φ_i ) )


where:
- `φ_i` = alpha metadata (cluster ID, turnover, horizon, etc.)
- `f_θ` = shared model (logistic regression, GBDT, or neural network)

This is a **multi-task conditional classification problem**.

### Local Weighted Loss

Training is local:



L_select =

Σ (j = 1 to k) Σ (i = 1 to N)
w_j · [ z_i,tj log π_i(s_tj)
+ (1 - z_i,tj) log(1 - π_i(s_tj)) ]


This is **case-based learning**, not global training.

---

## 3.8 Conditional Alpha Weighting

For alphas with `z_i,T = 1`, solve a local mean–variance problem:



maximize: μ̂_i(s_T) · w - λ · σ̂_i²(s_T) · w²


Local estimates:



μ̂_i(s_T) = Σ w_j · r_tj,i
σ̂_i²(s_T) = Σ w_j · (r_tj,i - μ̂_i)²


This yields stable, interpretable weights.

---

## 3.9 Joint Selection–Allocation Optimization

The full problem is:



maximize over {w, z}:
Σ z_i · μ̂_i · w_i - λ Σ z_i² · w_i² · σ̂_i²

subject to:
Σ |z_i · w_i| ≤ C


This is a **locally adaptive sparse quadratic program**.

---

## 3.10 Hierarchical Manager–Worker Structure

### Workers (Per Cluster)

For cluster `c`:



A_c = { α_i : i ∈ c }


Each worker:
- Estimates alpha relevance probabilities
- Computes local mean–variance statistics
- Operates at microstructure timescales

### Manager (Across Clusters)

The manager allocates capital across clusters:



maximize: Σ β_c · μ_c
subject to: βᵀ Σ β ≤ ρ


### Why the Hierarchy Works

- Reduces dimensionality
- Reduces variance
- Prevents representation failure
- Improves regime stability

---

## 3.11 Where Deep Learning Actually Helps

### Representation Learning
Learn latent states:


z_t = f_φ(x_{t-1})


Distance and relevance operate in latent space.

### Metric Learning
Learn:


D_θ(s_T, s_t)


so that:
- Similar regimes → similar alpha performance
- Dissimilar regimes → poor transfer

### Meta-Learning (Optional)
Learn how to:
- Choose `k`
- Adjust distance kernels
- Adapt thresholds dynamically

---

## 3.12 Why This Dominates Action Learning

| Aspect | Learn Actions | Learn Alphas |
|------|--------------|--------------|
| Sparsity | Emergent (unstable) | Structural |
| Interpretability | None | High |
| Locality | Weak | Native |
| Overfitting | Severe | Controlled |
| Failure Mode | Explodes | Degrades gracefully |

---

**In summary:**

> This system is a locally adaptive, hierarchical, sparse alpha inference engine designed explicitly for non-stationary crypto markets.
