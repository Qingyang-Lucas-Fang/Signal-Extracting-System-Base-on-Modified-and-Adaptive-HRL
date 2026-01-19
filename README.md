# Signal-Extracting-System-Base-on-Modified-and-Adaptive-HRL
This is a pragmatic algorithm trading system that utilize a modified HRL to trade crypto. Hope u find this useful!

## Background
This repo is used for crypto trading and need real time Level 1 data to power. While almost of the high performance signal digging algorithms are restricted to hedge funds and need high computation power to run the trading, this relatively poor developed is aimed for utilizing signal extracting driven trading engine on PC and make them accessible to the general public. While HRL may encounter some problems in crypto market's extremeley volatile condition, I modify some math frameworks of it including quality function, etc. 

# Methodology     
## Factor Pool Generation (`apg_exp1.py`)
This is the first part of this system, it generates a diverse set of technical alpha factors through recursive random generation and time-series transformations folowing some basic finance rules and incorporate onluy reasonable arithmetic operations.

### Key Components

#### **Feature Dimensions**
- **PRICE**: Returns, price spreads, body percentages, high-low bias, transformed returns
- **VOLUME**: Taker ratios, net aggression, trade intensity, volume z-scores, ADV

#### **Generation Process**
1. **Base Feature Construction**: Creates foundational features from raw OHLCV data including:
   - Log returns, volume z-scores, price spreads, taker ratios
   - Logarithmic and square root transformations of all base features

2. **Recursive Alpha Generation**: 
   - Combines features using binary operations (+, -, *, /)
   - Ensures dimensional consistency (price + price, volume + volume, etc.)
   - Limits recursion depth to prevent over-complex formulas

3. **Time-Series Transformations**:
   - Applies rolling statistics (mean, std, skew) across multiple windows
   - Calculates time-series deltas
   - Uses windows from Fibonacci sequence (3, 5, 8, 13, 21, 34, 55)

#### **Quality Filtering**
- Filters out signals with insufficient variability
- Requires minimum Information Coefficient (IC > 0.015) against forward returns
- Parallel processing for efficient generation (40,000 target factors)

#### **Output**
- `signal_matrix.npy`: T × N matrix of alpha signals
- `alpha_pool.csv`: Metadata including formulas and IC values

## 2. Signal Classification (`cst_main.py`)

### Purpose
Classifies generated alpha factors into economically meaningful clusters based on their statistical properties.

### Processing Pipeline

#### **Signal Validation**
1. Removes signals with:
   - Insufficient variability (std < 1e-6)
   - Low cardinality (< 10 unique values)
   - Excessive missing data (> 20% NaN)

#### **Feature Extraction**
For each valid signal, computes:
- **Return Correlation**: Relationship with future returns
- **Volatility Correlation**: Sensitivity to market volatility  
- **Spread Correlation**: Relationship with bid-ask spreads
- **Volume Correlation**: Alignment with trading volume
- **Persistence**: Autocorrelation (signal stability)
- **Turnover**: Average absolute daily changes

#### **Cluster Assignment**
Seven economically intuitive clusters:
1. **Trend Following Momentum**: Positive return correlation with high persistence
2. **Volatility Breakout**: High volatility correlation with above-median turnover
3. **Liquidity Imbalance**: Strong volume correlation with neutral return relationship
4. **Spread Compression**: Negative spread correlation
5. **Mean Reversion Microstructure**: Negative return correlation with low persistence
6. **Volume Shock Response**: High volume and volatility correlation
7. **Noise Sensitive Fast Decay**: Low persistence with high turnover

#### **Output**
- `alpha_cluster.csv`: Enhanced alpha pool with cluster assignments
- Cluster distribution statistics and quality metrics
- Separation of qualified vs. discarded factors

### Key Features
- **Economic Intuition**: Clusters reflect real trading strategies
- **Robust Filtering**: Multiple validation checks ensure signal quality
- **Interpretable Output**: Clear mapping between statistical properties and strategy types
- **Scalable Design**: Handles thousands of generated factors efficiently
