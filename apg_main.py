import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def build_mega_base(df):
    eps = 1e-9
    # Stage 1: Diversity - Using all Taker/Quote/Volume metrics
    base = {
        'ret': np.log(df['Close'] / df['Close'].shift(1)),
        't_base_ratio': df['Taker buy base asset volume'] / (df['Volume'] + eps),
        't_quote_ratio': df['Taker buy quote asset volume'] / (df['Quote asset volume'] + eps),
        'net_aggression': (2 * df['Taker buy base asset volume'] - df['Volume']) / (df['Volume'] + eps),
        'trade_intensity': df['Number of trades'] / (df['Volume'] + eps),
        'price_spread': (df['High'] - df['Low']) / df['Close'],
        'quote_pressure': df['Quote asset volume'] / (df['Volume'] * df['Close'] + eps),
        'vol_z': (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps),
        'body_pct': (df['Close'] - df['Open']) / (df['High'] - df['Low'] + eps)
    }

    # Apply Non-linear expansions
    expanded = {}
    for k, v in base.items():
        expanded[k] = v.fillna(0).values
        expanded[f"log_{k}"] = np.log(np.abs(v) + eps).fillna(0).values
        expanded[f"sqrt_{k}"] = (np.sign(v) * np.sqrt(np.abs(v))).fillna(0).values
    return expanded

def generate_and_screen(df, target_pool_size=80000, attempt_limit=500000):
    base_data = build_mega_base(df)
    cols = list(base_data.keys())

    # Target: Future returns (t+1) for screening
    future_ret = np.log(df['Close']).diff().shift(-1).fillna(0).values

    # Storage for "Qualified" alphas
    qualified_signals = []
    qualified_meta = []

    op_map = {np.add: '+', np.subtract: '-', np.multiply: '*', np.divide: '/'}
    ts_ops = ['mean', 'std', 'delta']
    windows = [5, 10, 15, 20, 30, 40, 60]

    print(f"Mining for {target_pool_size} Qualified Alphas (Max attempts: {attempt_limit})...")
    pbar = tqdm(total=target_pool_size)

    attempts = 0
    while len(qualified_signals) < target_pool_size and attempts < attempt_limit:
        attempts += 1
        try:
            # 1. Randomly compose 5 factors (Dimension 5 Alpha)
            subset = np.random.choice(cols, 4, replace=False)
            o1, o2, o3 = np.random.choice(list(op_map.keys()), 3)

            # (A op1 B) op3 (C op2 D)
            raw = o3(o1(base_data[subset[0]], base_data[subset[1]]),
                     o2(base_data[subset[2]], base_data[subset[3]]))

            # 2. Apply Time-Series operator
            ts_op = np.random.choice(ts_ops)
            w = np.random.choice(windows)
            s = pd.Series(raw)
            if ts_op == 'mean': signal = s.rolling(w).mean()
            elif ts_op == 'std': signal = s.rolling(w).std()
            else: signal = s.diff(w)

            signal = signal.fillna(0).values

            # 3. FAST SCREENING (The "Test" Phase)
            # A: Check for information (not a flat line)
            if np.std(signal) < 1e-8: continue

            # B: Check for predictive power (IC > 0.01)
            ic = np.corrcoef(signal, future_ret)[0, 1]
            if abs(ic) < 0.01: continue

            # C: Check for Diversity (Correlation with the last winner < 0.7)
            if qualified_signals:
                corr = abs(np.corrcoef(signal, qualified_signals[-1])[0, 1])
                if corr > 0.7: continue

            # SUCCESS: Keep the Alpha
            qualified_signals.append(signal.astype(np.float32)) # Float32 saves 50% RAM
            formula = f"TS_{ts_op}_{w}(({subset[0]}{op_map[o1]}{subset[1]}){op_map[o3]}({subset[2]}{op_map[o2]}{subset[3]}))"
            qualified_meta.append({'formula': formula, 'ic': ic})
            pbar.update(1)

        except:
            continue

    pbar.close()
    return pd.DataFrame(qualified_meta), np.array(qualified_signals).T

df=pd.read_csv('original_data.csv')
alpha_meta, signal_matrix = generate_and_screen(df)
# np.save("signal_matrix.npy", signal_matrix)


pool_df = alpha_meta.dropna(subset=['ic'])

# Sort and select top alphas
qualified_alphas = pool_df.sort_values('ic', key=abs, ascending=False).head(4000)

# Get indices of selected alphas
selected_idx = qualified_alphas.index.values

# Align signal matrix
signal_matrix_selected = signal_matrix[:, selected_idx]

print(f"Qualified Alpha Library Size: {len(qualified_alphas)}")

# Save outputs (ALIGNED)
qualified_alphas.to_csv("alpha_pool.csv", index=False)
np.save("signal_matrix.npy", signal_matrix_selected)
