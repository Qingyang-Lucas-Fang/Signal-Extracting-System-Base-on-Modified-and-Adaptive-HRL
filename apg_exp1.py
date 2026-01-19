import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
import functools

warnings.filterwarnings('ignore')


DIMENSIONS = {
    'PRICE': ['ret', 'price_spread', 'body_pct', 'hi_lo_bias', 'sqrt_ret', 'log_ret'],
    'VOLUME': ['t_base_ratio', 't_quote_ratio', 'net_aggression', 'trade_intensity', 'vol_z', 'adv_20'],
}

BINARY_OPS = ['+', '-', '*', '/']
TS_OPS = ['mean', 'std', 'delta', 'skew']
WINDOWS = [3, 5, 8, 13, 21, 34, 55]



def build_mega_base(df):
    eps = 1e-9
    base = {
        'ret': np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values,
        'vol_z': ((df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps)).fillna(
            0).values,
        'price_spread': ((df['High'] - df['Low']) / (df['Close'] + eps)).fillna(0).values,
        't_base_ratio': (df['Taker buy base asset volume'] / (df['Volume'] + eps)).fillna(0).values,
        'net_aggression': ((2 * df['Taker buy base asset volume'] - df['Volume']) / (df['Volume'] + eps)).fillna(
            0).values,
    }

    expanded = base.copy()
    for k in list(base.keys()):
        expanded[f"log_{k}"] = np.log(np.abs(base[k]) + eps)
        expanded[f"sqrt_{k}"] = np.sign(base[k]) * np.sqrt(np.abs(base[k]))
    return expanded


def get_dim(feat_name):
    if any(p in feat_name for p in DIMENSIONS['PRICE']): return 'PRICE'
    if any(v in feat_name for v in DIMENSIONS['VOLUME']): return 'VOLUME'
    return 'MIXED'


def generate_recursive_alpha(base_data, cols, depth=0, max_depth=2):
    if depth >= max_depth or np.random.rand() > 0.6:
        fname = np.random.choice(cols)
        return base_data[fname], fname, get_dim(fname)

    op = np.random.choice(BINARY_OPS)
    l_val, l_name, l_dim = generate_recursive_alpha(base_data, cols, depth + 1, max_depth)
    r_val, r_name, r_dim = generate_recursive_alpha(base_data, cols, depth + 1, max_depth)

    if op in ['+', '-'] and l_dim != r_dim: op = '*'

    try:
        if op == '+':
            res = l_val + r_val
        elif op == '-':
            res = l_val - r_val
        elif op == '*':
            res = l_val * r_val
        else:
            res = l_val / (r_val + 1e-9)
        return np.nan_to_num(res), f"({l_name}{op}{r_name})", (l_dim if op in ['+', '-'] else 'MIXED')
    except:
        return l_val, l_name, l_dim


def mine_batch(batch_id, target_size, base_data, cols, future_ret, T):
    batch_results = []
    batch_signals = []

    attempts = 0

    while len(batch_results) < target_size:
        attempts += 1

        raw_val, core_name, _ = generate_recursive_alpha(base_data, cols)

        ts_op = np.random.choice(TS_OPS)
        w = np.random.choice(WINDOWS)
        s = pd.Series(raw_val)

        if ts_op == 'mean':
            signal = s.rolling(w).mean()
        elif ts_op == 'std':
            signal = s.rolling(w).std()
        elif ts_op == 'delta':
            signal = s.diff(w)
        else:
            signal = s.rolling(w).skew()

        signal = np.nan_to_num(signal.values)

        if np.std(signal) < 1e-6:
            continue

        ic = np.corrcoef(signal[100:-100], future_ret[100:-100])[0, 1]
        if abs(ic) < 0.015:
            continue

        batch_results.append({
            "formula": f"TS_{ts_op}_{w}{core_name}",
            "ic": ic
        })

        batch_signals.append(signal[:T])

        if attempts > 50000:
            break

    return batch_results, np.array(batch_signals)



if __name__ == "__main__":
    df = pd.read_csv("original_data.csv")
    base_data = build_mega_base(df)
    cols = list(base_data.keys())
    future_ret = np.log(df['Close']).diff().shift(-1).fillna(0).values

    TOTAL_ALPHAS = 40000
    NUM_CORES = 8
    BATCH_SIZE = 20
    T = len(df)

    all_meta = []
    all_signals = []

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [
            executor.submit(
                mine_batch,
                i,
                BATCH_SIZE,
                base_data,
                cols,
                future_ret,
                T
            )
            for i in range(TOTAL_ALPHAS // BATCH_SIZE)
        ]

        for f in tqdm(futures):
            meta, signals = f.result()
            if len(meta) == 0:
                continue

            all_meta.extend(meta)
            all_signals.append(signals)

    signal_matrix = np.vstack(all_signals).T  # [T, N]
    np.save("signal_matrix.npy", signal_matrix)

    pd.DataFrame(all_meta).to_csv("alpha_pool.csv", index=False)
