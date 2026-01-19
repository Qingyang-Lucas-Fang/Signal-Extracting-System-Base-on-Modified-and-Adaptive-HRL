import numpy as np
import pandas as pd


df = pd.read_csv("original_data.csv")
signal_matrix = np.load("signal_matrix.npy")

ret = np.log(df['Close']).diff().fillna(0).values
vol = pd.Series(ret).rolling(20).std().fillna(0).values
spread = ((df['High'] - df['Low']) / (df['Close'] + 1e-9)).values
volume = df['Volume'].values

T, N = signal_matrix.shape


valid_idx = []
for i in range(N):
    a = signal_matrix[:, i]

    if np.std(a) < 1e-6:
        continue
    if len(np.unique(np.round(a, 6))) < 10:
        continue
    if np.isnan(a).mean() > 0.2:
        continue

    valid_idx.append(i)

signal_matrix = signal_matrix[:, valid_idx]
N = signal_matrix.shape[1]


clusters = {
    0: "trend_following_momentum",
    1: "volatility_breakout",
    2: "liquidity_imbalance",
    3: "spread_compression",
    4: "mean_reversion_microstructure",
    5: "volume_shock_response",
    6: "noise_sensitive_fast_decay"
}

cluster_assignments = []
qualified_mask = []


turnovers = [np.mean(np.abs(np.diff(signal_matrix[:, i]))) for i in range(N)]
turnover_median = np.median(turnovers)


for i in range(N):
    a = signal_matrix[:, i]

    corr_ret = np.corrcoef(a, ret)[0, 1]
    corr_vol = np.corrcoef(a, vol)[0, 1]
    corr_spread = np.corrcoef(a, spread)[0, 1]
    corr_volume = np.corrcoef(a, volume)[0, 1]
    persistence = np.corrcoef(a[:-1], a[1:])[0, 1]
    turnover = np.mean(np.abs(np.diff(a)))

    metrics = [corr_ret, corr_vol, corr_spread, corr_volume, persistence, turnover]
    metrics = [0 if np.isnan(x) else x for x in metrics]
    corr_ret, corr_vol, corr_spread, corr_volume, persistence, turnover = metrics

    assigned = False



    if corr_ret >= 0.2 and persistence >= 0.25:
        cid = 0
        assigned = True

    elif corr_vol >= 0.2 and turnover >= turnover_median:
        cid = 1
        assigned = True

    elif corr_spread <= -0.15:
        cid = 3
        assigned = True

    elif corr_ret <= -0.2 and persistence <= 0:
        cid = 4
        assigned = True

    elif corr_volume >= 0.25 and corr_vol >= 0.1:
        cid = 5
        assigned = True

    elif persistence <= 0.1 and turnover >= turnover_median:
        cid = 6
        assigned = True

    elif corr_volume >= 0.2 and abs(corr_ret) <= 0.1:
        cid = 2
        assigned = True

    if assigned:
        cluster_assignments.append(cid)
        qualified_mask.append(True)
    else:
        cluster_assignments.append(-1)
        qualified_mask.append(False)


alpha_pool = pd.read_csv("alpha_pool.csv")
alpha_pool = alpha_pool.iloc[valid_idx].copy()

alpha_pool["cluster_id"] = cluster_assignments
alpha_pool["qualified"] = qualified_mask
alpha_pool["cluster_name"] = alpha_pool["cluster_id"].map(clusters)

qualified_alpha_pool = alpha_pool[alpha_pool["qualified"]].copy()

qualified_alpha_pool.to_csv(
    "alpha_cluster.csv",
    index=False
)


print("Qualified alpha count:", len(qualified_alpha_pool))
print("Discarded alpha count:", (alpha_pool["qualified"] == False).sum())
print("\nCluster distribution:")
print(qualified_alpha_pool["cluster_name"].value_counts())
