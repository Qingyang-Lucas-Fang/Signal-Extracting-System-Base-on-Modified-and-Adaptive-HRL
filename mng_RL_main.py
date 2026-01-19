import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


df = pd.read_csv("original_data.csv")
signal_matrix = np.load("signal_matrix.npy")
alpha_pool = pd.read_csv("alpha_pool.csv")
alpha_cluster = pd.read_csv("alpha_cluster.csv")


qualified_mask = alpha_pool['formula'].isin(alpha_cluster['formula']).values

signal_matrix_clean = signal_matrix[:, qualified_mask]


clusters_names = alpha_cluster['cluster_name'].unique()
cluster_indices = {
    name: np.where(alpha_cluster['cluster_name'] == name)[0]
    for name in clusters_names
}


def get_rl_state(df):
    eps = 1e-9

    ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    vol_z = ((df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + eps)).fillna(0)
    price_spread = ((df['High'] - df['Low']) / (df['Close'] + eps)).fillna(0)
    t_base_ratio = (df['Taker buy base asset volume'] / (df['Volume'] + eps)).fillna(0)
    net_agg = ((2 * df['Taker buy base asset volume'] - df['Volume']) / (df['Volume'] + eps)).fillna(0)


    state = np.column_stack([ret, vol_z, price_spread, t_base_ratio, net_agg])
    return torch.tensor(state, dtype=torch.float32).to(DEVICE)



def get_manager_state(worker_state, window=60):
    df_temp = pd.DataFrame(worker_state.cpu().numpy())
    m_mean = df_temp.rolling(window).mean().fillna(0)
    m_std = df_temp.rolling(window).std().fillna(0)
    return torch.tensor(np.hstack([m_mean, m_std]), dtype=torch.float32).to(DEVICE)


worker_states = get_rl_state(df)
manager_states = get_manager_state(worker_states)



class Manager(nn.Module):
    def __init__(self, in_dim, num_clusters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, num_clusters),
            nn.Softmax(dim=-1)
        )

    def forward(self, x): return self.net(x)


class Worker(nn.Module):
    def __init__(self, in_dim, num_alphas):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, num_alphas),
            nn.Softmax(dim=-1)
        )

    def forward(self, x): return self.net(x)



mgr = Manager(manager_states.shape[1], len(clusters_names)).to(DEVICE)
workers = {name: Worker(worker_states.shape[1], len(idx)).to(DEVICE)
           for name, idx in cluster_indices.items()}


mgr_opt = optim.Adam(mgr.parameters(), lr=1e-4)
wrk_opts = {n: optim.Adam(w.parameters(), lr=1e-4) for n, w in workers.items()}


cut = int(len(df) * 0.8)
BATCH_SIZE = 512

for ep in range(5):
    epoch_pnl = 0
    for t in range(0, cut - BATCH_SIZE - 1, BATCH_SIZE):

        s_m = manager_states[t: t + BATCH_SIZE]
        cluster_weights = mgr(s_m)

        batch_combined_pnl = torch.zeros(BATCH_SIZE).to(DEVICE)

        for i, name in enumerate(clusters_names):
            worker = workers[name]
            idx = cluster_indices[name]

            s_w = worker_states[t: t + BATCH_SIZE]
            alpha_weights = worker(s_w)


            y = torch.tensor(signal_matrix_clean[t + 1: t + BATCH_SIZE + 1, idx], dtype=torch.float32).to(DEVICE)

            worker_pnl = (alpha_weights * y).sum(dim=1)

            wrk_loss = -worker_pnl.mean()
            wrk_opts[name].zero_grad()
            wrk_loss.backward(retain_graph=True)
            wrk_opts[name].step()

            batch_combined_pnl += cluster_weights[:, i] * worker_pnl.detach()

        mgr_loss = -batch_combined_pnl.mean()
        mgr_opt.zero_grad()
        mgr_loss.backward()
        mgr_opt.step()

        epoch_pnl += batch_combined_pnl.sum().item()

    print(f"Epoch {ep} PnL: {epoch_pnl:.4f}")