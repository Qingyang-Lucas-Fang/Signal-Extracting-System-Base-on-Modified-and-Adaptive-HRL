import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class HRL_Manager(nn.Module):


    def __init__(self, state_dim, num_clusters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_clusters),
            nn.Softmax(dim=-1)  # Total weight sum to 1
        )

    def forward(self, s_m):
        return self.network(s_m)


class HRL_Worker(nn.Module):

    def __init__(self, state_dim, num_alphas):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_alphas),
            nn.Softmax(dim=-1)  # Internal cluster weight sum to 1
        )

    def forward(self, s_w):
        return self.network(s_w)


def train_one_epoch(mgr, workers, data_loader, optimizer_mgr, optimizers_wrk):
    mgr.train()
    for w in workers.values(): w.train()

    epoch_metrics = []

    for batch in data_loader:
        s_m, s_w, alphas, target_ret = batch

        cluster_weights = mgr(s_m)  # [B, K]

        combined_cluster_signals = []
        for i, (name, worker) in enumerate(workers.items()):
            alpha_weights = worker(s_w)
            cluster_sig = torch.sum(alpha_weights * alphas[name], dim=1)
            combined_cluster_signals.append(cluster_sig)

        combined_cluster_signals = torch.stack(combined_cluster_signals, dim=1)  # [B, K]
        final_signal = torch.sum(cluster_weights * combined_cluster_signals, dim=1)

        reward = final_signal * target_ret
        loss = -torch.mean(reward)
        optimizer_mgr.zero_grad()
        for opt in optimizers_wrk.values(): opt.zero_grad()

        loss.backward()

        optimizer_mgr.step()
        for opt in optimizers_wrk.values(): opt.step()

        epoch_metrics.append(calculate_indices(target_ret, final_signal))

    return epoch_metrics