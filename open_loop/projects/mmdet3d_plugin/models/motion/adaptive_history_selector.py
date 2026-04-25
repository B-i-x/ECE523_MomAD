import torch
import torch.nn as nn


class AdaptiveHistorySelector(nn.Module):
    """3-way soft gate over {no-history, t-1, t-1+t-2} for the planning query."""

    def __init__(self, ego_status_dim=10, embed_dims=256, hidden_dim=64,
                 num_branches=3, cmd_dim=3):
        super().__init__()
        self.cmd_dim = cmd_dim
        in_dim = 2 * ego_status_dim + embed_dims + cmd_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_branches),
        )

    def forward(self, ego_status, last_ego_status, plan_query, ego_cmd):
        # plan_query: [B, 1, 18, 256] -> mean over (mode, group) -> [B, 256]
        pooled = plan_query.mean(dim=(1, 2))
        if ego_cmd.dim() > 2:
            ego_cmd = ego_cmd.view(ego_cmd.shape[0], -1)
        ego_cmd = ego_cmd.to(dtype=pooled.dtype)
        x = torch.cat([ego_status, last_ego_status, pooled, ego_cmd], dim=-1)
        return torch.softmax(self.mlp(x), dim=-1)
