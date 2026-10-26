"""CAPE: Contact-Aware Predictive Embedding from proprioceptive history."""
import torch
from torch import nn

class ContactPredictiveEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 16, hidden_dim: int = 256, horizon: int = 5):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.latent_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, latent_dim))
        self.contact_head = nn.Linear(hidden_dim, horizon * 4)
        self.support_head = nn.Linear(hidden_dim, horizon * 4)
        self.slip_head = nn.Linear(hidden_dim, horizon * 4)

    def forward(self, obs_hist: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs_hist.ndim != 3:
            raise ValueError("Expected [batch, history, obs_dim]")
        _, h = self.gru(obs_hist)
        h = h[-1]
        return {
            'latent': self.latent_head(h),
            'contact_logits': self.contact_head(h).reshape(-1, self.horizon, 4),
            'support': torch.sigmoid(self.support_head(h).reshape(-1, self.horizon, 4)),
            'slip_risk': torch.sigmoid(self.slip_head(h).reshape(-1, self.horizon, 4)),
        }
