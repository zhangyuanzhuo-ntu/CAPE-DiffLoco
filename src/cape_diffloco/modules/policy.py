import torch
from torch import nn


class MLPPolicy(nn.Module):
    """Proprioceptive actor. Input can be obs or obs concatenated with contact latent."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 0, hidden_dim: int = 256):
        super().__init__()
        in_dim = obs_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is not None:
            obs = torch.cat([obs, latent], dim=-1)
        return self.net(obs)
