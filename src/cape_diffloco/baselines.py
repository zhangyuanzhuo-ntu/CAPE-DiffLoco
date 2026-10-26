import torch
from torch import nn

class HistoryEncoder(nn.Module):
    def __init__(self,obs_dim,latent_dim=16):
        super().__init__()
        self.gru=nn.GRU(obs_dim,64,batch_first=True)
        self.proj=nn.Linear(64,latent_dim)
    def forward(self,h):
        _,last=self.gru(h)
        return self.proj(last[-1])

class ContactPredictor(HistoryEncoder):
    def __init__(self,obs_dim,horizon=5,latent_dim=16):
        super().__init__(obs_dim,latent_dim)
        self.contact_head=nn.Linear(latent_dim,horizon*4)
        self.horizon=horizon
    def forward(self,h):
        z=super().forward(h)
        return z,self.contact_head(z).reshape(-1,self.horizon,4)

class Actor(nn.Module):
    def __init__(self,obs_dim,latent_dim=0,action_dim=12):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim+latent_dim,256),nn.ELU(),
                               nn.Linear(256,256),nn.ELU(),nn.Linear(256,action_dim),nn.Tanh())
    def forward(self,obs,latent=None):
        return self.net(obs if latent is None else torch.cat([obs,latent],-1))

class ValueNet(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,1))
    def forward(self,obs): return self.net(obs).squeeze(-1)

# PPO requires stochastic policy distribution, GAE, clipped objective, rollout
# storage, and an actual Isaac environment. This file defines network modules,
# NOT a complete PPO implementation.
# History-only: Actor + HistoryEncoder, no contact supervision.
# Reactive contact: Actor with current contact estimate as extra observation.
# DiffSim: Actor + fixed-alpha aligned SRB dynamics.
# CAPE: Actor + ContactPredictor + contact-aware differentiable losses.
