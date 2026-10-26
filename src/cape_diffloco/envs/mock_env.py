"""Synthetic shape-and-gradient test fixture. No valid locomotion physics or measurements."""
import torch
class MockQuadrupedEnv:
    def __init__(self, batch_size: int, obs_dim: int, state_dim: int, action_dim: int,
                 contact_horizon: int = 5, device='cpu'):
        self.batch_size, self.obs_dim, self.state_dim = batch_size, obs_dim, state_dim
        self.contact_horizon = contact_horizon
        self.device = torch.device(device)
        self.state = torch.zeros(batch_size, state_dim, device=self.device)
        self.state[:, 3] = 1.0

    def reset(self):
        self.state = torch.zeros_like(self.state)
        self.state[:, 3] = 1.0
        return self.get_obs()

    def get_obs(self):
        obs = torch.randn(self.batch_size, self.obs_dim, device=self.device) * 0.03
        n = min(self.obs_dim, self.state_dim)
        obs[:, :n] += self.state[:, :n]
        return obs

    def step(self, action):
        # Detach emulates a non-differentiable forward simulator; not a dynamics model.
        with torch.no_grad():
            delta = torch.zeros_like(self.state)
            delta[:, 7:9] = 0.05 * action.detach()[:, :2]
            self.state = self.state + delta + 0.01 * torch.randn_like(self.state)
            self.state[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=self.device)
            labels = (torch.rand(self.batch_size, self.contact_horizon, 4, device=self.device) > 0.5).float()
            return {'obs': self.get_obs(), 'state': self.state.clone(),
                    'projected_gravity_xy': self.state[:, 10:12].clone(),
                    'contact_labels': labels}
