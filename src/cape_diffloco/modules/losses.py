"""Differentiable synthetic diagnostics; true contact/clearance objectives need real simulator labels."""
import torch
import torch.nn.functional as F

def total_cape_loss(batch: dict, weights: dict) -> dict:
    losses = {}
    losses['track'] = ((batch['aligned_velocity_xy'] - batch['command_xy'])**2).sum(-1).mean()
    losses['posture'] = (batch['projected_gravity_xy']**2).sum(-1).mean()
    losses['energy'] = (batch['action']**2).mean()
    losses['support'] = F.softplus(10.0 * (0.25 - batch['support'])).mean()
    losses['slip'] = F.softplus(10.0 * (batch['slip_risk'] - 0.4)).mean()
    losses['contact'] = F.binary_cross_entropy_with_logits(batch['contact_logits'], batch['contact_labels'])
    losses['total'] = sum(weights.get('w_' + k, 1.0) * val for k, val in losses.items())
    return losses
