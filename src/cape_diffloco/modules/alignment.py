"""State alignment: high-fidelity forward values, surrogate backward gradients."""
import torch

def fixed_state_alignment(x_sim_next: torch.Tensor, x_diff_next: torch.Tensor,
                          alpha: float | torch.Tensor = 1.0) -> torch.Tensor:
    if x_sim_next.shape != x_diff_next.shape:
        raise ValueError("Forward and differentiable states must have identical shapes")
    if torch.is_tensor(alpha) and torch.any(alpha < 0):
        raise ValueError("alpha must be nonnegative")
    if not torch.is_tensor(alpha) and alpha < 0:
        raise ValueError("alpha must be nonnegative")
    # The right-hand difference is numerically zero; its derivative is not zero.
    return x_sim_next.detach() + alpha * (x_diff_next - x_diff_next.detach())
