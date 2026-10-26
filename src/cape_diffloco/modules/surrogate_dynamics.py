import torch
from torch import nn


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    # Small-angle quaternion integration. q = [w, x, y, z]
    zeros = torch.zeros_like(omega[..., :1])
    omega_quat = torch.cat([zeros, omega], dim=-1)
    dq = 0.5 * quat_mul(q, omega_quat)
    return quat_normalize(q + dt * dq)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


class ContactAwareSRBDynamics(nn.Module):
    """Differentiable single-rigid-body surrogate with contact-conditioned support.

    State layout: [p(3), quat(4), v(3), omega(3)] -> 13D.
    This is a simplified research scaffold, not a complete quadruped physics engine.
    """

    def __init__(self, mass: float = 12.0, inertia_diag=(0.25, 0.8, 0.9), dt: float = 0.02):
        super().__init__()
        self.mass = mass
        self.dt = dt
        self.register_buffer('gravity', torch.tensor([0.0, 0.0, -9.81]))
        self.register_buffer('inertia', torch.tensor(inertia_diag))

    def forward(
        self,
        state: torch.Tensor,
        foot_forces: torch.Tensor,
        foot_positions_body: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        p = state[..., 0:3]
        quat = state[..., 3:7]
        v = state[..., 7:10]
        omega = state[..., 10:13]

        support = support.unsqueeze(-1)  # [B,4,1]
        weighted_forces = support * foot_forces
        total_force = weighted_forces.sum(dim=-2) + self.mass * self.gravity.to(state.device)
        acc = total_force / self.mass

        torque = torch.cross(foot_positions_body, weighted_forces, dim=-1).sum(dim=-2)
        ang_acc = torque / self.inertia.to(state.device)  # diagonal approximation

        v_next = v + self.dt * acc
        p_next = p + self.dt * v_next
        omega_next = omega + self.dt * ang_acc
        quat_next = quat_integrate(quat, omega_next, self.dt)
        return torch.cat([p_next, quat_next, v_next, omega_next], dim=-1)

