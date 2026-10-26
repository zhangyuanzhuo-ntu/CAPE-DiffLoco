import torch
from torch import nn

class PDForceMapper(nn.Module):
    """Joint-space PD torques mapped to estimated contact forces.
    J: [B,4,3,3], joint vectors [B,4,3]. Damped least squares
    solves J^T f ~= tau. This is an approximation, not a measured
    contact-force estimator; swing feet must be masked.
    """
    def __init__(self, kp=20., kd=0.5, damping=1e-3):
        super().__init__()
        self.kp,self.kd,self.damping=kp,kd,damping

    def forward(self,q_target,q,dq,jacobian,stance):
        tau=self.kp*(q_target-q)-self.kd*dq
        jt=jacobian.transpose(-1,-2)
        eye=torch.eye(3,device=q.device,dtype=q.dtype).view(1,1,3,3)
        # Minimize ||J^T f - tau||^2 + lambda ||f||^2
        lhs=jacobian@jt+self.damping*eye
        rhs=(jacobian@tau.unsqueeze(-1))
        forces=torch.linalg.solve(lhs,rhs).squeeze(-1)
        return tau,forces*stance.unsqueeze(-1)

class SRBStep(nn.Module):
    """Simplified differentiable translational + rotational SRB.
    Body-frame contact moment arms; frame transformations omitted.
    Use only as a CPU gradient reference until frame-consistent
    full simulator integration is validated.
    """
    def __init__(self,dt=0.01,mass=12.,inertia=(0.25,0.8,0.9)):
        super().__init__()
        self.dt=dt
        self.mass=mass
        self.register_buffer('inertia',torch.tensor(inertia))
    def forward(self,state,forces,arms):
        # state: [B,12] position, velocity, rotation-vector proxy, angular velocity
        p,v,r,w=state.split(3,-1)
        gravity=state.new_tensor([0,0,-9.81])
        acc=forces.sum(1)/self.mass+gravity
        moment=torch.cross(arms,forces,dim=-1).sum(1)
        I=self.inertia.to(state)
        wdot=(moment-torch.cross(w,I*w,dim=-1))/I
        vn=v+self.dt*acc
        wn=w+self.dt*wdot
        return torch.cat((p+self.dt*vn,vn,r+self.dt*wn,wn),-1)
