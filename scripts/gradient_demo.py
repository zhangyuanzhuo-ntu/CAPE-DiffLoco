import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cape_extension.dynamics import PDForceMapper,SRBStep
from cape_extension.optimization import state_alignment
torch.manual_seed(0)
B=2
q=torch.zeros(B,4,3)
dq=torch.zeros_like(q)
targets=torch.nn.Parameter(torch.randn_like(q)*0.02)
J=torch.eye(3).repeat(B,4,1,1)
stance=torch.ones(B,4)
mapper=PDForceMapper()
dyn=SRBStep()
state=torch.zeros(B,12)
arms=torch.tensor([[.2,.1,-.25],[.2,-.1,-.25],[-.2,.1,-.25],[-.2,-.1,-.25]]).repeat(B,1,1)
opt=torch.optim.Adam([targets],lr=1e-3)
for iteration in range(3):
    opt.zero_grad()
    tau,forces=mapper(targets,q,dq,J,stance)
    diff=dyn(state,forces,arms)
    sim=diff.detach()  # demo only: synthetic forward reference, NOT Isaac Lab
    aligned=state_alignment(sim,diff)
    loss=(aligned[:,3:6]-torch.tensor([0.,0.,0.1])).square().mean()
    loss.backward()
    assert targets.grad is not None and torch.isfinite(targets.grad).all()
    print(f'[SYNTHETIC CPU TEST] iteration={iteration} loss={loss.item():.6f} grad_norm={targets.grad.norm().item():.6f}')
    opt.step()
