import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cape_extension.optimization import state_alignment
from cape_extension.dynamics import PDForceMapper
def test_alignment_gradient():
    x=torch.tensor([[1.,2.]],requires_grad=True)
    y=state_alignment(torch.tensor([[3.,4.]]),x,0.7)
    assert torch.allclose(y,torch.tensor([[3.,4.]]))
    y.sum().backward()
    assert torch.allclose(x.grad,torch.full_like(x,0.7))
def test_pd_mapping_gradients():
    q=torch.zeros(1,4,3)
    target=torch.ones_like(q,requires_grad=True)
    j=torch.eye(3).repeat(1,4,1,1)
    tau,force=PDForceMapper()(target,q,q,j,torch.ones(1,4))
    force.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
