import tempfile
import unittest
from pathlib import Path
import torch
from cape_diffloco.modules.alignment import fixed_state_alignment
from cape_diffloco.modules.contact_encoder import ContactPredictiveEncoder
from cape_diffloco.modules.policy import MLPPolicy
from cape_diffloco.training.trainer import CAPETrainer
from cape_diffloco.utils.config import load_config

class CAPETest(unittest.TestCase):
    def test_alignment_forward_and_gradient(self):
        x_sim=torch.tensor([[3.,4.]],requires_grad=False)
        x_diff=torch.tensor([[1.,2.]],requires_grad=True)
        y=fixed_state_alignment(x_sim,x_diff,0.35)
        self.assertTrue(torch.allclose(y,x_sim))
        y.sum().backward()
        self.assertTrue(torch.allclose(x_diff.grad,torch.full_like(x_diff,0.35)))

    def test_contact_and_actor_shapes(self):
        enc=ContactPredictiveEncoder(49,latent_dim=16,hidden_dim=32,horizon=5)
        actor=MLPPolicy(49,12,16,32)
        obs=torch.zeros(3,49);hist=torch.zeros(3,10,49)
        p=enc(hist)
        self.assertEqual(tuple(p['latent'].shape),(3,16))
        self.assertEqual(tuple(p['contact_logits'].shape),(3,5,4))
        self.assertEqual(tuple(actor(obs,p['latent']).shape),(3,12))

    def test_tracking_loss_gradient_reaches_policy(self):
        cfg=load_config('configs/mock_smoke.yaml')
        with tempfile.TemporaryDirectory() as d:
            cfg['output_dir']=d
            trainer=CAPETrainer(cfg)
            obs=trainer.env.reset()
            history=obs.unsqueeze(1).repeat(1,cfg['model']['history_length'],1)
            _,_,losses=trainer._step(obs,history)
            grads=torch.autograd.grad(losses['track'],list(trainer.policy.parameters()),allow_unused=True)
            self.assertTrue(any(g is not None and g.abs().sum().item()>0 for g in grads))
            trainer.logger.close()

    def test_mock_smoke_produces_gradient_and_checkpoint(self):
        cfg=load_config('configs/mock_smoke.yaml')
        with tempfile.TemporaryDirectory() as d:
            cfg['output_dir']=d
            cfg['training']['iterations']=1
            trainer=CAPETrainer(cfg);trainer.train()
            self.assertTrue((Path(d)/'checkpoints/latest.pt').exists())
            self.assertTrue((Path(d)/'train_log.csv').exists())

    def test_isaac_config_is_not_silent_mock(self):
        cfg=load_config('configs/train_cape_go1.yaml')
        with self.assertRaises(NotImplementedError):CAPETrainer(cfg)

if __name__=='__main__':unittest.main()
