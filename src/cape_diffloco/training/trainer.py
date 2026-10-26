"""CPU synthetic smoke-test trainer; not a real Isaac Lab locomotion trainer."""
from pathlib import Path
import torch
from torch import optim
from cape_diffloco.modules.policy import MLPPolicy
from cape_diffloco.modules.contact_encoder import ContactPredictiveEncoder
from cape_diffloco.modules.alignment import fixed_state_alignment
from cape_diffloco.modules.surrogate_dynamics import ContactAwareSRBDynamics
from cape_diffloco.modules.losses import total_cape_loss
from cape_diffloco.envs.mock_env import MockQuadrupedEnv
from cape_diffloco.utils.logger import CSVLogger

class CAPETrainer:
    def __init__(self, cfg: dict):
        if cfg.get('simulator') != 'mock':
            raise NotImplementedError(
                'Isaac Lab trainer is not implemented. Do not treat synthetic smoke tests as real CAPE training. '
                'See docs/ISAAC_LAB_INTEGRATION.md.'
            )
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cpu'))
        m = cfg['model']; t = cfg['training']
        self.env = MockQuadrupedEnv(t['batch_size'], m['obs_dim'], m['state_dim'], m['action_dim'],
                                    contact_horizon=m['contact_horizon'], device=self.device)
        self.policy = MLPPolicy(m['obs_dim'], m['action_dim'], m['latent_dim'], m['hidden_dim']).to(self.device)
        self.encoder = ContactPredictiveEncoder(m['obs_dim'], m['latent_dim'], m['hidden_dim'], m['contact_horizon']).to(self.device)
        self.dynamics = ContactAwareSRBDynamics(dt=0.01).to(self.device)
        self.opt = optim.Adam(list(self.policy.parameters()) + list(self.encoder.parameters()), lr=t['lr'])
        self.output_dir = Path(cfg['output_dir']); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_length = m['history_length']
        self.logger = CSVLogger(self.output_dir/'train_log.csv',
                                ['iteration', 'loss_total', 'loss_track', 'loss_contact', 'grad_norm'])

    def _step(self, obs, hist):
        current_state = self.env.state.clone().detach()
        predicted = self.encoder(hist)
        action = self.policy(obs, predicted['latent'])
        sim = self.env.step(action.detach())
        # IMPORTANT: force proxy below is *synthetic* and NOT the PD/Jacobian contact map.
        # It only tests that action influences differentiable dynamics and locomotion loss.
        b = action.shape[0]
        forces = torch.zeros((b, 4, 3), device=self.device)
        forces[:, :, 2] = 30.0
        forces = forces + 5.0 * action.reshape(b, 4, 3)
        foot_pos = torch.tensor([[.2,.12,-.25],[.2,-.12,-.25],[-.2,.12,-.25],[-.2,-.12,-.25]],
                                device=self.device).unsqueeze(0).expand(b,-1,-1)
        x_diff = self.dynamics(current_state, forces, foot_pos, predicted['support'][:, 0, :])
        x_aligned = fixed_state_alignment(sim['state'], x_diff, self.cfg['training']['alpha'])
        cmd = torch.zeros((b,2), device=self.device); cmd[:,0] = 0.35
        losses = total_cape_loss({
            'aligned_velocity_xy': x_aligned[:,7:9], 'command_xy': cmd,
            'projected_gravity_xy': sim['projected_gravity_xy'], 'action': action,
            'support': predicted['support'], 'slip_risk': predicted['slip_risk'],
            'contact_logits': predicted['contact_logits'], 'contact_labels': sim['contact_labels'],
        }, self.cfg.get('loss', {}))
        nxt = sim['obs']
        hist = torch.cat([hist[:,1:],nxt.unsqueeze(1)],dim=1)
        return nxt,hist,losses

    def train(self):
        obs = self.env.reset()
        hist = obs.unsqueeze(1).repeat(1,self.history_length,1)
        try:
            for it in range(self.cfg['training']['iterations']):
                self.opt.zero_grad()
                losses_list=[]
                track_list=[]
                contact_list=[]
                for _ in range(self.cfg['training']['horizon']):
                    obs,hist,losses = self._step(obs,hist)
                    losses_list.append(losses['total'])
                    track_list.append(losses['track'])
                    contact_list.append(losses['contact'])
                total = torch.stack(losses_list).mean()
                total.backward()
                grad = sum(float(p.grad.norm().item())**2 for p in self.policy.parameters() if p.grad is not None)**0.5
                if grad <= 0:
                    raise RuntimeError('Actor received no gradient; check state alignment and surrogate force path')
                self.opt.step()
                self.logger.log({'iteration':it,'loss_total':float(total.detach()),
                                 'loss_track':float(torch.stack(track_list).mean().detach()),
                                 'loss_contact':float(torch.stack(contact_list).mean().detach()),'grad_norm':grad})
                save_every = self.cfg['training'].get('save_every', 1)
                if (it+1) % save_every == 0 or it+1==self.cfg['training']['iterations']:
                    folder = self.output_dir/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
                    torch.save({'policy': self.policy.state_dict(), 'encoder':self.encoder.state_dict(),
                                'model':self.cfg['model'],'iteration':it+1,'synthetic_mock':True},folder/'latest.pt')
        finally:
            self.logger.close()
