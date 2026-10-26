#!/usr/bin/env python3
"""Export CAPE encoder+actor jointly. Export does NOT validate locomotion performance."""
import argparse
from pathlib import Path
import torch
from torch import nn
from cape_diffloco.modules.contact_encoder import ContactPredictiveEncoder
from cape_diffloco.modules.policy import MLPPolicy

class DeployedCAPE(nn.Module):
    def __init__(self, encoder, actor):
        super().__init__();self.encoder=encoder;self.actor=actor
    def forward(self, current_obs: torch.Tensor, obs_history: torch.Tensor):
        return self.actor(current_obs, self.encoder(obs_history)['latent'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--format',choices=['torchscript','onnx'],default='torchscript')
    p.add_argument('--out',required=True);args=p.parse_args()
    # weights_only restricts deserialization to state dictionaries and primitive types.
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    m=checkpoint['model']
    actor=MLPPolicy(m['obs_dim'],m['action_dim'],m['latent_dim'],m['hidden_dim'])
    encoder=ContactPredictiveEncoder(m['obs_dim'],m['latent_dim'],m['hidden_dim'],m['contact_horizon'])
    actor.load_state_dict(checkpoint['policy']);encoder.load_state_dict(checkpoint['encoder'])
    module=DeployedCAPE(encoder,actor).eval()
    obs=torch.zeros(1,m['obs_dim'])
    history=torch.zeros(1,m['history_length'],m['obs_dim'])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if args.format=='torchscript':
        # Trace with fixed batch and history length; hardware safety validation remains separate.
        torch.jit.trace(module,(obs,history),strict=True).save(str(out))
    else:
        torch.onnx.export(module,(obs,history),str(out),input_names=['current_obs','obs_history'],
                          output_names=['joint_action'],opset_version=17)
    print(f'Exported {args.format} to {out}. Mock checkpoints are NOT walking policies.')

if __name__=='__main__':main()
