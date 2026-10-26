#!/usr/bin/env python3
from pathlib import Path
import argparse
from cape_diffloco.utils.config import load_config
from cape_diffloco.utils.seeding import set_seed
from cape_diffloco.training.trainer import CAPETrainer

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/mock_smoke.yaml')
    cfg=load_config(p.parse_args().config)
    if cfg.get('simulator')!='mock':
        raise ValueError('Smoke test is restricted to the mock environment')
    set_seed(cfg.get('seed',0));CAPETrainer(cfg).train()
    print('Synthetic smoke test complete (NOT a locomotion benchmark): '+str(Path(cfg['output_dir'])/'train_log.csv'))

if __name__=='__main__':main()
