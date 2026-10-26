#!/usr/bin/env python3
"""Entry point for the synthetic smoke trainer; Isaac Lab training is intentionally guarded."""
import argparse
from cape_diffloco.utils.config import load_config
from cape_diffloco.utils.seeding import set_seed
from cape_diffloco.training.trainer import CAPETrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    set_seed(cfg.get('seed',0))
    CAPETrainer(cfg).train()

if __name__=='__main__':main()
