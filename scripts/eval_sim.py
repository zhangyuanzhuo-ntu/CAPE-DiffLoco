#!/usr/bin/env python3
"""Guard against misleading evaluation templates being presented as real results."""
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='outputs/eval_sim.csv')
    p.parse_args()
    raise NotImplementedError('Real Isaac Lab environment rollout, checkpoint loading and metric logging are not implemented. No results were generated.')
if __name__=='__main__':main()
