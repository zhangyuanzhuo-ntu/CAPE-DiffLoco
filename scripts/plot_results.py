#!/usr/bin/env python3
"""Plot genuine logged smoke-test losses, not fabricated locomotion metrics."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--logdir',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();log=Path(args.logdir)/'train_log.csv'
    df=pd.read_csv(log)
    if df.empty:raise ValueError(f'Empty log: {log}')
    fig,ax=plt.subplots(figsize=(6,3.4))
    ax.plot(df.iteration,df.loss_total,label='synthetic mock loss')
    ax.set_xlabel('iteration');ax.set_ylabel('loss');ax.set_title('Synthetic smoke test — not paper results')
    ax.legend();fig.tight_layout()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=200)
    print(f'Saved to {out}')
if __name__=='__main__':main()
