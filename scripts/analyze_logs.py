#!/usr/bin/env python3
"""Aggregate episode-level actual evaluation data. Refuse empty or malformed inputs."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();source=Path(args.input)
    files=sorted(source.rglob('*.csv')) if source.is_dir() else [source]
    if not files:raise ValueError('No episode CSV files found. Cannot create paper tables from templates.')
    tables=[]
    mandatory={'scenario','method','trial_id','success','fall'}
    for file in files:
        df=pd.read_csv(file)
        if df.empty: raise ValueError(f'Empty episode file: {file}')
        if not mandatory.issubset(df.columns):raise ValueError(f'{file}: required columns {sorted(mandatory)}')
        for c in ['success','fall']:
            if not df[c].isin([0,1]).all():raise ValueError(f'{file}: {c} must be binary')
        if df.duplicated(['scenario','method','trial_id']).any():raise ValueError(f'{file}: duplicated trial identifiers')
        tables.append(df)
    merged=pd.concat(tables,ignore_index=True)
    if merged.duplicated(['scenario','method','trial_id']).any():raise ValueError('Repeated trial IDs across files')
    metrics={'success':'mean','fall':'mean'}
    for col in ['velocity_error','yaw_error','recovery_time_s']:
        if col in merged.columns:metrics[col]='mean'
    summary=merged.groupby(['method','scenario'],as_index=False).agg(**{
        'num_trials':('trial_id','size'), **{k:(k,v) for k,v in metrics.items()}
    })
    summary=summary.rename(columns={'success':'success_rate','fall':'fall_rate'})
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);summary.to_csv(out,index=False)
    print(f'Aggregated {len(merged)} genuine trial rows to {out}')
if __name__=='__main__':main()
