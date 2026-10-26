#!/usr/bin/env python3
"""No fake hardware trial rows. Implement safe Go1 read-only telemetry integration first."""
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--out',required=True);p.parse_args()
    raise NotImplementedError('Go1 telemetry collection is not yet connected. No real robot trial data were generated.')
if __name__=='__main__':main()
