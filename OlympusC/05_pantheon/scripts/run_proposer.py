#!/usr/bin/env python3
import argparse
from pantheon.runner import run_package
p=argparse.ArgumentParser(); p.add_argument('--package',required=True); p.add_argument('--run-dir',required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--fault',default='clean'); a=p.parse_args()
run_package(a.package,a.run_dir,'proposer',a.seed,a.fault,proposer_canary=True)
