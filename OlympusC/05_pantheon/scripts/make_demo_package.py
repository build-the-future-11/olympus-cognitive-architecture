#!/usr/bin/env python3
import argparse
from pantheon.package import create_synthetic_package
p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
create_synthetic_package(a.out,a.seed); print(a.out)
