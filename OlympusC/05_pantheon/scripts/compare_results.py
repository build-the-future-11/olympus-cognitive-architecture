#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from pantheon.adjudication import pantheon_adjudicate
p=argparse.ArgumentParser(); p.add_argument('--package',required=True); p.add_argument('--proposer',required=True); p.add_argument('--replicator',required=True); p.add_argument('--out',required=True); a=p.parse_args()
r=pantheon_adjudicate(a.package,a.proposer,a.replicator); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
