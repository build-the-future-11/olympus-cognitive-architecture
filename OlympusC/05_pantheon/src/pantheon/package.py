from __future__ import annotations
import json
import platform
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .hashing import hash_tree, sha256_file

ANALYSIS_SCRIPT = r'''BEGIN { FS="," }
NR==1 {
  for (i=1; i<=NF; i++) idx[$i]=i
  next
}
{
  n++
  x[n]=$(idx["x"])+0
  t[n]=$(idx["treatment"])+0
  y[n]=$(idx["y"])+0
  sx+=x[n]; st+=t[n]; sy+=y[n]
}
END {
  mx=sx/n; mt=st/n; my=sy/n
  for (i=1; i<=n; i++) {
    xc=x[i]-mx; tc=t[i]-mt; yc=y[i]-my
    s11+=tc*tc; s22+=xc*xc; s12+=tc*xc
    s1y+=tc*yc; s2y+=xc*yc
  }
  det=s11*s22-s12*s12
  if (det==0) { print "singular design" > "/dev/stderr"; exit 2 }
  b1=(s1y*s22-s2y*s12)/det
  b2=(s2y*s11-s1y*s12)/det
  b0=my-b1*mt-b2*mx
  for (i=1; i<=n; i++) {
    r=y[i]-(b0+b1*t[i]+b2*x[i]); rss+=r*r
  }
  sigma2=rss/(n-3)
  se=sqrt(sigma2*s22/det)
  effect=b1
  if (ENVIRON["PANTHEON_NEGATE_METRIC"]=="1") effect=-effect
  lo=effect-1.96*se; hi=effect+1.96*se
  canary=ENVIRON["PANTHEON_CANARY_VISIBLE"]
  printf "{\n  \"effect\": %.17g,\n  \"se\": %.17g,\n  \"ci95\": [%.17g, %.17g],\n  \"n\": %d,\n  \"seed\": %d,\n  \"metric\": \"adjusted_ols_treatment_effect\"", effect,se,lo,hi,n,seed
  if (canary!="") printf ",\n  \"observed_canary\": \"%s\"", canary
  printf "\n}\n"
}
'''


def create_synthetic_package(out: str | Path, seed: int, n: int = 320, effect: float = 1.25, noise: float = 1.0) -> Path:
    out=Path(out); shutil.rmtree(out, ignore_errors=True); (out/'configs').mkdir(parents=True)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=n)
    treatment=rng.integers(0,2,size=n)
    y=0.8*x + effect*treatment + rng.normal(scale=noise,size=n)
    pd.DataFrame({'x':x,'treatment':treatment,'y':y}).to_csv(out/'data.csv',index=False)
    (out/'analysis.awk').write_text(ANALYSIS_SCRIPT)
    claim={
      'claim_id':f'synth-linear-{seed:03d}',
      'hypothesis':'The treatment has a positive adjusted effect on y.',
      'primary_metric':'adjusted_ols_treatment_effect',
      'expected_direction':'positive',
      'claim_boundary':{'effect_min':0.0},
      'data_kind':'synthetic', 'generator_seed':seed,
      'blinded_replication':True,
    }
    (out/'claim.yaml').write_text(yaml.safe_dump(claim, sort_keys=False))
    (out/'protocol.md').write_text('# Protocol\nEstimate the treatment coefficient by OLS with intercept and covariate `x`. Report coefficient, standard error, and 95% Wald CI. Do not inspect proposer outputs before freezing the replication result.\n')
    (out/'stopping_rule.md').write_text('Single prespecified analysis; no optional stopping.\n')
    (out/'claim_boundary.md').write_text('Supported if the adjusted treatment effect is positive; exact numerical equality is not required outside R0.\n')
    (out/'environment.lock').write_text(f'analysis_runtime=awk\ncontroller_python={sys.version.split()[0]}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\n')
    (out/'expected_schema.json').write_text(json.dumps({'required':['effect','se','ci95','n','seed','metric']},indent=2))
    manifest={'files':hash_tree(out, ['data.csv','analysis.awk','claim.yaml','protocol.md','stopping_rule.md','claim_boundary.md','environment.lock','expected_schema.json'])}
    (out/'data_manifest.json').write_text(json.dumps({'data.csv':{'sha256':sha256_file(out/'data.csv'),'rows':n,'columns':['x','treatment','y'],'synthetic':True}},indent=2))
    (out/'package_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    return out
