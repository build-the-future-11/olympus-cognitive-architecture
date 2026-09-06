from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from .hashing import hash_tree, sha256_file

CANARY = 'PANTHEON-PROPOSER-ONLY-7F3A9C'


def _mutate(work: Path, fault: str) -> None:
    if fault == 'data_mismatch':
        import pandas as pd
        p=work/'data.csv'; df=pd.read_csv(p); df.loc[0,'y'] += 8.0; df.to_csv(p,index=False)
    elif fault == 'implementation_mismatch':
        p=work/'analysis.awk'; txt=p.read_text(); txt=txt.replace('x[n]=$(idx["x"])+0', 'x[n]=($(idx["x"])+0)^2')
        p.write_text(txt)
    elif fault == 'environment_mismatch':
        p=work/'environment.lock'; p.write_text(p.read_text()+"fault_injected_dependency=0.0.1\n")
    elif fault == 'protocol_ambiguous':
        p=work/'protocol.md'; p.write_text('# Protocol\nEstimate a treatment effect. Covariate adjustment is unspecified.\n')
    elif fault == 'shared_bug':
        p=work/'analysis.awk'; txt=p.read_text().replace("effect=b1", "effect=-b1")
        p.write_text(txt)
    elif fault in {'clean','metric_mismatch','hidden_contamination','claim_overstated'}:
        return
    else:
        raise ValueError(f'unknown fault: {fault}')


def run_package(package_dir: str | Path, run_dir: str | Path, role: str, seed: int, fault: str='clean', proposer_canary: bool=False) -> dict:
    package_dir=Path(package_dir); run_dir=Path(run_dir); shutil.rmtree(run_dir, ignore_errors=True); run_dir.mkdir(parents=True)
    start=time.perf_counter(); start_wall=time.time()
    with tempfile.TemporaryDirectory(prefix=f'pantheon-{role}-') as td:
        work=Path(td)/'package'; shutil.copytree(package_dir, work)
        _mutate(work, fault)
        env=os.environ.copy()
        if fault == 'metric_mismatch': env['PANTHEON_NEGATE_METRIC']='1'
        if fault == 'hidden_contamination': env['PANTHEON_CANARY_VISIBLE']=CANARY
        if role == 'proposer' and proposer_canary: (work/'PROPOSER_PRIVATE.txt').write_text(CANARY+'\n')
        output=run_dir/'metrics.json'
        proc=subprocess.run(['awk','-v',f'seed={seed}','-f',str(work/'analysis.awk'),str(work/'data.csv')], env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            output.write_text(proc.stdout)
        (run_dir/'stdout.log').write_text(proc.stdout); (run_dir/'stderr.log').write_text(proc.stderr)
        if proc.returncode != 0:
            status='FAILED'; metrics={}
        else:
            status='SUCCEEDED'; metrics=json.loads(output.read_text())
        observed=hash_tree(work, ['data.csv','analysis.awk','claim.yaml','protocol.md','stopping_rule.md','claim_boundary.md','environment.lock','expected_schema.json'])
        canonical=json.loads((package_dir/'package_manifest.json').read_text())['files']
        provenance={
          'role':role,'seed':seed,'fault':fault,'status':status,'returncode':proc.returncode,
          'start_unix':start_wall,'runtime_sec':time.perf_counter()-start,
          'workspace_kind':'fresh_tempdir','pid':os.getpid(),'controller_python':sys.version.split()[0],'executor':'awk',
          'package_path':str(package_dir.resolve()),'canonical_hashes':canonical,'observed_hashes':observed,
          'data_hash':sha256_file(work/'data.csv'),'code_hash':sha256_file(work/'analysis.awk'),
          'protocol_hash':sha256_file(work/'protocol.md'),'environment_hash':sha256_file(work/'environment.lock'),
          'metric_mode':'negated' if fault=='metric_mismatch' else 'canonical',
          'canary_observed':metrics.get('observed_canary') if metrics else None,
          'independence_vector':{'context':1,'seed':0,'environment':0,'runtime':1,'model':0},
        }
        (run_dir/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True))
        (run_dir/'observed_hashes.json').write_text(json.dumps(observed,indent=2,sort_keys=True))
    return {'metrics':metrics,'provenance':provenance}
