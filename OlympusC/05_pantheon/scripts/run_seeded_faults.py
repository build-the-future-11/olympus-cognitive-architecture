#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import pandas as pd
import yaml
from pantheon.package import create_synthetic_package
from pantheon.runner import run_package
from pantheon.adjudication import pantheon_adjudicate, numeric_only_baseline, pairwise_artifact_baseline, FAULT_TO_LABEL

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/seeded_faults.yaml'); p.add_argument('--out-root',default='runs/seeded_faults'); p.add_argument('--results',default='results/seeded_faults_predictions.csv'); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text()); outroot=Path(a.out_root); shutil.rmtree(outroot,ignore_errors=True); outroot.mkdir(parents=True)
rows=[]; executions=[]; exp_counter=0
for seed in cfg['seeds']:
    pkg=Path('packages')/f'synth_{seed:03d}'; create_synthetic_package(pkg,seed)
    clean_prop=outroot/f'seed_{seed:03d}'/'proposer_clean'; r=run_package(pkg,clean_prop,'proposer',seed,'clean',proposer_canary=True)
    executions.append({'run_id':f's{seed:03d}-proposer-clean','seed':seed,'role':'proposer','condition':'clean','status':r['provenance']['status'],'runtime_sec':r['provenance']['runtime_sec'],'run_dir':str(clean_prop)})
    shared_prop=None
    for cond in cfg['conditions']:
        prop=clean_prop
        if cond=='shared_bug':
            shared_prop=outroot/f'seed_{seed:03d}'/'proposer_shared_bug'
            rr=run_package(pkg,shared_prop,'proposer',seed,'shared_bug',proposer_canary=True)
            executions.append({'run_id':f's{seed:03d}-proposer-shared','seed':seed,'role':'proposer','condition':'shared_bug','status':rr['provenance']['status'],'runtime_sec':rr['provenance']['runtime_sec'],'run_dir':str(shared_prop)})
            prop=shared_prop
        rep=outroot/f'seed_{seed:03d}'/f'replicator_{cond}'
        rr=run_package(pkg,rep,'replicator',seed,cond)
        executions.append({'run_id':f's{seed:03d}-rep-{cond}','seed':seed,'role':'replicator','condition':cond,'status':rr['provenance']['status'],'runtime_sec':rr['provenance']['runtime_sec'],'run_dir':str(rep)})
        expected=FAULT_TO_LABEL[cond]
        preds={
          'numeric_only': numeric_only_baseline(prop,rep),
          'pairwise_artifact': pairwise_artifact_baseline(prop,rep),
          'pantheon': pantheon_adjudicate(pkg,prop,rep,expected_fault=cond)['label']
        }
        comp=pantheon_adjudicate(pkg,prop,rep,expected_fault=cond)
        pm=json.loads((prop/'metrics.json').read_text()); rm=json.loads((rep/'metrics.json').read_text())
        for method,pred in preds.items():
            exp_counter += 1
            rows.append({'evaluation_id':f'eval-{exp_counter:04d}','package_id':f'synth-linear-{seed:03d}','seed':seed,'condition':cond,'expected_label':expected,'method':method,'predicted_label':pred,'correct':int(pred==expected),'effect_proposer':pm['effect'],'effect_replicator':rm['effect'],**comp['disagreement'],'proposer_dir':str(prop),'replicator_dir':str(rep)})
Path(a.results).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.results,index=False)
pd.DataFrame(executions).to_csv('results/execution_registry.csv',index=False)
summary=(pd.DataFrame(rows).groupby('method')['correct'].agg(['count','mean']).reset_index().rename(columns={'mean':'localization_accuracy'}))
summary.to_csv('results/baselines.csv',index=False)
# condition-wise accuracy for paper tables
cond=pd.DataFrame(rows).pivot_table(index='condition',columns='method',values='correct',aggfunc='mean').reset_index(); cond.to_csv('results/main_results.csv',index=False)
print(summary.to_string(index=False))
print(f'actual subprocess executions={len(executions)}; classifier evaluations={len(rows)}')
