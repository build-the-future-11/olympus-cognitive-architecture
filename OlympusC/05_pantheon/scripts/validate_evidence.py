#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

required=['results/seeded_faults_predictions.csv','results/execution_registry.csv','results/experiment_registry.csv','results/baselines.csv','results/main_results.csv','results/public_case_results.csv','results/public_case_summary.json','results/ablations.csv','results/efficiency.csv','results/statistics.json','results/dataset_audit.json','figures/fig1_fault_localization.png','figures/fig2_condition_matrix.png','figures/fig3_public_case.png']
missing=[p for p in required if not Path(p).exists()]
if missing: raise SystemExit(f'missing evidence: {missing}')
pred=pd.read_csv('results/seeded_faults_predictions.csv'); exe=pd.read_csv('results/execution_registry.csv')
n_seeds=pred.seed.nunique(); n_conditions=pred.condition.nunique(); n_methods=pred.method.nunique()
assert len(pred)==n_seeds*n_conditions*n_methods, (len(pred),n_seeds,n_conditions,n_methods)
assert len(exe)==n_seeds*(1+n_conditions+1), (len(exe),n_seeds,n_conditions)
assert (exe.status=='SUCCEEDED').all()
assert pred[['effect_proposer','effect_replicator','d_z','d_rel','sign_agreement','ci_overlap']].notna().all().all()
pub=pd.read_csv('results/public_case_results.csv'); assert len(pub)==20 and pub.isna().sum().sum()==0
reg=pd.read_csv('results/experiment_registry.csv'); assert len(reg)==len(exe)+len(pub), (len(reg),len(exe),len(pub)); assert (reg.status=='SUCCEEDED').all()
summary=json.loads(Path('results/public_case_summary.json').read_text())
assert summary['paired']['direction_agreement_rate'] >= 0 and summary['paired']['direction_agreement_rate'] <= 1
print(json.dumps({'evidence_validation':'PASS','seeded_fault_classifier_evaluations':len(pred),'actual_seeded_fault_executions':len(exe),'public_case_runs':len(pub)},indent=2))
