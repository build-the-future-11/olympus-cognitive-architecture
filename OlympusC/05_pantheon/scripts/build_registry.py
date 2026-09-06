#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

exe=pd.read_csv('results/execution_registry.csv')
pub=pd.read_csv('results/public_case_results.csv')
rows=[]
for _,r in exe.iterrows():
    rows.append({
      'experiment_id':r.run_id,'family':'seeded_fault','role':r.role,'condition':r.condition,'seed':int(r.seed),
      'dataset':'synthetic_linear','dataset_split':'full','model_or_method':'canonical_OLS_or_fault_variant',
      'executor':'awk','device':'CPU','runtime_sec':float(r.runtime_sec),'status':r.status,'evidence_path':r.run_dir,
      'controller_python':platform.python_version(),'platform':platform.platform()
    })
for _,r in pub.iterrows():
    rows.append({
      'experiment_id':f"{r.case_id}-{r.role}",'family':'public_case','role':r.role,'condition':'cross_implementation','seed':int(r.split_seed),
      'dataset':'WDBC_sklearn','dataset_split':f"stratified_75_25_seed_{int(r.split_seed)}",'model_or_method':r.implementation,
      'executor':'python','device':'CPU','runtime_sec':float(r.runtime_sec),'status':'SUCCEEDED','evidence_path':f"runs/public_case/{r.case_id}/{r.role}",
      'controller_python':platform.python_version(),'platform':platform.platform()
    })

# Optional external-agent runs are appended when present.
for rf in sorted(Path('runs/external').glob('*/agent_*/agent_result.json')):
    try:
        r=json.loads(rf.read_text())
    except Exception:
        continue
    rows.append({
      'experiment_id':f"{r.get('task_id',rf.parts[-3])}-{r.get('role',rf.parent.name)}",'family':'external_agent',
      'role':r.get('role',rf.parent.name),'condition':'published_reproduction','seed':-1,
      'dataset':r.get('task_id','external'),'dataset_split':'benchmark_task','model_or_method':f"{r.get('provider')}:{r.get('model')}",
      'executor':'tool_using_agent','device':'CPU','runtime_sec':float(r.get('runtime_sec',0.0)),
      'status':'SUCCEEDED' if r.get('report_exists') else 'INCOMPLETE','evidence_path':str(rf.parent),
      'controller_python':platform.python_version(),'platform':platform.platform()
    })

reg=pd.DataFrame(rows); reg.to_csv('results/experiment_registry.csv',index=False)
reg.groupby(['family','role','model_or_method'],dropna=False)['runtime_sec'].agg(['count','mean','std','sum']).reset_index().to_csv('results/efficiency.csv',index=False)

paired=pd.read_csv('results/public_case_paired.csv')
pred=pd.read_csv('results/seeded_faults_predictions.csv')
stats={
  'seeded_fault':{
    'n_comparisons':int(pred[['seed','condition']].drop_duplicates().shape[0]),
    'n_seeds':int(pred.seed.nunique()),
    'n_conditions':int(pred.condition.nunique()),
    'exact_counts_by_method':pred.groupby('method').correct.agg(['sum','count']).reset_index().to_dict(orient='records'),
    'note':'Exact counts on controlled injected faults; no sampling-based significance test is claimed.'
  },
  'public_case':{
    'n_split_seeds':int(len(paired)),
    'mean_abs_effect_difference':float(paired.effect_abs_diff.mean()),
    'sd_abs_effect_difference':float(paired.effect_abs_diff.std(ddof=1)),
    'max_abs_effect_difference':float(paired.effect_abs_diff.max()),
    'direction_agreement_rate':float(paired.direction_agreement.mean()),
    'note':'Split-seed summaries are descriptive because train/test splits overlap and are not independent studies.'
  }
}
Path('results/statistics.json').write_text(json.dumps(stats,indent=2))

bc=load_breast_cancer(); X=np.asarray(bc.data); y=np.asarray(bc.target)
dup=int(pd.DataFrame(X).assign(target=y).duplicated().sum())
synth=[]
for p in sorted(Path('packages').glob('synth_*')):
    df=pd.read_csv(p/'data.csv')
    seed=int(p.name.split('_')[-1])
    synth.append({'seed':seed,'rows':len(df),'duplicates':int(df.duplicated().sum()),'missing_cells':int(df.isna().sum().sum()),'treatment_ones':int(df.treatment.sum()),'treatment_zeros':int((df.treatment==0).sum())})
audit={
 'synthetic':{'packages':synth,'generator':'y = 0.8*x + 1.25*treatment + Normal(0,1)','leakage_risk':'No predictive split; claim is a frozen-table effect estimate.'},
 'public':{'name':'Breast Cancer Wisconsin (Diagnostic)','rows':len(y),'features':X.shape[1],'missing_cells':int(np.isnan(X).sum()),'duplicate_rows_in_features_plus_target':dup,'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'split':'10 prespecified stratified 75/25 splits; scaler fit on train only','raw_redistribution':'Not included; loaded at runtime from scikit-learn.'}
}
Path('results/dataset_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'experiment_registry_rows':len(reg),'seeded_executions':len(exe),'public_runs':len(pub)},indent=2))
