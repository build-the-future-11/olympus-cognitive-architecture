#!/usr/bin/env python3
from __future__ import annotations
import json, math, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pantheon.hashing import sha256_json

OUT=Path('runs/public_case');
import shutil
shutil.rmtree(OUT,ignore_errors=True); OUT.mkdir(parents=True,exist_ok=True)
DATA=load_breast_cancer()
X=np.asarray(DATA.data,float); y=np.asarray(DATA.target,int)
dataset_hash=sha256_json({'X':X.round(12).tolist(),'y':y.tolist(),'feature_names':DATA.feature_names.tolist()})

def sigmoid(z):
    z=np.clip(z,-40,40); return 1/(1+np.exp(-z))

def numpy_logreg_fit(X,y,lr=0.08,steps=7000,l2=1e-3):
    w=np.zeros(X.shape[1]); b=0.0
    n=len(y)
    for _ in range(steps):
        p=sigmoid(X@w+b); err=p-y
        w -= lr*((X.T@err)/n + l2*w); b -= lr*float(err.mean())
    return w,b

rows=[]
for split_seed in [101,211,307,401,503,601,701,809,907,1009]:
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=split_seed,stratify=y)
    # Shared protocol but independently implemented training paths. Train-only scaling statistics.
    sc=StandardScaler().fit(Xtr); A=sc.transform(Xtr); B=sc.transform(Xte)
    majority=max(np.mean(yte==0),np.mean(yte==1))
    t=time.perf_counter(); model=LogisticRegression(C=10.0,max_iter=5000,solver='lbfgs',random_state=split_seed).fit(A,ytr)
    p1=model.predict_proba(B)[:,1]; pred1=(p1>=0.5).astype(int); rt1=time.perf_counter()-t
    t=time.perf_counter(); w,b=numpy_logreg_fit(A,ytr); p2=sigmoid(B@w+b); pred2=(p2>=0.5).astype(int); rt2=time.perf_counter()-t
    for role,pred,prob,rt,impl in [('proposer',pred1,p1,rt1,'sklearn_lbfgs'),('replicator',pred2,p2,rt2,'numpy_batch_gd')]:
        acc=float(accuracy_score(yte,pred)); auc=float(roc_auc_score(yte,prob)); delta=acc-float(majority)
        row={'case_id':f'wdbc-{split_seed}','split_seed':split_seed,'role':role,'implementation':impl,'n_train':len(ytr),'n_test':len(yte),'majority_accuracy':majority,'accuracy':acc,'roc_auc':auc,'effect':delta,'runtime_sec':rt,'dataset_hash':dataset_hash}
        rows.append(row)
        rd=OUT/f'wdbc-{split_seed}'/role; rd.mkdir(parents=True,exist_ok=True); (rd/'metrics.json').write_text(json.dumps(row,indent=2,sort_keys=True))

df=pd.DataFrame(rows); df.to_csv('results/public_case_results.csv',index=False)
wide=df.pivot(index='split_seed',columns='role',values=['effect','accuracy','roc_auc'])
paired=[]
for seed in sorted(df.split_seed.unique()):
    a=df[(df.split_seed==seed)&(df.role=='proposer')].iloc[0]; b=df[(df.split_seed==seed)&(df.role=='replicator')].iloc[0]
    paired.append({'split_seed':seed,'effect_proposer':a.effect,'effect_replicator':b.effect,'effect_abs_diff':abs(a.effect-b.effect),'accuracy_proposer':a.accuracy,'accuracy_replicator':b.accuracy,'auc_proposer':a.roc_auc,'auc_replicator':b.roc_auc,'direction_agreement':int(np.sign(a.effect)==np.sign(b.effect))})
pd.DataFrame(paired).to_csv('results/public_case_paired.csv',index=False)
summary={}
for role in ['proposer','replicator']:
    d=df[df.role==role]
    summary[role]={k:{'mean':float(d[k].mean()),'sd':float(d[k].std(ddof=1))} for k in ['effect','accuracy','roc_auc','runtime_sec']}
summary['paired']={'mean_abs_effect_difference':float(pd.DataFrame(paired).effect_abs_diff.mean()),'direction_agreement_rate':float(pd.DataFrame(paired).direction_agreement.mean())}
summary['dataset']={'name':'Breast Cancer Wisconsin (Diagnostic), sklearn bundled copy','rows':len(y),'features':X.shape[1],'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'sha256_semantic':dataset_hash,'source_description':DATA.DESCR[-1600:]}
summary['environment']={'python':sys.version.split()[0],'platform':platform.platform()}
Path('results/public_case_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
