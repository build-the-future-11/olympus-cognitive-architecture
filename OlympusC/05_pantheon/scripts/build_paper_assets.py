#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Path('figures').mkdir(exist_ok=True); Path('paper/generated').mkdir(parents=True,exist_ok=True)
base=pd.read_csv('results/seeded_faults_predictions.csv')
summary=base.groupby('method')['correct'].agg(['count','mean']).reset_index().rename(columns={'mean':'localization_accuracy'})
summary.to_csv('paper/generated/table_main.csv',index=False)
# Figure 1: localization accuracy
fig,ax=plt.subplots(figsize=(6.2,4.0)); ax.bar(summary['method'],summary['localization_accuracy']); ax.set_ylim(0,1.05); ax.set_ylabel('Fault localization accuracy'); ax.set_xlabel('Adjudication method'); ax.set_title('Controlled seeded-fault benchmark'); fig.tight_layout(); fig.savefig('figures/fig1_fault_localization.png',dpi=220); plt.close(fig)
# condition heatmap-like matrix without custom colors
pivot=base.pivot_table(index='condition',columns='method',values='correct',aggfunc='mean')
fig,ax=plt.subplots(figsize=(7.0,4.8)); im=ax.imshow(pivot.to_numpy(),vmin=0,vmax=1,aspect='auto'); ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=25,ha='right'); ax.set_yticks(range(len(pivot.index)),pivot.index); ax.set_title('Localization accuracy by injected fault'); fig.colorbar(im,ax=ax,label='Accuracy'); fig.tight_layout(); fig.savefig('figures/fig2_condition_matrix.png',dpi=220); plt.close(fig)
# public case
pub=pd.read_csv('results/public_case_results.csv'); pub.to_csv('paper/generated/table_public_case.csv',index=False)
fig,ax=plt.subplots(figsize=(6.2,4.0));
for role,g in pub.groupby('role'):
    ax.plot(g['split_seed'].astype(str),g['effect'],marker='o',label=role)
ax.axhline(0,linewidth=1); ax.set_ylabel('Accuracy − majority baseline'); ax.set_xlabel('Prespecified split seed'); ax.set_title('Public-data cross-implementation case study'); ax.legend(); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig('figures/fig3_public_case.png',dpi=220); plt.close(fig)
# diagnostic ablations from same frozen runs
abl=[]
for method,label in [('numeric_only','No artifact hashes'),('pairwise_artifact','No canonical manifest'),('pantheon','Full Pantheon')]:
    sub=base[base.method==method]
    abl.append({'ablation':label,'localization_accuracy':float(sub.correct.mean()),'n':len(sub)})
pd.DataFrame(abl).to_csv('results/ablations.csv',index=False)
Path('paper/generated/summary.json').write_text(json.dumps({'fault_localization':summary.to_dict(orient='records'),'public_case':json.loads(Path('results/public_case_summary.json').read_text()),'ablations':abl},indent=2))
print(pd.DataFrame(abl).to_string(index=False))
