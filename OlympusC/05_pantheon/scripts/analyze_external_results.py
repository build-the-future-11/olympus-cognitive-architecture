#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

def wilson(k,n,z=1.959963984540054):
    if n==0:return [0.0,1.0]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]

p=Path('results/external/external_task_results.csv')
if not p.exists():
    raise SystemExit('No external results found. Run scripts/run_external_study.py first.')
df=pd.read_csv(p)
q=int(df.question_count.sum()); fc=int(df.false_consensus_count.sum())
summary={
 'n_tasks':int(len(df)), 'n_questions':q,
 'agent_a_mean_task_accuracy':float(df.agent_a_accuracy.mean()),
 'agent_b_mean_task_accuracy':float(df.agent_b_accuracy.mean()),
 'mean_pairwise_agreement_rate':float(df.pairwise_agreement_rate.mean()),
 'false_consensus_count':fc,'false_consensus_question_rate':float(fc/q) if q else 0.0,
 'false_consensus_rate_wilson95':wilson(fc,q),
 'agent_pairs':df[['agent_a_provider','agent_a_model','agent_b_provider','agent_b_model']].drop_duplicates().to_dict('records')
}
Path('results/external/external_aggregate.json').write_text(json.dumps(summary,indent=2))
md=f'''# External Agent Results\n\n- Tasks: **{summary['n_tasks']}**\n- Scored questions: **{q}**\n- Agent A mean task accuracy: **{summary['agent_a_mean_task_accuracy']:.3f}**\n- Agent B mean task accuracy: **{summary['agent_b_mean_task_accuracy']:.3f}**\n- Mean pairwise agreement: **{summary['mean_pairwise_agreement_rate']:.3f}**\n- False-consensus questions: **{fc}/{q} ({summary['false_consensus_question_rate']:.3f})**\n- Wilson 95% interval for false-consensus question rate: **[{summary['false_consensus_rate_wilson95'][0]:.3f}, {summary['false_consensus_rate_wilson95'][1]:.3f}]**\n\nFalse consensus means both agents returned mutually matching answers that were outside the accepted canonical answer set. These values are generated from raw run artifacts and are not prefilled.\n'''
Path('paper/generated/external_results.md').write_text(md)
print(json.dumps(summary,indent=2))
