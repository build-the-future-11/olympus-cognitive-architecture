#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd
checks=[]
def add(name,ok,detail): checks.append({'check':name,'pass':bool(ok),'detail':detail})
# local evidence
r=subprocess.run([sys.executable,'scripts/validate_evidence.py'],capture_output=True,text=True)
add('local_evidence_validator',r.returncode==0,(r.stdout+r.stderr).strip()[-1000:])
import os
env={**os.environ,'PYTHONPATH':str(Path('src').resolve())}
r=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True,env=env)
add('tests',r.returncode==0,(r.stdout+r.stderr).strip()[-1000:])
# external evidence
rp=Path('results/external/external_task_results.csv'); mp=Path('external/corebench/manifest.jsonl')
if rp.exists() and mp.exists():
    df=pd.read_csv(rp); tasks=[json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    tmeta={t['task_id']:t for t in tasks}; vis={tmeta.get(x,{}).get('benchmark_visibility','unknown') for x in df.task_id}
    fields={tmeta.get(x,{}).get('field') for x in df.task_id if tmeta.get(x,{}).get('field')}
    distinct_pairs=df[['agent_a_provider','agent_a_model','agent_b_provider','agent_b_model']].drop_duplicates()
    independent=all((r.agent_a_provider,r.agent_a_model)!=(r.agent_b_provider,r.agent_b_model) for _,r in df.iterrows())
    add('external_tasks',len(df)>=15,f'{len(df)} tasks; threshold 15')
    add('external_questions',int(df.question_count.sum())>=20,f"{int(df.question_count.sum())} scored questions; threshold 20")
    add('agent_independence',independent,'agent identities differ on every task')
    add('cross_domain',len(fields)>=2,f'{len(fields)} fields: {sorted(fields)}')
    add('heldout_or_ood',bool(vis & {'heldout','ood'}),f'visibility={sorted(vis)}; public_train is pilot evidence only')
    numeric_complete=not df[['agent_a_accuracy','agent_b_accuracy','pairwise_agreement_rate','false_consensus_rate']].isna().any().any()
    report_columns={'agent_a_report_exists','agent_b_report_exists'}
    reports_complete=report_columns.issubset(df.columns) and bool(df[list(report_columns)].all().all())
    error_columns={'agent_a_provider_error','agent_b_provider_error'}
    providers_clean=error_columns.issubset(df.columns) and bool(df[list(error_columns)].isna().all().all())
    add('complete_reports',numeric_complete and reports_complete and providers_clean,f'numeric={numeric_complete}; reports={reports_complete}; provider_errors_absent={providers_clean}')
else:
    for n in ['external_tasks','external_questions','agent_independence','cross_domain','heldout_or_ood','complete_reports']: add(n,False,'external study not yet run')
passed=all(c['pass'] for c in checks); verdict='PUBLICATION-READY' if passed else 'FINAL-EXPERIMENT READY'
out={'verdict':verdict,'checks':checks}
Path('PUBLICATION_READINESS.json').write_text(json.dumps(out,indent=2))
Path('PUBLICATION_READINESS.md').write_text('# Publication Readiness\n\n**Verdict: '+verdict+'**\n\n'+'\n'.join(f"- {'PASS' if c['pass'] else 'BLOCK'} — **{c['check']}**: {c['detail']}" for c in checks)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if passed else 3)
