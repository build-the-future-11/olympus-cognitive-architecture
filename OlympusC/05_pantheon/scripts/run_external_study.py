#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, tarfile, tempfile, time, zipfile
from pathlib import Path
from pantheon.external.agent_loop import run_agent
from pantheon.external.scoring import summarize_pair


def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_zip_extract(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src) as archive:
        for member in archive.infolist():
            if not _inside(dst, dst / member.filename):
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        archive.extractall(dst)


def _safe_tar_extract(src: Path, dst: Path) -> None:
    with tarfile.open(src) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported TAR link/device member: {member.name}")
            if not _inside(dst, dst / member.name):
                raise ValueError(f"unsafe TAR member: {member.name}")
        archive.extractall(dst)


def copy_task(src: Path, dst: Path):
    if src.is_dir(): shutil.copytree(src,dst)
    elif src.suffix=='.zip':
        dst.mkdir(parents=True); _safe_zip_extract(src,dst)
    elif src.name.endswith(('.tar.gz','.tgz','.tar')):
        dst.mkdir(parents=True); _safe_tar_extract(src,dst)
    else: raise ValueError(f"Unsupported task source: {src}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='external/corebench_manifest.jsonl')
    ap.add_argument('--agent-a-provider',default=os.getenv('PANTHEON_AGENT_A_PROVIDER','openai'))
    ap.add_argument('--agent-a-model',default=os.getenv('PANTHEON_AGENT_A_MODEL','gpt-5.6-terra'))
    ap.add_argument('--agent-b-provider',default=os.getenv('PANTHEON_AGENT_B_PROVIDER','anthropic'))
    ap.add_argument('--agent-b-model',default=os.getenv('PANTHEON_AGENT_B_MODEL','claude-sonnet-4-5'))
    ap.add_argument('--max-tasks',type=int,default=0)
    ap.add_argument('--max-turns',type=int,default=40)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    tasks=[json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    if args.max_tasks: tasks=tasks[:args.max_tasks]
    outroot=Path('runs/external'); outroot.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ti,t in enumerate(tasks,1):
        tid=t['task_id']; print(f"[{ti}/{len(tasks)}] {tid}",flush=True)
        canonical=Path(t['canonical_answers'])
        expected=json.loads(canonical.read_text())
        questions=list(expected)
        taskdir=outroot/tid; taskdir.mkdir(parents=True,exist_ok=True)
        pair={}
        for role,provider,model in [('agent_a',args.agent_a_provider,args.agent_a_model),('agent_b',args.agent_b_provider,args.agent_b_model)]:
            rd=taskdir/role; result_file=rd/'agent_result.json'
            if args.resume and result_file.exists():
                pair[role]=json.loads(result_file.read_text()); continue
            shutil.rmtree(rd,ignore_errors=True); rd.mkdir(parents=True)
            start=time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f"pantheon-{tid}-{role}-", dir="/private/tmp") as tmp:
                workspace=Path(tmp)/'workspace'; copy_task(Path(t['source']),workspace)
                result=run_agent(provider,model,workspace,t['task_prompt'],questions,max_turns=args.max_turns)
                for retained in ("report.json", "report.agent.json", "agent_notes.md"):
                    artifact = workspace / retained
                    if artifact.exists() and artifact.is_file():
                        shutil.copy2(artifact, rd / retained)
            result.update({'role':role,'provider':provider,'model':model,'runtime_sec':time.perf_counter()-start,'task_id':tid,'source':t['source'],'source_sha256':sha256(Path(t['source'])) if Path(t['source']).is_file() else None})
            result.update({'workspace_isolation':'macos sandbox-exec; repository and user home denied; network denied','workspace_ephemeral':True})
            result_file.write_text(json.dumps(result,indent=2,sort_keys=True)); pair[role]=result
        summary=summarize_pair(pair['agent_a']['answers'],pair['agent_b']['answers'],expected,atol=float(t.get('atol',1e-6)),rtol=float(t.get('rtol',1e-4)))
        summary.update({'task_id':tid,'agent_a_provider':args.agent_a_provider,'agent_a_model':args.agent_a_model,'agent_b_provider':args.agent_b_provider,'agent_b_model':args.agent_b_model,'canonical_answers_sha256':sha256(canonical),'agent_a_report_exists':pair['agent_a']['report_exists'],'agent_b_report_exists':pair['agent_b']['report_exists'],'agent_a_submitted_report':pair['agent_a'].get('agent_report_exists',False),'agent_b_submitted_report':pair['agent_b'].get('agent_report_exists',False),'agent_a_report_origin':pair['agent_a'].get('report_origin'),'agent_b_report_origin':pair['agent_b'].get('report_origin'),'agent_a_provider_error':pair['agent_a'].get('provider_error'),'agent_b_provider_error':pair['agent_b'].get('provider_error')})
        (taskdir/'comparison.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
        rows.append(summary)
    flat=[]
    for s in rows:
        flat.append({'task_id':s['task_id'],'agent_a_accuracy':s['agent_a']['accuracy'],'agent_b_accuracy':s['agent_b']['accuracy'],'pairwise_agreement_rate':s['pairwise_agreement_rate'],'false_consensus_count':s['false_consensus_count'],'false_consensus_rate':s['false_consensus_rate'],'question_count':s['question_count'],'agent_a_provider':s['agent_a_provider'],'agent_a_model':s['agent_a_model'],'agent_b_provider':s['agent_b_provider'],'agent_b_model':s['agent_b_model'],'agent_a_report_exists':s['agent_a_report_exists'],'agent_b_report_exists':s['agent_b_report_exists'],'agent_a_submitted_report':s['agent_a_submitted_report'],'agent_b_submitted_report':s['agent_b_submitted_report'],'agent_a_report_origin':s['agent_a_report_origin'],'agent_b_report_origin':s['agent_b_report_origin'],'agent_a_provider_error':s['agent_a_provider_error'],'agent_b_provider_error':s['agent_b_provider_error']})
    Path('results/external').mkdir(parents=True,exist_ok=True)
    with Path('results/external/external_task_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['task_id']); w.writeheader(); w.writerows(flat)
    Path('results/external/external_summary.json').write_text(json.dumps({'n_tasks':len(rows),'tasks':rows},indent=2))

if __name__=='__main__': main()
