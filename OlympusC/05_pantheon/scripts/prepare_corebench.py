#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, time, urllib.error, urllib.request
from pathlib import Path


def unique_values(values):
    out=[]
    for v in values:
        if not any(v==x for x in out): out.append(v)
    return out


def download_atomic(url: str, destination: Path, attempts: int = 4) -> None:
    """Download with retry and Range resume; publish only a complete archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, attempts + 1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"failed to download {url} after {attempts} attempts") from exc
            time.sleep(min(2 ** attempt, 8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Prepare a blinded Pantheon manifest from official CORE-Bench metadata.')
    ap.add_argument('--metadata',default='external/corebench/core_train.json')
    ap.add_argument('--limit',type=int,default=30)
    ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--visibility',choices=['public_train','heldout','ood'],default='public_train')
    ap.add_argument('--download-capsules',action='store_true')
    ap.add_argument('--task-ids',default='',help='comma-separated capsule ids; overrides offset/limit filtering')
    ap.add_argument('--capsule-base-url',default='https://corebench.cs.princeton.edu/capsules/{task_id}.tar.gz')
    args=ap.parse_args()
    meta=json.loads(Path(args.metadata).read_text())
    if args.task_ids:
        wanted=[x.strip() for x in args.task_ids.split(',') if x.strip()]; byid={t['capsule_id']:t for t in meta}; missing=[x for x in wanted if x not in byid]
        if missing: raise SystemExit(f'task ids absent from metadata: {missing}')
        chosen=[byid[x] for x in wanted]
    else:
        chosen=meta[args.offset:args.offset+args.limit]
    base=Path('external/corebench'); capsules=base/'capsules'; canonical=base/'canonical'; capsules.mkdir(parents=True,exist_ok=True); canonical.mkdir(parents=True,exist_ok=True)
    lines=[]
    for t in chosen:
        tid=t['capsule_id']; archive=capsules/f'{tid}.tar.gz'
        if args.download_capsules and not archive.exists():
            url=args.capsule_base_url.format(task_id=tid)
            print(f'downloading {url}',flush=True); download_atomic(url,archive)
        answer_sets=t.get('results',[]); questions=[]
        for s in answer_sets:
            for q in s:
                if q not in questions: questions.append(q)
        accepted={q:unique_values([s[q] for s in answer_sets if q in s]) for q in questions}
        can=canonical/f'{tid}.json'; can.write_text(json.dumps(accepted,indent=2,sort_keys=True))
        lines.append(json.dumps({'task_id':tid,'source':str(archive),'source_sha256':sha256(archive) if archive.exists() else None,'canonical_answers':str(can),'canonical_answers_sha256':sha256(can),'task_prompt':t['task_prompt'],'field':t.get('field'),'language':t.get('language'),'capsule_title':t.get('capsule_title'),'capsule_doi':t.get('capsule_doi'),'benchmark':'CORE-Bench','benchmark_visibility':args.visibility,'atol':1e-4,'rtol':1e-3}))
    out=base/'manifest.jsonl'; out.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'tasks':len(lines),'manifest':str(out),'downloaded':args.download_capsules,'visibility':args.visibility},indent=2))

if __name__=='__main__': main()
