from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS=['experiment_id','package_id','condition','seed','method','expected_label','predicted_label','correct','effect_proposer','effect_replicator','d_z','d_rel','sign_agreement','ci_overlap','runtime_sec','status','evidence_dir']

def append_rows(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows)
    if path.exists():
        old=pd.read_csv(path); df=pd.concat([old,df],ignore_index=True)
    df.to_csv(path,index=False)
