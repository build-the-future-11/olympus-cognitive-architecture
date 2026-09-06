from pantheon.package import create_synthetic_package
from pantheon.runner import run_package
from pantheon.adjudication import pantheon_adjudicate, pairwise_artifact_baseline

def test_clean_exact_reproduction(tmp_path):
    pkg=create_synthetic_package(tmp_path/'pkg',seed=1,n=100)
    p=tmp_path/'p'; r=tmp_path/'r'
    run_package(pkg,p,'proposer',1,'clean',proposer_canary=True)
    run_package(pkg,r,'replicator',1,'clean')
    out=pantheon_adjudicate(pkg,p,r,'clean')
    assert out['label']=='REPRODUCED'
    assert out['disagreement']['d_z']==0.0

def test_data_mismatch_localized(tmp_path):
    pkg=create_synthetic_package(tmp_path/'pkg',seed=2,n=100)
    p=tmp_path/'p'; r=tmp_path/'r'
    run_package(pkg,p,'proposer',2,'clean')
    run_package(pkg,r,'replicator',2,'data_mismatch')
    assert pantheon_adjudicate(pkg,p,r,'data_mismatch')['label']=='DATA_MISMATCH'

def test_shared_bug_requires_canonical_audit(tmp_path):
    pkg=create_synthetic_package(tmp_path/'pkg',seed=4,n=100)
    p=tmp_path/'p'; r=tmp_path/'r'
    run_package(pkg,p,'proposer',4,'shared_bug')
    run_package(pkg,r,'replicator',4,'shared_bug')
    assert pairwise_artifact_baseline(p,r)=='REPRODUCED'
    assert pantheon_adjudicate(pkg,p,r,'shared_bug')['label']=='IMPLEMENTATION_MISMATCH'

def test_canary_contamination_detected(tmp_path):
    pkg=create_synthetic_package(tmp_path/'pkg',seed=5,n=100)
    p=tmp_path/'p'; r=tmp_path/'r'
    run_package(pkg,p,'proposer',5,'clean',proposer_canary=True)
    run_package(pkg,r,'replicator',5,'hidden_contamination')
    assert pantheon_adjudicate(pkg,p,r,'hidden_contamination')['label']=='HIDDEN_STATE_CONTAMINATION'
