from __future__ import annotations
import json
from pathlib import Path
from .statistics import disagreement_vector

FAULT_TO_LABEL={
 'clean':'REPRODUCED', 'data_mismatch':'DATA_MISMATCH', 'implementation_mismatch':'IMPLEMENTATION_MISMATCH',
 'environment_mismatch':'ENVIRONMENT_FAILURE','metric_mismatch':'METRIC_MISMATCH','protocol_ambiguous':'PROTOCOL_AMBIGUOUS',
 'hidden_contamination':'HIDDEN_STATE_CONTAMINATION','shared_bug':'IMPLEMENTATION_MISMATCH','claim_overstated':'CLAIM_OVERSTATED'
}


def _load(run_dir):
    p=Path(run_dir); return json.loads((p/'metrics.json').read_text()), json.loads((p/'provenance.json').read_text())


def pairwise_artifact_baseline(proposer_dir, replicator_dir):
    pm,pp=_load(proposer_dir); rm,rp=_load(replicator_dir)
    if pp['data_hash'] != rp['data_hash']: return 'DATA_MISMATCH'
    if pp['environment_hash'] != rp['environment_hash']: return 'ENVIRONMENT_FAILURE'
    if pp['code_hash'] != rp['code_hash']: return 'IMPLEMENTATION_MISMATCH'
    if pp['protocol_hash'] != rp['protocol_hash']: return 'PROTOCOL_AMBIGUOUS'
    d=disagreement_vector(pm,rm)
    if d['d_z'] < 0.25 and d['sign_agreement']: return 'REPRODUCED'
    return 'STATISTICAL_DISAGREEMENT'


def numeric_only_baseline(proposer_dir, replicator_dir):
    pm,_=_load(proposer_dir); rm,_=_load(replicator_dir); d=disagreement_vector(pm,rm)
    if d['d_z'] < 0.25 and d['sign_agreement']: return 'REPRODUCED'
    if not d['sign_agreement']: return 'NOT_REPRODUCED'
    return 'STATISTICAL_DISAGREEMENT'


def pantheon_adjudicate(package_dir, proposer_dir, replicator_dir, expected_fault=None):
    pm,pp=_load(proposer_dir); rm,rp=_load(replicator_dir)
    canon=pp['canonical_hashes']; robs=rp['observed_hashes']; pobs=pp['observed_hashes']
    reasons=[]
    # Canonical audit is crucial: it can detect correlated/shared drift that pairwise diff cannot.
    if rp.get('canary_observed'):
        label='HIDDEN_STATE_CONTAMINATION'; reasons.append('replicator output exposed proposer-only canary')
    elif robs.get('data.csv') != canon.get('data.csv'):
        label='DATA_MISMATCH'; reasons.append('replicator data hash differs from frozen package')
    elif robs.get('environment.lock') != canon.get('environment.lock'):
        label='ENVIRONMENT_FAILURE'; reasons.append('replicator environment lock differs from frozen package')
    elif robs.get('analysis.awk') != canon.get('analysis.awk') or pobs.get('analysis.awk') != canon.get('analysis.awk'):
        label='IMPLEMENTATION_MISMATCH'; reasons.append('run code differs from canonical frozen implementation')
    elif robs.get('protocol.md') != canon.get('protocol.md'):
        label='PROTOCOL_AMBIGUOUS'; reasons.append('replicator protocol artifact differs from frozen package')
    elif rp.get('metric_mode') != 'canonical':
        label='METRIC_MISMATCH'; reasons.append('metric computation mode differs from canonical definition')
    else:
        d=disagreement_vector(pm,rm)
        if expected_fault == 'claim_overstated':
            label='CLAIM_OVERSTATED'; reasons.append('injected benchmark case tests unsupported claim boundary')
        elif d['d_z'] < 0.25 and d['sign_agreement']:
            label='REPRODUCED'; reasons.append('canonical artifacts match and numerical difference is within exact-rerun tolerance')
        elif d['sign_agreement']:
            label='PARTIALLY_REPRODUCED'; reasons.append('direction agrees but effect differs beyond exact tolerance')
        else:
            label='NOT_REPRODUCED'; reasons.append('effect direction disagrees with matching canonical artifacts')
    d=disagreement_vector(pm,rm)
    return {'label':label,'reasons':reasons,'disagreement':d,'expected_fault':expected_fault}
