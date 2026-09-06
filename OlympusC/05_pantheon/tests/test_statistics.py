from pantheon.statistics import disagreement_vector, ci_overlap

def test_identical_results_have_zero_disagreement():
    x={'effect':1.0,'se':0.1,'ci95':[0.8,1.2]}
    d=disagreement_vector(x,x)
    assert d['d_z']==0.0 and d['d_rel']==0.0 and d['sign_agreement']==1
    assert 0.99 < d['ci_overlap'] <= 1.0
