from pantheon.external.scoring import answers_match,score_answer_set,summarize_pair


def test_joint_null_is_abstention_not_false_consensus():
    summary=summarize_pair({'q':None},{'q':None},{'q':7})
    assert summary['pairwise_agreement_count']==0
    assert summary['false_consensus_count']==0

def test_numeric_tolerance_and_string_normalization():
    assert answers_match('  LASSO ','lasso')
    assert answers_match(1.00001,1.0,atol=1e-3)

def test_accepted_value_sets():
    s=score_answer_set({'q':0.9373},{'q':[0.9372,0.9373,0.9375]},atol=1e-4)
    assert s['all_correct']

def test_false_consensus_detected():
    s=summarize_pair({'q':4},{'q':4},{'q':[5,6]})
    assert s['false_consensus_count']==1
    assert s['pairwise_agreement_count']==1

def test_correct_consensus_not_false():
    s=summarize_pair({'q':'Blue'},{'q':' blue '},{'q':['Blue']})
    assert s['false_consensus_count']==0
    assert s['agent_a']['all_correct'] and s['agent_b']['all_correct']
