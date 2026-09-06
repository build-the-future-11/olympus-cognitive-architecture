from __future__ import annotations
import math
from typing import Any


def _coerce_number(x: Any) -> float | None:
    if isinstance(x, bool): return None
    if isinstance(x,(int,float)) and math.isfinite(float(x)): return float(x)
    if isinstance(x,str):
        try:
            v=float(x.strip()); return v if math.isfinite(v) else None
        except ValueError: return None
    return None


def answers_match(pred: Any, truth: Any, *, atol: float=1e-6, rtol: float=1e-4) -> bool:
    pn=_coerce_number(pred); tn=_coerce_number(truth)
    if pn is not None and tn is not None: return math.isclose(pn,tn,abs_tol=atol,rel_tol=rtol)
    if isinstance(pred,str) and isinstance(truth,str): return " ".join(pred.strip().lower().split()) == " ".join(truth.strip().lower().split())
    return pred == truth


def value_is_accepted(pred: Any, truth: Any, *, atol: float=1e-6, rtol: float=1e-4) -> bool:
    vals=truth if isinstance(truth,list) else [truth]
    return any(answers_match(pred,v,atol=atol,rtol=rtol) for v in vals)


def score_answer_set(predicted: dict[str,Any], expected: dict[str,Any], *, atol: float=1e-6, rtol: float=1e-4) -> dict[str,Any]:
    keys=list(expected); per={k:value_is_accepted(predicted.get(k),expected[k],atol=atol,rtol=rtol) for k in keys}; n=len(keys); correct=sum(per.values())
    return {"correct":int(correct),"total":int(n),"accuracy":float(correct/n) if n else 0.0,"all_correct":bool(n and correct==n),"per_question":per,"missing":[k for k in keys if k not in predicted],"extra":sorted(set(predicted)-set(expected))}


def summarize_pair(a: dict[str,Any], b: dict[str,Any], expected: dict[str,Any], *, atol: float=1e-6, rtol: float=1e-4) -> dict[str,Any]:
    keys=list(expected)
    # Two missing or null answers are joint abstention, not scientific agreement.
    agreement={k:bool(k in a and k in b and a[k] is not None and b[k] is not None and answers_match(a[k],b[k],atol=atol,rtol=rtol)) for k in keys}
    a_ok={k:value_is_accepted(a.get(k),expected[k],atol=atol,rtol=rtol) for k in keys}
    b_ok={k:value_is_accepted(b.get(k),expected[k],atol=atol,rtol=rtol) for k in keys}
    false_consensus={k:bool(agreement[k] and not a_ok[k] and not b_ok[k]) for k in keys}
    return {"question_count":len(keys),"pairwise_agreement_count":sum(agreement.values()),"pairwise_agreement_rate":sum(agreement.values())/len(keys) if keys else 0.0,"false_consensus_count":sum(false_consensus.values()),"false_consensus_rate":sum(false_consensus.values())/len(keys) if keys else 0.0,"agreement":agreement,"false_consensus":false_consensus,"agent_a":score_answer_set(a,expected,atol=atol,rtol=rtol),"agent_b":score_answer_set(b,expected,atol=atol,rtol=rtol)}
