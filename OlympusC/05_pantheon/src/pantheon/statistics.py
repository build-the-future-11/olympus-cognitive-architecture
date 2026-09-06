from __future__ import annotations
import math
from typing import Iterable
import numpy as np

EPS = 1e-12


def ci_overlap(ci_a: Iterable[float], ci_b: Iterable[float]) -> float:
    a0, a1 = map(float, ci_a); b0, b1 = map(float, ci_b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / (union + EPS)


def disagreement_vector(p: dict, r: dict) -> dict:
    ep, er = float(p["effect"]), float(r["effect"])
    sp, sr = float(p["se"]), float(r["se"])
    dz = abs(ep-er) / math.sqrt(sp*sp + sr*sr + EPS)
    rel = abs(ep-er) / (abs(ep) + EPS)
    sign = int(np.sign(ep) == np.sign(er))
    overlap = ci_overlap(p["ci95"], r["ci95"])
    return {"d_z": dz, "d_rel": rel, "sign_agreement": sign, "ci_overlap": overlap}


def mean_sd_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean-1.96*se, "ci95_high": mean+1.96*se}
