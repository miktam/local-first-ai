"""exp_038 W7: the registered analysis (DESIGN §6, §7; rev 2 §0b). Pure functions plus one entry point; stdlib only.

H1  K4 - G26 correctness on P (v1, n = 60; v2 NOT RUN). Paired bootstrap stratified by set x language, 10,000
    replicates, seed sha256("exp038|H1"). p = (1 + #{rep mean <= 0}) / (B + 1); p_rev = (1 + #{rep mean >= 0.20}) / (B + 1).
    CONFIRMED if p <= 0.05; REFUTED if not CONFIRMED and p_rev <= 0.05; INCONCLUSIVE otherwise.
H3  K4 - G26 on the 33 in-scope DE rows: exact paired sign-flip test by convolution over the integer differences
    (primary), bootstrap beside it. Tested at 0.05 only if H1 is CONFIRMED (fixed sequence); otherwise an estimate
    with a 95 % CI, "not tested".
Headroom: C = mean per-row attainable correctness under X1 (a substance_in_context = false row caps at 1). If
    C - mean(G26) < 0.30, every H1 verdict carries "; ceiling-limited", and REFUTED reads "a +0.20 gain is not
    attainable on this set".
Harm guard: h+ = rows harmful (usefulness <= 1) for K4 and not for G26, h- the reverse; holds if h+ - h- <= 0.
Verdict map: ordered, first match wins (§7), with E38-D1 in place of E37-D1 (rev 2 R2).
D4b: I_i = (K4_DE - G26_DE) - (K4_EN - G26_EN) over the 33 in-scope pairs; mean, paired-bootstrap 95 % CI, sign-flip p.
Reliability: exact agreement and quadratic-weighted kappa on the re-dealt blocks, and the H1 statistic on that subset
    under each judgment (§5.2).
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from fractions import Fraction

DELTA = 0.20
B_REPS = 10_000


def seed_rng(tag: str) -> random.Random:
    return random.Random(int(hashlib.sha256(tag.encode()).hexdigest(), 16))


# --------------------------------------------------------------------------------------------------- tests
def stratified_bootstrap(diffs: list[float], strata: list[str], tag: str, reps: int = B_REPS) -> list[float]:
    by = defaultdict(list)
    for d, s in zip(diffs, strata):
        by[s].append(d)
    rng = seed_rng(tag)
    n = len(diffs)
    means = []
    keys = sorted(by)
    for _ in range(reps):
        tot = 0.0
        for k in keys:
            xs = by[k]
            tot += sum(xs[rng.randrange(len(xs))] for _ in xs)
        means.append(tot / n)
    return means


def h1_test(diffs, strata, tag="exp038|H1", reps=B_REPS) -> dict:
    means = stratified_bootstrap(diffs, strata, tag, reps)
    p = (1 + sum(m <= 0 for m in means)) / (reps + 1)
    p_rev = (1 + sum(m >= DELTA for m in means)) / (reps + 1)
    point = sum(diffs) / len(diffs)
    s = sorted(means)
    ci = (s[int(0.025 * reps)], s[int(0.975 * reps) - 1])
    state = "CONFIRMED" if p <= 0.05 else "REFUTED" if p_rev <= 0.05 else "INCONCLUSIVE"
    return {"n": len(diffs), "point": point, "ci95": ci, "p": p, "p_rev": p_rev, "state": state}


def signflip_upper(values: list[int]) -> float:
    """Exact P(sum of independently sign-flipped values >= observed sum), by convolution (integers)."""
    obs = sum(values)
    dist = {0: Fraction(1)}
    for v in values:
        nd = defaultdict(Fraction)
        for s, pr in dist.items():
            nd[s + v] += pr / 2
            nd[s - v] += pr / 2
        dist = nd
    return float(sum(pr for s, pr in dist.items() if s >= obs))


def h3_test(diffs: list[int], tested: bool, strata=None) -> dict:
    """diffs are integer correctness differences (-2..2). p: H0 mean <= 0, sign-flip of d. p_rev: H0 mean >= 0.20,
    sign-flip of (d - 0.20), done on 5d - 1 to keep integers; lower tail."""
    p = signflip_upper(diffs)
    shifted = [5 * d - 1 for d in diffs]
    p_rev = signflip_upper([-x for x in shifted])
    boot = h1_test([float(d) for d in diffs], strata or ["de"] * len(diffs), tag="exp038|H3")
    point = sum(diffs) / len(diffs)
    if not tested:
        return {"n": len(diffs), "point": point, "ci95": boot["ci95"], "p": p, "p_rev": p_rev, "state": "not tested",
                "bootstrap": boot}
    state = "CONFIRMED" if p <= 0.05 else "REFUTED" if p_rev <= 0.05 else "INCONCLUSIVE"
    return {"n": len(diffs), "point": point, "ci95": boot["ci95"], "p": p, "p_rev": p_rev, "state": state, "bootstrap": boot}


# --------------------------------------------------------------------------------------------------- rules
def ceiling(rows: list[dict]) -> float:
    """rows: dicts with type and substance_in_context. oos rows and substance rows can reach 2, miss rows 1."""
    caps = [2 if (r.get("type") == "oos" or r.get("substance_in_context")) else 1 for r in rows]
    return sum(caps) / len(caps)


def harm_guard(k4: dict, g26: dict, ids: list[str]) -> dict:
    hp = sum(1 for i in ids if k4[i]["usefulness"] <= 1 and g26[i]["usefulness"] > 1)
    hm = sum(1 for i in ids if g26[i]["usefulness"] <= 1 and k4[i]["usefulness"] > 1)
    return {"h_plus": hp, "h_minus": hm, "holds": hp - hm <= 0}


def verdict(h1: dict, floors_ok: bool, failing_floors: list[str], harm: dict, d1_state: str, stop: str | None,
            ceiling_limited: bool, h3: dict, d4b: dict) -> str:
    if stop:
        v = "NOT RUN"
    elif h1["state"] == "REFUTED":
        v = "No gain of +0.20 (in the A12 glue)" if not ceiling_limited else "A +0.20 gain is not attainable on this set (in the A12 glue)"
    elif h1["state"] == "CONFIRMED":
        fails = list(failing_floors)
        if d1_state != "CONFIRMED":
            fails.append(f"E38-D1 {d1_state}")
        if not harm["holds"]:
            fails.append("harm guard")
        if h1["point"] >= DELTA and floors_ok and d1_state == "CONFIRMED" and harm["holds"]:
            v = "Earns a live-bot trial (evidence from the A12 glue only)"
        elif fails:
            v = f"Better, not drop-in ({'; '.join(fails)})"
        else:
            v = "Better by less than the margin"
    else:
        v = f"Not shown (point {h1['point']:+.3f}, 95 % CI [{h1['ci95'][0]:+.3f}, {h1['ci95'][1]:+.3f}]; a gain of +0.20 is neither shown nor ruled out at n = {h1['n']})"
    if v != "NOT RUN":
        v += f"; German: {h3['state']}, interaction {d4b['point']:+.3f} [{d4b['ci95'][0]:+.3f}, {d4b['ci95'][1]:+.3f}]"
        if ceiling_limited and h1["state"] != "REFUTED":
            v += "; ceiling-limited"
    return v


def interaction(k4_de, g26_de, k4_en, g26_en) -> dict:
    I = [(a - b) - (c - d) for a, b, c, d in zip(k4_de, g26_de, k4_en, g26_en)]
    boot = h1_test([float(x) for x in I], ["de"] * len(I), tag="exp038|D4b")
    up = signflip_upper([int(x) for x in I])
    lo = signflip_upper([-int(x) for x in I])
    return {"n": len(I), "point": sum(I) / len(I), "ci95": boot["ci95"], "p_two_sided": min(1.0, 2 * min(up, lo))}


# --------------------------------------------------------------------------------------------------- reliability
def quadratic_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    ra = [sum(O[i]) for i in range(k)]
    cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * ra[i] * cb[j] / n
    return 1.0 - num / den if den else 1.0


def reliability(first: dict, second: dict) -> dict:
    """first/second: {answer_key: correctness}; same keys."""
    keys = sorted(set(first) & set(second))
    a, b = [first[k] for k in keys], [second[k] for k in keys]
    exact = sum(x == y for x, y in zip(a, b)) / len(keys) if keys else math.nan
    return {"n": len(keys), "exact_agreement": exact, "kappa_quadratic": quadratic_kappa(a, b) if keys else math.nan,
            "double_judging_required": exact < 0.80}
