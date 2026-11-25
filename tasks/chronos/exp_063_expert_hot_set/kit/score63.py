"""Coverage: rank the experts of each block on one text, score the top-k's share of activations on another.
Per block normalised by the block's own total, then the plain mean over blocks."""
import itertools, json, os, sys
R = sys.argv[1]; CL = ["en_book", "ru_lecture", "es_site", "cpp_code"]; KS = [.25, .5, .75]
def load(m, t): return {int(k): v for k, v in json.load(open(f"{R}/{m}-{t}.json")).items()}
def add(ds):
    o = {}
    for b in ds[0]: o[b] = [sum(d[b][i] / sum(d[b]) for d in ds) for i in range(len(ds[0][b]))]
    return o
def cov(rank, score, f):
    c = []
    for b in score:
        n = len(score[b]); k = round(n * f); top = sorted(range(n), key=lambda i: (-rank[b][i], i))[:k]
        c.append(sum(score[b][i] for i in top) / sum(score[b]))
    return sum(c) / len(c)
res = {}
for m in ("gemma", "inkling"):
    if not all(os.path.exists(f"{R}/{m}-{c}_{h}.json") for c in CL for h in "ab"): continue
    A = {c: load(m, c + "_a") for c in CL}; B = {c: load(m, c + "_b") for c in CL}; r = {}
    for f in KS:
        r[f] = {"matrix_rank_a_score_b": {x: {y: round(cov(A[x], B[y], f), 4) for y in CL} for x in CL},
                "mixed": {y: round(cov(add(list(A.values())), B[y], f), 4) for y in CL},
                "leave_one_out": {y: round(cov(add([A[x] for x in CL if x != y]), B[y], f), 4) for y in CL},
                "oracle": {y: round(cov(B[y], B[y], f), 4) for y in CL}, "null": f}
    s = r[.5]["matrix_rank_a_score_b"]
    same = {c: s[c][c] for c in CL}; cross = {y: sum(s[x][y] for x in CL if x != y) / 3 for y in CL}
    r["H1"] = {"same": same, "pass": all(v >= .80 for v in same.values())}
    r["H2"] = {"drop": {c: round(same[c] - cross[c], 4) for c in CL}, "mean_drop": round(sum(same[c] - cross[c] for c in CL) / 4, 4)}
    r["H2"]["pass"] = r["H2"]["mean_drop"] >= .10
    r["H3"] = {"mixed": r[.5]["mixed"], "pass": all(v >= .75 for v in r[.5]["mixed"].values())}
    res[m] = r
json.dump(res, open(f"{R}/score.json", "w"), indent=1); print(json.dumps(res, indent=1))
