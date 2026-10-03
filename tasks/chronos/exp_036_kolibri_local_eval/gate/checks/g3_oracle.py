# SPDX-License-Identifier: MIT
"""G3: the reference as an oracle (HYPOTHESIS Phase 0 G3; BUILD_SPEC §5.3 g3_oracle.py).

  ref_bpb          reference bits per UTF-8 byte on the plain texts T1-T6, per
                   text (each <= 1.2) and the mean pooled over T1-T6 (<= 1.25 x
                   the best of the peers G8 and Q36-8, pooled the same way, read
                   from results/peers_<UTC>.json, tools/peer_check.py)
  ref_mutant_nll   the seven reference mutants on T3: a 48-block bootstrap of
                   NLL_mutant - NLL_ref; the 1st percentile must be > 0. A
                   difference whose interval contains 0 is "not resolvable"
                   and is reported as resting on vendor source, without failing
                   the gate; a mutant resolvably *better* than the reference
                   (99th percentile < 0) fails it (our reading of the rule).

G3 and the vendor's routing test are the only checks independent of the shared
specification (HYPOTHESIS Phase 0 "Reference").
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gate import common
from gate.checks.ref_pass import next_token_nll

PLAIN = ("T1", "T2", "T3", "T4", "T5", "T6")
PEERS = ("G8", "Q36-8")


def ref_bpb(dump, textset) -> dict:
    """Bits per byte of the predicted tokens 1..T-1 of each text; `mean_bpb`
    pools the six texts (sum of NLL / ln 2 / sum of bytes), as the peer check's
    mean_bpb does."""
    L = textset.profile.text_len
    logits = dump.load("logits.npy")
    order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    per_text, per_nll, tot_nll, tot_bytes = {}, {}, 0.0, 0
    for j, tid in enumerate(order):
        if tid not in PLAIN:
            continue
        nll = next_token_nll(logits[j * L:(j + 1) * L], textset.ids[tid])
        nbytes = sum(textset.token_bytes[tid][1:])
        per_nll[tid] = float(nll.mean())
        per_text[tid] = float(nll.sum() / math.log(2) / nbytes)
        tot_nll += float(nll.sum())
        tot_bytes += nbytes
    return {"per_text_bpb": per_text, "mean_bpb": tot_nll / math.log(2) / tot_bytes,
            "unweighted_mean_bpb": float(np.mean(list(per_text.values()))), "per_text_nll": per_nll,
            "rule": "NLL(ids[t+1] | ids[:t+1]) summed over t = 0..T-2, / ln 2, / UTF-8 bytes of tokens 1..T-1; mean pooled over T1-T6"}


def peer_bpb(results_dir: Path) -> dict:
    """{arm: pooled bpb over T1-T6} for G8 and Q36-8 from the newest
    results/peers_<UTC>.json (tools/peer_check.py: rec["arms"][arm]["nll"]
    ["mean_bpb"], with "per_text_bpb" alongside)."""
    recs = sorted(Path(results_dir).glob("peers_*.json"))
    if not recs:
        return {"source": None, "peers": {}}
    rec = common.read_json(recs[-1])
    out, per_text = {}, {}
    for arm in PEERS:
        nll = ((rec.get("arms") or {}).get(arm) or {}).get("nll") or {}
        if isinstance(nll.get("mean_bpb"), (int, float)):
            out[arm] = float(nll["mean_bpb"])
            per_text[arm] = nll.get("per_text_bpb")
    return {"source": common.redact_path(recs[-1]), "peers": out, "per_text": per_text}


def block_bootstrap(d: np.ndarray, blocks: int, B: int, seed: int, q: float = 0.01) -> dict:
    """Mean of d over resampled contiguous blocks (token-weighted); the q and
    1 - q quantiles of the bootstrap distribution as p01 / p99."""
    parts = np.array_split(np.asarray(d, dtype=np.float64), blocks)
    sums = np.array([p.sum() for p in parts])
    counts = np.array([p.size for p in parts], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, blocks, size=(B, blocks))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"mean": float(d.mean()), "p01": float(np.quantile(boot, q)), "p99": float(np.quantile(boot, 1 - q)),
            "blocks": blocks, "B": B, "q": q}


def ref_mutant_nll(nll: dict, blocks: int, quantile: float, B: int = 10_000) -> dict:
    """nll: {"ref": per-token NLL, <mutant>: per-token NLL} on T3 (from
    ref_pass.mutant_nll, one forward_streams pass)."""
    ref_nll = np.asarray(nll["ref"], dtype=np.float64)
    out = {}
    for name, m in nll.items():
        if name == "ref":
            continue
        d = np.asarray(m, dtype=np.float64) - ref_nll
        s = block_bootstrap(d, blocks, B, common.seed_from("G3", name), quantile)
        if s["p01"] > 0:
            verdict = "reference_better"
        elif s["p99"] < 0:
            verdict = "mutant_better"
        else:
            verdict = "not_resolvable (rests on vendor source)"
        out[name] = {"dnll_mean": s["mean"], "dnll_p01": s["p01"], "dnll_p99": s["p99"],
                     "resolvable": s["p01"] > 0 or s["p99"] < 0, "verdict": verdict, "blocks": blocks, "B": B}
    return out


def evaluate(bpb: dict | None, peers: dict | None, mutants: dict | None, th: dict, tiny: bool) -> dict:
    checks = {}
    if bpb is not None:
        worst = max(bpb["per_text_bpb"].values())
        checks["g3_bpb_per_text"] = {"pass": worst <= th["ref_bpb_per_text_max"], "applicable": not tiny,
                                     "measured": bpb, "threshold": {"per_text_max": th["ref_bpb_per_text_max"]}}
        pv = (peers or {}).get("peers", {})
        if pv:
            best = min(pv.values())
            ok = bpb["mean_bpb"] <= th["ref_bpb_mean_vs_best_peer_max"] * best
            checks["g3_bpb_vs_peers"] = {"pass": ok, "applicable": not tiny,
                                         "measured": {"mean_bpb": bpb["mean_bpb"], "peers": pv, "best_peer": best,
                                                      "source": peers.get("source")},
                                         "threshold": {"mean_vs_best_peer_max": th["ref_bpb_mean_vs_best_peer_max"]}}
        else:
            checks["g3_bpb_vs_peers"] = {"pass": None, "applicable": not tiny,
                                         "measured": {"note": "no peer bpb in results/peers_<UTC>.json"},
                                         "threshold": {"mean_vs_best_peer_max": th["ref_bpb_mean_vs_best_peer_max"]}}
    if mutants is not None:
        better = [n for n, m in mutants.items() if m["verdict"] == "mutant_better"]
        checks["g3_ref_mutants"] = {"pass": not better and len(mutants) == th["ref_mutants"], "applicable": not tiny,
                                    "measured": {"mutants": mutants, "mutant_better": better,
                                                 "rests_on_vendor_source": sorted(n for n, m in mutants.items() if not m["resolvable"])},
                                    "threshold": {"text": th["ref_mutant_text"], "blocks": th["ref_mutant_blocks"],
                                                  "quantile": th["ref_mutant_dnll_quantile"], "n_mutants": th["ref_mutants"]}}
    return checks
