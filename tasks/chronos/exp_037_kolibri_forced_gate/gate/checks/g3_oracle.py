# SPDX-License-Identifier: MIT
"""G3: the reference as an oracle (HYPOTHESIS Phase 0 G3; BUILD_SPEC §5.3 g3_oracle.py).

  ref_bpb          reference bits per UTF-8 byte on the plain texts T1-T6, per
                   text and pooled over T1-T6
  peer_bpb         the peers G8 and Q36-8, pooled the same way, read from the
                   newest results/peers_<UTC>.json (tools/peer_check.py)
  ref_mutant_nll   the seven reference mutants on T3: a 48-block bootstrap of
                   NLL_mutant - NLL_ref, its 1st and 99th percentiles

G3 and the vendor's routing test are the only checks independent of the shared
specification (HYPOTHESIS Phase 0 "Reference").

exp_037 (DESIGN §3.6; build task W5a): this module only measures. The rules
are gate/rules.py's g3_checks and g3_mutant_class (exp_036's l.95-100 and
evaluate(), l.106-131): G3a, bits per byte, is descriptive, with exp_036's
limits printed beside it; G3b fails only if some mutant's p99 < 0 or the
mutant count is not 7 (p01 <= 0 "rests on vendor source" and does not fail).
The bootstrap seeds are common.seed_from("G3", <mutant>), under the exp037
prefix (DESIGN §2.10). Nothing here emits "pass" or any verdict.
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


def bootstrap_seed(name: str) -> int:
    """The G3b bootstrap seed of one mutant: common.seed_from("G3", name), under
    the exp037 prefix (DESIGN §2.10; exp_036 drew under exp036)."""
    return common.seed_from("G3", name)


def ref_mutant_nll(nll: dict, blocks: int, quantile: float, B: int = 10_000) -> dict:
    """nll: {"ref": per-token NLL, <mutant>: per-token NLL} on T3 (from
    ref_pass.mutant_nll, one forward_streams pass). Per mutant the bootstrap
    of NLL_mutant - NLL_ref: {"dnll_mean", "dnll_p01", "dnll_p99", "blocks",
    "B", "quantile", "seed"}; rules.g3_mutant_class classifies it."""
    ref_nll = np.asarray(nll["ref"], dtype=np.float64)
    out = {}
    for name, m in nll.items():
        if name == "ref":
            continue
        d = np.asarray(m, dtype=np.float64) - ref_nll
        seed = bootstrap_seed(name)
        s = block_bootstrap(d, blocks, B, seed, quantile)
        out[name] = {"dnll_mean": s["mean"], "dnll_p01": s["p01"], "dnll_p99": s["p99"], "blocks": blocks, "B": B,
                     "quantile": quantile, "seed": seed}
    return out
