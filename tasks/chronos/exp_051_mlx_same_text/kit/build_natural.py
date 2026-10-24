#!/usr/bin/env python3
"""exp_051: build the natural-text inputs. The published synthetic enquiries of exp_043, exp_044 and exp_049
(test sets, file order), joined by blank lines and cut at the last enquiry boundary that keeps the raw text at or
under the target token count (Gemma 4 tokenizer of the MLX build). Every enquiry is distinct, so nothing repeats.

    HF_HUB_OFFLINE=1 python3.14 -I kit/build_natural.py
"""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
C = Path.home() / "REPOS/local-first-ai/tasks/chronos"
SOURCES = ["exp_043_kev_lead_triage/data/test.jsonl", "exp_044_kev_lead_triage_rev4/data/test.jsonl",
           "exp_049_kev_rev5/data/test.jsonl"]
TARGETS = {"4k": 4000, "15k": 15000, "25k": 25000, "35k": 35000}
MLX_ID = "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"


def main():
    tok = AutoTokenizer.from_pretrained(MLX_ID)
    texts = [json.loads(line)["text"].strip() for s in SOURCES for line in open(C / s)]
    manifest = {"sources": {s: hashlib.sha256((C / s).read_bytes()).hexdigest() for s in SOURCES},
                "enquiries_available": len(texts), "sizes": {}}
    for key, target in TARGETS.items():
        lo, hi = 1, len(texts)                       # largest n with tokens(join(texts[:n])) <= target
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tok.encode("\n\n".join(texts[:mid]), add_special_tokens=False)) <= target:
                lo = mid
            else:
                hi = mid - 1
        body = "\n\n".join(texts[:lo])
        n_tok = len(tok.encode(body, add_special_tokens=False))
        out = HERE / "natural" / f"nat_{key}.txt"
        out.write_text(body)
        manifest["sizes"][key] = {"enquiries": lo, "chars": len(body), "tokens_hf": n_tok,
                                  "sha256": hashlib.sha256(body.encode()).hexdigest()}
        print(key, lo, len(body), n_tok)
    (HERE / "natural" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
