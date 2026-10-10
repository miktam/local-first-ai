"""R7 (DESIGN §3.5): cross-host replay on the mini of every MLX row produced on the mbp.

For each MLX arm's row, the vendored glue (with the A12-DE path for German rows) is re-run on the mini with a stub
model that returns, call by call, the raw texts the mbp recorded. Retrieval, routing, citation stripping, the advice
lead and the hand-off are recomputed here. Pass: for every row, the final answer, chunk_ids, context and scope equal
the mbp's record byte for byte. A mismatch is an adapter or host defect.

    python3 harness/r7.py <runs/arms> [arm ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
FIELDS = ("answer", "chunk_ids", "context", "scope")
import os
OUT = Path(os.environ.get("EXP038_R7_OUT", P38 / "runs"))  # tests write elsewhere


def main(arms_dir: Path, arms: list[str]) -> int:
    sys.path[:0] = [str(VENDOR), str(P38 / "harness")]
    import glue as G  # noqa: E402
    import guide_bm25 as g  # noqa: E402
    import run_eval as ev  # noqa: E402
    import de_path  # noqa: E402

    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    de = de_path.load_de()
    report = {}
    for arm in arms:
        n, bad = 0, []
        for f in sorted((arms_dir / arm).glob("*.jsonl")):
            for l in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                if r.get("error"):
                    continue
                raws = [c["raw"] for c in r["calls"] if "raw" in c]
                queue = list(raws)

                def stub(model, messages, options, _q=queue):
                    return {"message": {"content": _q.pop(0)}, "prompt_eval_count": 0, "eval_count": 0}

                real = ev.ollama_chat
                ev.ollama_chat = stub
                try:
                    if r["lang"] == "de":
                        res = de_path.answer_de(G, r["q"], r["route_q"], model=arm, index=idx, de=de)
                    else:
                        res = G.answer(r["q"], model=arm, index=idx, verify=False)
                finally:
                    ev.ollama_chat = real
                n += 1
                diff = [k for k in FIELDS if res.get(k) != r.get(k)]
                if diff or queue:
                    bad.append({"set": f.stem, "id": r["id"], "fields": diff, "unused_raw_calls": len(queue)})
        report[arm] = {"rows": n, "mismatches": bad, "pass": n > 0 and not bad}
    report["verdict"] = "PASS" if all(v["pass"] for k, v in report.items() if k != "verdict") else "FAIL"
    (OUT / "r7_result.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (v if k == "verdict" else {"rows": v["rows"], "mismatches": len(v["mismatches"])}) for k, v in report.items()}))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2:] or ["K4", "K8", "K4-MED"]))
