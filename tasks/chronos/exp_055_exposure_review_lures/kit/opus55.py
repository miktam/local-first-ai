#!/usr/bin/env python3
"""exp_055 arm O: batch files for Opus subagents, the same task as arm G, blind.

    python3.14 kit/opus55.py build runs/<ts>/O      # O/batch_{n}.md, 28 messages each, opaque ids only
    python3.14 kit/opus55.py collect runs/<ts>/O    # O/verdicts_{n}.jsonl -> O/results.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms55 as A  # noqa: E402

PER_BATCH = 28


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((A.DATA / "MANIFEST.json").read_text())
    ids = sorted(manifest)
    for b in range(0, len(ids), PER_BATCH):
        n = b // PER_BATCH + 1
        parts = [f"# exp_055 batch {n}: {len(ids[b:b + PER_BATCH])} messages\n",
                 "Each block below is a complete, separate task. Answer each one exactly as it asks.\n"]
        for mid in ids[b:b + PER_BATCH]:
            parts += [f"\n=== MESSAGE {mid} ===\n", A.prompt_for(manifest[mid])]
        (out / f"batch_{n}.md").write_text("\n".join(parts))
    print("built", (len(ids) + PER_BATCH - 1) // PER_BATCH, "batches in", out)


def collect(out: Path):
    rows = []
    for f in sorted(out.glob("verdicts_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            ok = d.get("band") in A.BANDS
            rows.append({"id": d["id"], "band": d["band"] if ok else None, "used_items": d.get("used_items", []),
                         "why": d.get("why")})
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "verdicts collected")


if __name__ == "__main__":
    (build if sys.argv[1] == "build" else collect)(Path(sys.argv[2]))
