"""exp_046's sets: exp_043's and exp_044's public dev/test copies concatenated, each row tagged with its source.
Rebuilds data/leads46/{dev,test}.jsonl from a local-first-ai checkout and checks the registered hashes.

usage: build_leads46.py <path to local-first-ai/tasks/chronos>"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTERED = {"dev": "91fb8e0cde3824eb", "test": "a5f823d9c98f674d"}   # prefixes of the sha256 in translate46.SETS

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ch = Path(sys.argv[1])
    src = {"exp_043": ch / "exp_043_kev_lead_triage/data", "exp_044": ch / "exp_044_kev_lead_triage_rev4/data"}
    (ROOT / "data/leads46").mkdir(parents=True, exist_ok=True)
    for s in ("dev", "test"):
        lines = []
        for name, d in src.items():
            for l in (d / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    r["source"] = name
                    lines.append(json.dumps(r, ensure_ascii=False))
        p = ROOT / "data/leads46" / f"{s}.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(s, len(lines), h[:16], "OK" if h.startswith(REGISTERED[s]) else "DIFFERS from the registered set")
