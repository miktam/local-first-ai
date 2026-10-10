"""exp_038 R4 (DESIGN §3.3, §3.5): freeze the whole answer path before the DE writer starts.

Copies private/w3/de_strings_edited.json to de_strings_final.json (refuses if a final file already exists with
other bytes), then records the sha256 of every file on the answer path (the vendored glue tree, the German strings,
the A12-DE path, the harness, the MLX backend and their tests) and one tree hash over them (sorted "relpath\\tsha256\\n"
lines, as exp_037's hash_tree). After R4, any change to these files is a typed "adapter fix" amendment.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
SCOPE = ["vendor/coapi_voice", "private/w3/de_strings_final.json", "harness/de_path.py", "harness/run_arm.py",
         "harness/tests/test_de_path.py", "kit/mlx_backend.py", "kit/tests/test_mlx_backend.py", "kit/tests/conftest.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    w3 = P38 / "private/w3"
    final, edited = w3 / "de_strings_final.json", w3 / "de_strings_edited.json"
    if final.exists() and final.read_bytes() != edited.read_bytes():
        sys.exit("REFUSED: de_strings_final.json exists and differs from the edited file")
    if not final.exists():
        shutil.copyfile(edited, final)
    files = {}
    for s in SCOPE:
        p = P38 / s
        for f in ([p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)):
            files[str(f.relative_to(P38))] = sha(f)
    tree = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode("utf-8")).hexdigest()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = {"check": "R4", "utc": stamp, "tree_sha256": tree, "n_files": len(files), "files": files,
           "de_strings_final_sha256": sha(final)}
    out = P38 / "runs" / f"r4_{stamp}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"tree_sha256": tree, "n_files": len(files), "de_strings_final": sha(final)[:12], "record": out.name}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
