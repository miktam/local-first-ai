"""Asset presence and revision checks for preflight and the version record
(BUILD_SPEC §5.5 `verify_revision`, §5.9 preflight; HYPOTHESIS "Fixed before
any run": every Kolibri shard's sha256 is checked against its LFS oid).

Hugging Face directories: `hf download --local-dir` writes one
`.cache/huggingface/download/<relpath>.metadata` per file: line 1 the commit,
line 2 the etag (the LFS sha256 for LFS files, the git blob sha1 otherwise).
A file counts as at the pinned revision when its metadata commit equals the
pin. Git clones: `.git/HEAD` (detached commit, or a ref resolved through
`refs/` and `packed-refs`). No network.

Quick checks read metadata, sizes and `.git`; files assets.json gives a sha256
for are hashed (they are small). `deep=True` hashes every file that has
metadata against its etag (sha256 or git blob sha1), with parallel workers;
Kolibri shards are also compared with the committed LFS oids in
`tools/kolibri_lfs_oids.json`.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common
from tools.redact import redact_path

LFS_OIDS_FILE = common.TOOLS_DIR / "kolibri_lfs_oids.json"
KOLIBRI_REPO = "Aleph-Alpha/Kolibri-1-BF16"
META_ROOT = Path(".cache") / "huggingface" / "download"
NON_DATA = {"README.md", ".gitattributes", "LICENSE", "LICENSE.txt"}


def load_lfs_oids(path=LFS_OIDS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hf_metadata(d: Path) -> dict:
    """{relpath: {"commit": str, "etag": str}} from the download metadata."""
    root = Path(d) / META_ROOT
    out = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*.metadata"):
        rel = p.relative_to(root).as_posix()[: -len(".metadata")]
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        out[rel] = {"commit": lines[0].strip() if lines else "", "etag": lines[1].strip() if len(lines) > 1 else ""}
    return out


def incomplete_files(d: Path) -> int:
    root = Path(d) / META_ROOT
    return sum(1 for _ in root.rglob("*.incomplete")) if root.is_dir() else 0


def content_files(d: Path) -> dict:
    """{relpath: bytes} of the files outside .cache/ and .git/."""
    d = Path(d)
    out = {}
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = [x for x in dirnames if x not in (".cache", ".git")]
        for fn in filenames:
            p = Path(dirpath, fn)
            try:
                out[p.relative_to(d).as_posix()] = p.stat().st_size
            except OSError:
                pass
    return out


def git_head(d: Path) -> str | None:
    g = Path(d) / ".git"
    try:
        head = (g / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    try:
        return (g / ref).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        for line in (g / "packed-refs").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        pass
    return None


def _hash_one(args):
    path, etag = args
    try:
        if len(etag) == 64:
            return path, "sha256", common.sha256_file(path)
        if len(etag) == 40:
            return path, "git_blob_sha1", common.git_blob_sha1_file(path)
        return path, "none", None
    except OSError as e:
        return path, "error", str(e)


def deep_hash(d: Path, meta: dict, workers: int) -> dict:
    """{relpath: {"algo", "value", "etag", "ok"}} for every file with metadata."""
    jobs = []
    for rel, m in sorted(meta.items()):
        p = Path(d) / rel
        if p.is_file() and m["etag"]:
            jobs.append((p, m["etag"]))
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for p, algo, value in ex.map(_hash_one, jobs):
            rel = p.relative_to(d).as_posix()
            etag = meta[rel]["etag"]
            out[rel] = {"algo": algo, "value": value, "etag": etag, "ok": value == etag}
    return out


def _hf_entry(entry: dict, kind: str, deep: bool, workers: int, lfs: dict | None) -> dict:
    d = common.asset_path(entry["path"])
    pin = entry["revision"]
    rec = {
        "kind": kind, "repo": entry["repo"], "path": redact_path(d), "pinned_revision": pin,
        "present": d.is_dir(),
    }
    problems: list[str] = []
    if "bytes" in entry:
        rec["bytes_manifest"] = entry["bytes"]
    if not d.is_dir():
        rec.update(complete=False, revision_ok=False, bytes_on_disk=0, problems=["absent"])
        return rec
    meta = hf_metadata(d)
    files = content_files(d)
    revs = sorted({m["commit"] for m in meta.values() if m["commit"]})
    data_files = [r for r in files if Path(r).name not in NON_DATA]
    without_meta = sorted(r for r in files if r not in meta)
    rec.update(
        recorded_revisions=revs,
        revision_ok=bool(meta) and revs == [pin],
        incomplete_files=incomplete_files(d),
        files=len(files), data_files=len(data_files),
        files_without_metadata=len(without_meta),
        bytes_on_disk=sum(files.values()),
    )
    if not rec["revision_ok"]:
        problems.append(f"recorded revisions {revs or 'none'} != pin {pin[:8]}")
    if rec["incomplete_files"]:
        problems.append(f"{rec['incomplete_files']} .incomplete downloads")
    if not data_files:
        problems.append("no data files (only README/attributes)")
    if without_meta:
        problems.append(f"{len(without_meta)} files without download metadata")
    if "bytes" in entry:
        rec["bytes_manifest"] = entry["bytes"]
        rec["size_ok"] = abs(rec["bytes_on_disk"] - entry["bytes"]) <= 0.01 * entry["bytes"]
        if not rec["size_ok"]:
            problems.append(f"{rec['bytes_on_disk']:,} bytes on disk vs {entry['bytes']:,} in assets.json")
    for rel in entry.get("files", []):
        if rel not in files:
            problems.append(f"missing {rel}")
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        shards = sorted(set(json.loads(idx.read_text()).get("weight_map", {}).values()))
        missing = [s for s in shards if s not in files]
        rec["index_shards"] = len(shards)
        if missing:
            problems.append(f"{len(missing)} of {len(shards)} shards missing")
    # sha256 recorded in assets.json (small files): hashed in every mode.
    sha_ok = {}
    for rel, want in (entry.get("sha256") or {}).items():
        p = d / rel
        sha_ok[rel] = p.is_file() and common.sha256_file(p) == want
        if not sha_ok[rel]:
            problems.append(f"sha256 mismatch or missing: {rel}")
    if sha_ok:
        rec["sha256_ok"] = sha_ok
    if lfs is not None:
        rec["lfs"] = _lfs_quick(d, meta, files, lfs, problems)
    if deep:
        t0 = common.utc_iso()
        hashed = deep_hash(d, meta, workers)
        bad = sorted(r for r, h in hashed.items() if not h["ok"])
        rec["deep"] = {
            "t_start": t0, "t_end": common.utc_iso(), "files_hashed": len(hashed), "mismatches": bad,
            "sha256": {r: h["value"] for r, h in hashed.items() if h["algo"] == "sha256"},
        }
        if bad:
            problems.append(f"{len(bad)} files differ from their etag")
        if lfs is not None:
            lfs_bad = sorted(
                r for r, v in lfs["shards"].items()
                if hashed.get(r, {}).get("value") != v["sha256"]
            )
            rec["lfs"]["deep_mismatches"] = lfs_bad
            if lfs_bad:
                problems.append(f"{len(lfs_bad)} shards differ from the committed LFS oids")
    rec["problems"] = problems
    rec["complete"] = not problems
    return rec


def _lfs_quick(d: Path, meta: dict, files: dict, lfs: dict, problems: list) -> dict:
    """Metadata etags and sizes of the Kolibri shards against the committed oids."""
    etag_bad, size_bad, missing = [], [], []
    for rel, v in sorted(lfs["shards"].items()):
        if rel not in files:
            missing.append(rel)
            continue
        if files[rel] != v["bytes"]:
            size_bad.append(rel)
        if meta.get(rel, {}).get("etag") != v["sha256"]:
            etag_bad.append(rel)
    if missing:
        problems.append(f"{len(missing)} Kolibri shards missing")
    if size_bad:
        problems.append(f"{len(size_bad)} Kolibri shards differ in size from the LFS record")
    if etag_bad:
        problems.append(f"{len(etag_bad)} Kolibri shard etags differ from the committed LFS oids")
    return {"source": "tools/kolibri_lfs_oids.json", "shards_expected": len(lfs["shards"]),
            "missing": missing, "size_mismatch": size_bad, "etag_mismatch": etag_bad}


def _code_entry(entry: dict) -> dict:
    d = common.asset_path(entry["path"])
    rec = {"kind": "code", "repo": entry["repo"], "path": redact_path(d),
           "pinned_commit": entry["commit"], "present": d.is_dir()}
    problems = []
    if not d.is_dir():
        rec.update(complete=False, revision_ok=False, problems=["absent"])
        return rec
    head = git_head(d)
    rec["head"] = head
    rec["revision_ok"] = head == entry["commit"]
    if not rec["revision_ok"]:
        problems.append(f"HEAD {str(head)[:8]} != pin {entry['commit'][:8]}")
    sha_ok = {}
    for rel, want in (entry.get("sha256") or {}).items():
        p = d / rel
        sha_ok[rel] = p.is_file() and common.sha256_file(p) == want
        if not sha_ok[rel]:
            problems.append(f"sha256 mismatch or missing: {rel}")
    if sha_ok:
        rec["sha256_ok"] = sha_ok
    rec["problems"] = problems
    rec["complete"] = not problems
    return rec


def check_all(assets: dict | None = None, deep: bool = False, workers: int = 8) -> dict:
    """Check every models / datasets / code entry of assets.json.
    Returns {"models": [...], "datasets": [...], "code": [...],
    "environments": [...], "complete": bool, "problems": [str]}."""
    assets = assets or common.load_assets()
    lfs = load_lfs_oids() if LFS_OIDS_FILE.is_file() else None
    out = {"models": [], "datasets": [], "code": [], "environments": []}
    for e in assets.get("models", []):
        use_lfs = lfs if (lfs and e["repo"] == KOLIBRI_REPO and e["revision"] == lfs["revision"]) else None
        out["models"].append(_hf_entry(e, "model", deep, workers, use_lfs))
    for e in assets.get("datasets", []):
        out["datasets"].append(_hf_entry(e, "dataset", deep, workers, None))
    for e in assets.get("code", []):
        out["code"].append(_code_entry(e))
    for e in assets.get("environments", []):
        p = common.asset_path(e["path"])
        out["environments"].append({"name": e["name"], "path": redact_path(p), "present": p.exists(),
                                    "required_on_run_host": False})
    problems = []
    for group in ("models", "datasets", "code"):
        for r in out[group]:
            for p in r.get("problems", []):
                problems.append(f"{r['repo']}: {p}")
    out["problems"] = problems
    out["complete"] = not problems
    return out


def revisions_summary(result: dict) -> dict:
    """{repo: {"pinned", "found", "ok"}} for the version record."""
    out = {}
    for group in ("models", "datasets"):
        for r in result.get(group, []):
            out[r["repo"]] = {"pinned": r["pinned_revision"], "found": r.get("recorded_revisions", []),
                              "ok": bool(r.get("revision_ok"))}
    for r in result.get("code", []):
        out[r["repo"]] = {"pinned": r["pinned_commit"], "found": [r["head"]] if r.get("head") else [],
                          "ok": bool(r.get("revision_ok"))}
    return out
