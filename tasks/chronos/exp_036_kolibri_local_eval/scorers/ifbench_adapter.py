"""IFBench scoring with allenai/IFBench's own checkers, run in place (BUILD_SPEC §5.6; HYPOTHESIS H2, H4).

IFBench @ 1c40f0c1 (Apache-2.0) is never vendored. Its checkers run from the local checkout
($EXP036_DATA/IFBench-src) inside their own environment (the IFBench venv, built from IFBench's uv.lock;
BUILD_SPEC §8), in a subprocess:

    adapter (kit venv)                             driver (IFBench venv, scorers/ifbench_driver.py)
    read IFBench_test (the HF copy, parquet)  ->   evaluation_lib.read_prompt_list
    post-reasoning answers per key            ->   evaluation_lib.read_prompt_to_response_dict
                                                   strict pass, then loose pass (run_eval.py order)
    per-key results                           <-   JSON

The subprocess gets NLTK_DATA (the pre-fetched punkt, punkt_tab, stopwords and perceptron tagger), a
clean environment (no PYTHONPATH, no user site, no bytecode writes into the checkout) and the offline
flags; the driver blocks sockets and nltk.download before IFBench is imported.

The headline is IFBench's prompt-level loose accuracy, as the vendor reports it (HYPOTHESIS H2 row
"IFBench loose-prompt", H4); strict is reported alongside. A truncated item is wrong whatever the
checkers say (HYPOTHESIS C7); its checker results are still recorded.

The response handed to IFBench is the post-reasoning answer exactly as split (scorers/reasoning.py),
not stripped: what a vLLM reasoning parser returns as `content` (text after `</think>`, including the
template's "\\n\\n"). IFBench's loose variants strip and drop the first / last line, so loose is
insensitive to that separator; strict can be (e.g. format:no_whitespace).

Paths (one convention, all overridable): EXP036_MODELS (default ~/models/exp036), EXP036_DATA (default
$EXP036_MODELS/data). The IFBench python is $EXP036_IFBENCH_PY, else <models>/ifbench-venv/bin/python,
else the ifbench-venv next to the data directory; NLTK data is $EXP036_NLTK_DATA, else
<models>/nltk_data, else the nltk_data next to the data directory (on the mini both sit next to
~/models/exp036-mini/data).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "ifbench_driver.py"

IFBENCH_SRC_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"  # assets.json pin of allenai/IFBench
TIMEOUT_S = 1800


# ---------------------------------------------------------------------------------------------------
# Paths


def models_dir() -> Path:
    return Path(os.environ.get("EXP036_MODELS", "~/models/exp036")).expanduser()


def data_dir() -> Path:
    env = os.environ.get("EXP036_DATA")
    return Path(env).expanduser() if env else models_dir() / "data"


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def default_ifbench_src() -> Path:
    return data_dir() / "IFBench-src"


def default_ifbench_test() -> Path:
    return data_dir() / "IFBench_test"


def default_python() -> Path:
    env = os.environ.get("EXP036_IFBENCH_PY")
    if env:
        return Path(env).expanduser()
    rel = Path("ifbench-venv") / "bin" / "python"
    found = _first_existing([models_dir() / rel, data_dir().parent / rel])
    if found is None:
        raise FileNotFoundError(
            "no IFBench venv: set EXP036_IFBENCH_PY, or build it as in ASSETS.md "
            f"(looked in {models_dir() / rel} and {data_dir().parent / rel})"
        )
    return found


def default_nltk_data() -> Path:
    env = os.environ.get("EXP036_NLTK_DATA")
    if env:
        return Path(env).expanduser()
    found = _first_existing([models_dir() / "nltk_data", data_dir().parent / "nltk_data"])
    if found is None:
        raise FileNotFoundError(
            "no NLTK data for IFBench: set EXP036_NLTK_DATA, or fetch it as in ASSETS.md "
            f"(looked in {models_dir() / 'nltk_data'} and {data_dir().parent / 'nltk_data'})"
        )
    return found


# ---------------------------------------------------------------------------------------------------
# IFBench_test


def load_ifbench_test(path: str | os.PathLike | None = None) -> dict[str, dict]:
    """key -> {key, prompt, instruction_id_list, kwargs} from the local IFBench_test copy.

    Accepts the HF dataset directory (its data/*.parquet; pyarrow is imported lazily), a .parquet file or
    an IFBench-format .jsonl. Keys are strings. kwargs are kept exactly as stored (the parquet copy holds
    every kwarg name with None for the unused ones; IFBench's strict pass drops those).
    """
    p = Path(path).expanduser() if path is not None else default_ifbench_test()
    if p.is_dir():
        parquet = sorted(p.glob("data/*.parquet")) or sorted(p.glob("*.parquet"))
        jsonl = sorted(p.glob("*.jsonl")) + sorted(p.glob("data/*.jsonl"))
        if parquet:
            rows = _read_parquet(parquet)
        elif jsonl:
            rows = _read_jsonl(jsonl[0])
        else:
            raise FileNotFoundError(f"no parquet or jsonl IFBench_test file under {p}")
    elif p.suffix == ".parquet":
        rows = _read_parquet([p])
    else:
        rows = _read_jsonl(p)
    out: dict[str, dict] = {}
    for r in rows:
        key = str(r["key"])
        if key in out:
            raise ValueError(f"duplicate IFBench key {key}")
        out[key] = {
            "key": key,
            "prompt": r["prompt"],
            "instruction_id_list": list(r["instruction_id_list"]),
            "kwargs": [dict(k) if k is not None else {} for k in r["kwargs"]],
        }
    return out


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_parquet(paths: list[Path]) -> list[dict]:
    import pyarrow.parquet as pq  # lazy: only the kit venvs carry pyarrow

    rows: list[dict] = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


# ---------------------------------------------------------------------------------------------------
# Provenance


def ifbench_provenance(ifbench_src: str | os.PathLike | None = None) -> dict:
    """The checkout's commit (read from .git, no subprocess) and the sha256 of the files the checkers use."""
    src = Path(ifbench_src).expanduser() if ifbench_src is not None else default_ifbench_src()
    files = ["evaluation_lib.py", "ifbench/instructions.py", "ifbench/instructions_registry.py",
             "ifbench/instructions_util.py", "ifbench/classic_instructions.py", "uv.lock"]
    return {
        "commit": _git_head(src),
        "commit_pinned": IFBENCH_SRC_COMMIT,
        "files_sha256": {f: _sha256(src / f) for f in files if (src / f).is_file()},
        "driver_sha256": _sha256(DRIVER),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(src: Path) -> str | None:
    head = src / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    name = ref[5:]
    loose = src / ".git" / name
    if loose.is_file():
        return loose.read_text().strip()
    packed = src / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == name:
                return parts[0]
    return None


# ---------------------------------------------------------------------------------------------------
# Scoring


def check_responses(
    responses: Mapping[str, str],
    *,
    ifbench_src: str | os.PathLike | None = None,
    ifbench_test: str | os.PathLike | Mapping[str, dict] | None = None,
    python: str | os.PathLike | None = None,
    nltk_data: str | os.PathLike | None = None,
    require_pinned_commit: bool = True,
) -> dict[str, dict]:
    """key -> {instruction_id_list, strict, loose, prompt_strict, prompt_loose, has_response}.

    `responses` maps an IFBench key to the post-reasoning answer text. Every key must be in IFBench_test.
    `ifbench_test` may also be an already-loaded mapping (as returned by load_ifbench_test).
    """
    src = Path(ifbench_src).expanduser() if ifbench_src is not None else default_ifbench_src()
    if not (src / "evaluation_lib.py").is_file():
        raise FileNotFoundError(f"no IFBench checkout at {src} (evaluation_lib.py missing)")
    if require_pinned_commit:
        head = _git_head(src)
        if head != IFBENCH_SRC_COMMIT:
            raise RuntimeError(f"IFBench checkout at {src} is at {head}, not the pinned {IFBENCH_SRC_COMMIT}")
    rows = ifbench_test if isinstance(ifbench_test, Mapping) else load_ifbench_test(ifbench_test)
    unknown = sorted(set(responses) - set(rows))
    if unknown:
        raise KeyError(f"responses for keys not in IFBench_test: {unknown[:10]}")
    if not responses:
        return {}
    py = Path(python).expanduser() if python is not None else default_python()
    nltk = Path(nltk_data).expanduser() if nltk_data is not None else default_nltk_data()

    keys = sorted(responses, key=_key_order)
    with tempfile.TemporaryDirectory(prefix="exp036_ifbench_") as tmp:
        tmpd = Path(tmp)
        inputs, resp, out = tmpd / "inputs.jsonl", tmpd / "responses.jsonl", tmpd / "out.json"
        with open(inputs, "w", encoding="utf-8") as f:
            for k in keys:
                r = rows[k]
                f.write(json.dumps({"key": r["key"], "prompt": r["prompt"],
                                    "instruction_id_list": r["instruction_id_list"],
                                    "kwargs": r["kwargs"]}, ensure_ascii=False) + "\n")
        with open(resp, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps({"prompt": rows[k]["prompt"], "response": responses[k]},
                                   ensure_ascii=False) + "\n")
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmpd),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "NLTK_DATA": str(nltk),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
        }
        proc = subprocess.run(
            [str(py), "-s", "-B", str(DRIVER), "--ifbench-src", str(src), "--inputs", str(inputs),
             "--responses", str(resp), "--out", str(out)],
            cwd=tmpd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"IFBench driver failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        payload = json.loads(out.read_text(encoding="utf-8"))

    # The checkers must have come from the checkout we were given (in place), not from elsewhere.
    lib = Path(payload["evaluation_lib"]).resolve()
    if lib != (src / "evaluation_lib.py").resolve():
        raise RuntimeError(f"IFBench driver imported evaluation_lib from {lib}, not from {src}")
    results = payload["results"]
    if sorted(results, key=_key_order) != keys:
        raise RuntimeError("IFBench driver returned a different key set")
    return {k: results[k] for k in keys}


def _key_order(k: str):
    return (0, int(k)) if str(k).isdigit() else (1, str(k))


def score_answers(
    answers: Mapping[str, str],
    truncated: Mapping[str, bool] | None = None,
    **kwargs,
) -> dict[str, dict]:
    """key -> {extracted, correct, correct_strict, truncated, parse_status} from post-reasoning answers.

    extracted = {"instruction_id_list", "loose", "strict"} (the per-instruction checker results);
    correct = prompt-level loose and not truncated; parse_status "empty_answer" when the answer is blank
    (IFBench then fails every instruction), "truncated" when truncated, else "ok".
    """
    truncated = truncated or {}
    checked = check_responses(answers, **kwargs)
    out: dict[str, dict] = {}
    for k, r in checked.items():
        is_trunc = bool(truncated.get(k, False))
        if is_trunc:
            status = "truncated"
        elif not answers[k].strip():
            status = "empty_answer"
        else:
            status = "ok"
        out[k] = {
            "extracted": {"instruction_id_list": r["instruction_id_list"], "loose": r["loose"], "strict": r["strict"]},
            "correct": bool(r["prompt_loose"]) and not is_trunc,
            "correct_strict": bool(r["prompt_strict"]) and not is_trunc,
            "truncated": is_trunc,
            "parse_status": status,
        }
    return out


def score(raw_jsonl, ifbench_src=None, ifbench_test=None, **kwargs) -> list[dict]:
    """BUILD_SPEC §5.6 signature: score one raw IFBench JSONL file into score records.

    Delegates to scorers.score_all, which splits off the reasoning and calls `score_answers`.
    """
    from scorers import score_all  # local import: score_all imports this module

    return score_all.score_ifbench_file(raw_jsonl, ifbench_src=ifbench_src, ifbench_test=ifbench_test, **kwargs)
