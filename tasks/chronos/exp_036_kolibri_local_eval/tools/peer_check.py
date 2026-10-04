"""Peer check (BUILD_SPEC §5.9 `tools/peer_check.py`; HYPOTHESIS C3, C9,
"Peers are verified, not gated"; RUNBOOK step 9).

Two modes.

1. Build-time parity (on the build host, where the upstream files exist):

       peer_check.py --parity [--mlx-root DIR] [--upstream-root DIR] [--out FILE]

   For each peer family, the mlx-community folders (8-bit and 4-bit) against
   the pinned upstream repo (assets.json `mini_build_assets`): sha256 and byte
   parity of tokenizer.json, tokenizer_config.json, chat_template.jinja,
   generation_config.json and config.json; token-id parity on fixed EN/DE
   strings (`FIXED_STRINGS`, our own text) through each side's own tokenizer;
   template rendering parity on fixed conversations × kwargs, rendered with
   the same tokenizer object so only the template differs; and the token ids
   of each upstream render under both tokenizers. Differences are recorded,
   never fixed here. Default output: tools/peer_parity_build.json (committed;
   it holds hashes, counts and positions, no file content).

2. Run-host check of the arms (RUNBOOK step 9):

       peer_check.py --arms G8,G4,Q36-8,Q36-4,Q38-8,Q38-4 [--checks ...]

   Per arm: the tokenizer and template files have the sha256 the build-time
   parity covered; the template equals the committed runner/templates copy;
   config quantization {group_size 64, bits, affine} and its per-module
   overrides (the routers); EOS ids; parameters by config arithmetic within
   2 % of the HYPOTHESIS totals. With weights: a strict text-only load
   (mlx_lm load_model, strict=True) whose logical parameter count equals the
   config arithmetic within 2 %; NLL and bits per byte on T1–T6. Per family
   with both bit-widths loaded together: NLL(8) ≤ NLL(4) + 0.02 nats/token and
   mean KL(8‖4) < 0.2. For G8 and Q36-8, the batched-path check: batched-path
   parity (B = 8 vs B = 1, teacher-forced, mixed lengths 37–1,100 with mid-run
   admission, under the G5 noise-floor rule) through the gate's G5 functions,
   and the greedy answer-flip rate on 30 MMLU-ProX full EN items with thinking
   off (≤ 2 %). Writes results/peers_<UTC>.json with t_start / t_end. Per arm:
   `ok`; `B=1` (only the batched-path check failed, its parity or its greedy
   flip rate: the plan rule runs every cell of the arm at B = 1);
   `speed-only` (Amendment 6, below: no quality cell and not in H8's peer
   median; H1, the speed cells and the B4 ladder keep the arm); or `fail`
   (anything else failed, or a check could not run: dropped by amendment
   before any scored run).

Attribution by a bf16 reference (Amendment 6). The family rule compares a
family's two builds with each other, so when it fails it cannot say which
build is unfaithful. When it fails and tools/fidelity_reference.json (in the
TOOLS tree) pins a committed record of the family's builds against an
unquantised reference, each arm is judged on its own: the record must have the
pinned sha256 and rows for every arm of the family on exactly the pinned
texts, and each row must agree with this run's chat-wrapped NLL of that arm
(token count exact, NLL within REFERENCE_NLL_TOL); then an arm whose
token-weighted KL(reference ‖ arm) is < KL_MAX passes fidelity (its verdict
follows its other checks, e.g. the batched path gives `B=1`), and an arm with
KL ≥ KL_MAX is `speed-only`. The attribution is recorded in
`families.<f>.fidelity_attribution` and `arms.<a>.fidelity_reference`; the
pin and record hashes in `fidelity_reference`. Without a pin for the family,
or with a pin that cannot be read or does not match the run, the registered
consequence stands (both arms `fail`), and
`families.<f>.fidelity_attribution` says why ({"used": false, "reason"} or
the mismatches).

Texts (Amendment 5, Andrei 2026-10-04, "Chat-wrapped"). The NLL(8) / KL(8‖4)
rule reads each gate text as the assistant turn after the fixed user message
"Write a text.", rendered through the model's own chat template with thinking
off (bench/kl_8v4.chat_wrapper, which renders with runner.chat.render); only
the text's tokens are scored. Its numbers are `families.<f>.fidelity` and
`arms.<a>.nll_chat`, and `families.<f>.wrapper` records the wrapper. The
raw-text NLL and bits per byte (the text after the context-start token) are
still computed and stay where the gate's G3 reads them, `arms.<a>.nll`;
`families.<f>.fidelity_raw_text` is descriptive. The batched-path parity
teacher-forces the same wrapper followed by the T1–T4 token stream, and every
one of its prompts starts with the wrapper (`batched_path.wrapper`). The
greedy flip renders its MMLU items through runner.chat already.

The teacher-forcing and KL arithmetic are bench/kl_8v4.py's (the H8 code),
used lazily, so the peer check and H8 compute NLL and KL the same way.

Exit codes: 0 every arm ok; 2 some arm only B=1 or speed-only; 1 a failure or
a check that could not run on the run host.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common
from tools.redact import HOST_LABEL, redact_path

common.set_offline_env()

SCHEMA = "exp036 peer check v1"
PARITY_SCHEMA = "exp036 peer parity v1"
PARITY_FILE = common.TOOLS_DIR / "peer_parity_build.json"
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json")
PARITY_BOUND_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

FAMILIES = {
    "gemma4": {"arms": ("G8", "G4"), "upstream_repo": "google/gemma-4-26B-A4B-it", "published_total": 25.2e9,
               "template": "gemma4.jinja"},
    "qwen3_6": {"arms": ("Q36-8", "Q36-4"), "upstream_repo": "Qwen/Qwen3.6-35B-A3B", "published_total": 34.7e9,
                "template": "qwen3_6.jinja"},
    "qwen3_8": {"arms": ("Q38-8", "Q38-4"), "upstream_repo": "Qwen/Qwen3.8-27B", "published_total": 27.0e9,
                "template": "qwen3_8.jinja"},
}
ARM_DIRS = {
    "G8": "gemma-4-26b-a4b-it-8bit", "G4": "gemma-4-26b-a4b-it-4bit",
    "Q36-8": "Qwen3.6-35B-A3B-8bit", "Q36-4": "Qwen3.6-35B-A3B-4bit",
    "Q38-8": "Qwen3.8-27B-8bit", "Q38-4": "Qwen3.8-27B-4bit",
}
ARM_BITS = {a: (8 if a.endswith("8") else 4) for a in ARM_DIRS}
BATCHED_PATH_ARMS = ("G8", "Q36-8")
ALL_CHECKS = ("files", "config", "params", "load", "nll", "kl", "batch", "flip")

PARAM_TOLERANCE = 0.02
NLL_MARGIN = 0.02          # NLL(8) <= NLL(4) + 0.02 nats/token
KL_MAX = 0.2               # mean KL(8||4) < 0.2 nats/token
FLIP_MAX = 0.02            # greedy answer-flip rate <= 2 %
FLIP_ITEMS = 30
FLIP_MAX_TOKENS = 4096
# Batched path (HYPOTHESIS "Peers are verified, not gated"): B = 8 with 12 prompts of mixed lengths
# 37–1,100 tokens and staggered max_tokens, so sequences are admitted mid-run (the gate G5 layout,
# gate/textset.py Profile); the noise floor is G5's (prefill 2048 vs 64) on positions 520–1,100.
BATCH_LENGTHS = (37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100)
BATCH_MAX_TOKENS = (48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44)
FLOOR_RANGE = (520, 1100)

# Amendment 6: attribution of a failed family rule by a pinned bf16 reference record.
FAMILY_RULE_PROBLEM = "NLL(8) / KL(8‖4) rule failed"
SPEED_ONLY = "speed-only"
FIDELITY_REFERENCE_FILE = common.TOOLS_DIR / "fidelity_reference.json"
FIDELITY_REFERENCE_SCHEMA = "exp036 fidelity reference v1"
REFERENCE_NLL_TOL = NLL_MARGIN      # the record's arm NLL vs this run's, nats/token (the registered NLL margin)
REFERENCE_FIELDS = {"arm": "model", "text": "text", "tokens": "n_tokens", "kl": "kl_bf16_model",
                    "nll_arm": "nll_model", "nll_ref": "nll_bf16"}
_TEXT_ID = re.compile(r"^(T\d+)(?:[_.]|$)")

# Our own EN/DE strings for token-id parity (no third-party text).
FIXED_STRINGS = (
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Größenänderung, Straße, Äpfel, Öl, Übermut — ß und ẞ.",
    "Donaudampfschifffahrtsgesellschaftskapitänsmützenabzeichen",
    "Am 3. Oktober 2026 kostete es 1.234,56 € (inkl. 19 % MwSt.).",
    "On 2026-10-03 it cost $1,234.56 (incl. 19% VAT).",
    "3.14159265358979 × 2 = 6.28318530717958",
    "  leading spaces, double  spaces and a trailing space ",
    "line one\nline two\n\nline four\r\nwindows line\tand a tab",
    "„Anführungszeichen“ und »Guillemets« und 'single' and \"double\"",
    "Ellipsis… em dash — en dash – non-breaking space and zero​width",
    "def f(x: int) -> int:\n    return x ** 2  # comment",
    "SELECT name FROM users WHERE id = 42;",
    "https://localfirstai.eu/posts/2026/10/03/kolibri?ref=exp_036#results",
    "Emoji: 🙂 🚀 👩‍💻 and flags 🇩🇪 🇪🇸",
    "数学 と 日本語 と 한국어 — mixed scripts",
    "<think>reasoning</think> answer",
    "<|im_start|>user\nhello<|im_end|>",
    "<start_of_turn>user\nhello<end_of_turn>",
    "The answer is (B).",
    "Die Antwort ist (C).",
    "Antwort: $\\boxed{204}$",
    "Let x = \\frac{a}{b} and y = \\sqrt{x^2 + 1}.",
    "Wie viele Bundesländer hat Deutschland? Sechzehn.",
    "Bitte antworte auf Deutsch und begründe kurz.",
    "Please answer in English and explain briefly.",
    "0123456789 00 000 0000 12345678901234567890",
    "ALL CAPS SENTENCE WITH ACRONYMS LIKE MLX, GPU, KV AND MOE.",
    "tab\tseparated\tvalues\t1\t2\t3",
    "Ein Satz mit Bindestrich-Komposita, z. B. E-Mail-Adresse und Open-Source-Software.",
)

_TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "Current time in a time zone",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]
TEMPLATE_CONVERSATIONS = (
    ("user_en", [{"role": "user", "content": "What is the capital of France?"}], None),
    ("system_user_de", [{"role": "system", "content": "Antworte kurz."},
                        {"role": "user", "content": "Wie heißt die Hauptstadt von Österreich?"}], None),
    ("multi_turn", [{"role": "user", "content": "Name a prime number."},
                    {"role": "assistant", "content": "Seven."},
                    {"role": "user", "content": "And an even one?"}], None),
    ("assistant_reasoning", [{"role": "user", "content": "2+2?"},
                             {"role": "assistant", "content": "4", "reasoning_content": "Two plus two is four."},
                             {"role": "user", "content": "3+3?"}], None),
    ("tools", [{"role": "user", "content": "What time is it in Madrid?"}], _TOOLS),
)
TEMPLATE_KWARGS = (
    ("default", {}),
    ("thinking_on", {"enable_thinking": True}),
    ("thinking_off", {"enable_thinking": False}),
)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def family_of(arm: str) -> str:
    for fam, spec in FAMILIES.items():
        if arm in spec["arms"]:
            return fam
    raise ValueError(f"not a peer arm: {arm!r}")


def upstream_pins(assets: dict | None = None) -> dict:
    """{repo: [revision, ...]} of the build-time upstream files (assets.json)."""
    assets = assets or common.load_assets()
    for item in assets.get("mini_build_assets", {}).get("items", []):
        if isinstance(item.get("repos"), dict):
            return item["repos"]
    return {}


def upstream_dir(root: Path, repo: str, rev: str) -> Path:
    return Path(root) / f"{repo.split('/', 1)[1]}@{rev[:8]}"


def _sha(p: Path) -> str | None:
    return common.sha256_file(p) if p.is_file() else None


def file_hashes(d: Path) -> dict:
    return {name: _sha(Path(d) / name) for name in FILES}


# --------------------------------------------------------------------------
# Tokenizer and template parity
# --------------------------------------------------------------------------

def _tokenizer(d: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(d), local_files_only=True)


def _first_diff(a, b) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def token_parity(tok_a, tok_b, strings=FIXED_STRINGS) -> dict:
    mismatches = []
    roundtrip_fail = 0
    for i, s in enumerate(strings):
        a = list(tok_a.encode(s, add_special_tokens=False))
        b = list(tok_b.encode(s, add_special_tokens=False))
        if a != b:
            mismatches.append({"string": i, "len_a": len(a), "len_b": len(b), "first_diff": _first_diff(a, b)})
        if tok_a.decode(a) != s:
            roundtrip_fail += 1
    return {"n": len(strings), "mismatches": mismatches, "decode_roundtrip_differs": roundtrip_fail,
            "strings_sha256": common.sha256_bytes(json.dumps(list(strings), ensure_ascii=False).encode())}


def _render(tok, template: str, messages, tools, kwargs: dict, add_generation_prompt: bool = True):
    try:
        out = tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False,
                                      add_generation_prompt=add_generation_prompt, **kwargs)
        return {"ok": True, "text": out}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def template_parity(tok_mlx, tok_up, tpl_mlx: str, tpl_up: str) -> dict:
    """Render every case with both templates through the same tokenizer; and
    the upstream render's ids under both tokenizers."""
    cases, mism, ids_mism = 0, [], []
    for name, msgs, tools in TEMPLATE_CONVERSATIONS:
        for kname, kw in TEMPLATE_KWARGS:
            cases += 1
            a = _render(tok_mlx, tpl_mlx, msgs, tools, kw)
            b = _render(tok_mlx, tpl_up, msgs, tools, kw)
            equal = (a["ok"] and b["ok"] and a["text"] == b["text"]) or (
                not a["ok"] and not b["ok"] and a["error"] == b["error"])
            if not equal:
                entry = {"case": f"{name}/{kname}", "mlx_ok": a["ok"], "upstream_ok": b["ok"]}
                if a["ok"] and b["ok"]:
                    entry["first_diff_char"] = _first_diff(a["text"], b["text"])
                    entry["len_mlx"], entry["len_upstream"] = len(a["text"]), len(b["text"])
                else:
                    entry["errors"] = [a.get("error"), b.get("error")]
                mism.append(entry)
            if b["ok"]:
                ia = list(tok_mlx.encode(b["text"], add_special_tokens=False))
                ib = list(tok_up.encode(b["text"], add_special_tokens=False))
                if ia != ib:
                    ids_mism.append({"case": f"{name}/{kname}", "first_diff": _first_diff(ia, ib)})
    return {"cases": cases, "render_mismatches": mism, "rendered_ids_mismatches": ids_mism}


def parity_family(family: str, mlx_root: Path, upstream_root: Path, pins: dict) -> dict:
    spec = FAMILIES[family]
    revs = pins.get(spec["upstream_repo"], [])
    if not revs:
        return {"error": f"no upstream pin for {spec['upstream_repo']} in assets.json"}
    pinned = revs[0]
    up = upstream_dir(upstream_root, spec["upstream_repo"], pinned)
    out = {"upstream": {"repo": spec["upstream_repo"], "revision": pinned, "present": up.is_dir(),
                        "files": file_hashes(up) if up.is_dir() else {}}}
    if not up.is_dir():
        out["error"] = f"upstream folder {redact_path(up)} missing"
        return out
    tok_up = _tokenizer(up)
    tpl_up = (up / "chat_template.jinja").read_text(encoding="utf-8")
    committed = common.EXP_DIR / "runner" / "templates" / spec["template"]
    out["committed_template"] = {"file": f"runner/templates/{spec['template']}", "sha256": _sha(committed),
                                 "equal_to_upstream": _sha(committed) == out["upstream"]["files"]["chat_template.jinja"]}
    out["peers"] = {}
    for arm in spec["arms"]:
        d = Path(mlx_root) / ARM_DIRS[arm]
        if not d.is_dir():
            out["peers"][arm] = {"dir": ARM_DIRS[arm], "present": False}
            continue
        files = file_hashes(d)
        tok = _tokenizer(d)
        tpl = (d / "chat_template.jinja").read_text(encoding="utf-8")
        out["peers"][arm] = {
            "dir": ARM_DIRS[arm], "present": True, "files": files,
            "byte_equal_to_upstream": {k: files[k] == out["upstream"]["files"].get(k) for k in FILES},
            "default_template_is_chat_template_jinja": getattr(tok, "chat_template", None) == tpl,
            "token_ids": token_parity(tok, tok_up),
            "template": template_parity(tok, tok_up, tpl, tpl_up),
        }
    # Informational: later upstream revisions (e.g. Gemma main), not the pin.
    extra = {}
    for rev in revs[1:]:
        d2 = upstream_dir(upstream_root, spec["upstream_repo"], rev)
        if d2.is_dir():
            first_arm = next((a for a in spec["arms"] if out["peers"].get(a, {}).get("present")), None)
            if first_arm:
                d = Path(mlx_root) / ARM_DIRS[first_arm]
                tok = _tokenizer(d)
                extra[rev] = {
                    "files": file_hashes(d2),
                    "template_vs_mlx": template_parity(
                        tok, _tokenizer(d2), (d / "chat_template.jinja").read_text(encoding="utf-8"),
                        (d2 / "chat_template.jinja").read_text(encoding="utf-8")),
                }
    if extra:
        out["other_upstream_revisions"] = extra
    return out


def build_parity(mlx_root, upstream_root) -> dict:
    pins = upstream_pins()
    rec = {"schema": PARITY_SCHEMA, "utc": common.utc_iso(),
           "method": "transformers AutoTokenizer (local files only); apply_chat_template(tokenize=False, "
                     "add_generation_prompt=True) with each template string through the mlx-community tokenizer",
           "n_fixed_strings": len(FIXED_STRINGS),
           "template_cases": [f"{n}/{k}" for n, _, _ in TEMPLATE_CONVERSATIONS for k, _ in TEMPLATE_KWARGS],
           "families": {}}
    for fam in FAMILIES:
        rec["families"][fam] = parity_family(fam, Path(mlx_root), Path(upstream_root), pins)
    return rec


# --------------------------------------------------------------------------
# Run-host checks
# --------------------------------------------------------------------------

def peer_root() -> Path:
    return common.models_dir()


def config_check(cfg: dict, bits: int) -> dict:
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    base = {k: q.get(k) for k in ("group_size", "bits", "mode")}
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    summary: dict = {}
    for k, v in overrides.items():
        leaf = ".".join(p for p in k.split(".") if not p.isdigit()).split("layers.")[-1]
        summary.setdefault(leaf, set()).add((v.get("bits"), v.get("group_size")))
    ok = base == {"group_size": 64, "bits": bits, "mode": "affine"}
    return {"base": base, "expected": {"group_size": 64, "bits": bits, "mode": "affine"}, "ok": ok,
            "n_overrides": len(overrides),
            "overrides": {k: sorted([list(x) for x in v]) for k, v in sorted(summary.items())}}


def eos_info(d: Path) -> dict:
    gen = json.loads((d / "generation_config.json").read_text()) if (d / "generation_config.json").is_file() else {}
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
    g = gen.get("eos_token_id")
    c = cfg.get("eos_token_id")
    return {"generation_config": g if isinstance(g, list) else ([g] if g is not None else []),
            "config": c if isinstance(c, list) else ([c] if c is not None else [])}


def static_check(arm: str, root: Path | None = None, parity: dict | None = None) -> dict:
    """Everything that needs no weights."""
    from tools import params as P

    fam = family_of(arm)
    d = Path(root or peer_root()) / ARM_DIRS[arm]
    rec = {"arm": arm, "family": fam, "dir": redact_path(d), "present": d.is_dir(), "problems": []}
    if not d.is_dir():
        rec["problems"].append("model directory missing")
        return rec
    files = file_hashes(d)
    rec["files"] = files
    if parity is None and PARITY_FILE.is_file():
        parity = json.loads(PARITY_FILE.read_text(encoding="utf-8"))
    if parity:
        ref = parity.get("families", {}).get(fam, {}).get("peers", {}).get(arm, {}).get("files", {})
        same = {k: files.get(k) == ref.get(k) for k in PARITY_BOUND_FILES}
        rec["same_files_as_build_parity"] = same
        if not all(same.values()):
            rec["problems"].append("tokenizer/template files differ from those of the build-time parity check")
        fam_par = parity["families"].get(fam, {})
        p = fam_par.get("peers", {}).get(arm, {})
        rec["build_parity"] = {
            "token_id_mismatches": len(p.get("token_ids", {}).get("mismatches", [])),
            "template_render_mismatches": len(p.get("template", {}).get("render_mismatches", [])),
            "byte_equal_to_upstream": p.get("byte_equal_to_upstream"),
        }
    else:
        rec["problems"].append("no build-time parity record (tools/peer_parity_build.json)")
    committed = common.EXP_DIR / "runner" / "templates" / FAMILIES[fam]["template"]
    rec["template_equals_committed"] = _sha(committed) == files.get("chat_template.jinja") if committed.is_file() else None
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    rec["config"] = config_check(cfg, ARM_BITS[arm])
    if not rec["config"]["ok"]:
        rec["problems"].append(f"quantization {rec['config']['base']} != {rec['config']['expected']}")
    rec["eos"] = eos_info(d)
    try:
        cp = P.config_params(d)
        pub = FAMILIES[fam]["published_total"]
        cp["published_total"] = pub
        cp["rel_diff_vs_published"] = abs(cp["total"] - pub) / pub
        rec["params_config"] = cp
        if cp["rel_diff_vs_published"] > PARAM_TOLERANCE:
            rec["problems"].append(f"config parameter count {cp['total']:,} not within 2 % of {pub:,.0f}")
    except Exception as e:
        rec["params_config"] = {"error": f"{type(e).__name__}: {e}"}
        rec["problems"].append("config arithmetic failed")
    return rec


def _load(d: Path):
    from mlx_lm.utils import load_model

    model, _cfg = load_model(Path(d), strict=True)
    return model


def _release():
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def text_stats(m8, m4, d8: Path, texts: dict, prompt_ids: list | None = None) -> dict:
    """Per text: tokens, bytes, NLL sums of each model, KL sum (m4 may be None).

    prompt_ids None: the raw text after the context-start token (the G3
    numbers). Otherwise the text is teacher-forced after prompt_ids (the
    Amendment 5 chat wrapper) and only its own tokens are scored."""
    from bench import kl_8v4 as K   # sibling area; lazy

    pre = [K.context_prefix(d8)["id"]] if prompt_ids is None else [int(i) for i in prompt_ids]
    out = {}
    for t, rec in sorted(texts.items()):
        ids, _ = K.tokenize(d8 / "tokenizer.json", rec["text"])
        inputs, score_from = K.scored_input(pre, ids)
        if m4 is not None:
            stats, _ = K.compare_loaded(m8, m4, inputs, ids, score_from=score_from)
            out[t] = {"tokens": len(ids), "bytes": rec["bytes"], "nll8": float(stats["nll8"].sum()),
                      "nll4": float(stats["nll4"].sum()), "kl": float(stats["kl"].sum()),
                      "top1_agree": float(stats["agree"].mean())}
        else:
            import mlx.core as mx
            import numpy as np

            nll = 0.0
            for t0, logits, _ in K.scored_chunks(m8, inputs, score_from):
                lp = K.logprobs(logits, True)
                tg = mx.array(np.asarray(ids[t0:t0 + lp.shape[0]], dtype=np.int32))[:, None]
                v = -mx.take_along_axis(lp, tg, axis=-1)
                mx.eval(v)
                nll += float(np.array(v).sum())
            out[t] = {"tokens": len(ids), "bytes": rec["bytes"], "nll8": nll}
    return out


def nll_summary(stats: dict, key: str) -> dict:
    per_text = {}
    for t, s in stats.items():
        if key in s:
            per_text[t] = {"nll_per_token": s[key] / s["tokens"], "bpb": s[key] / math.log(2) / s["bytes"],
                           "tokens": s["tokens"], "bytes": s["bytes"]}
    tok = sum(v["tokens"] for v in per_text.values())
    byt = sum(v["bytes"] for v in per_text.values())
    tot = sum(stats[t][key] for t in per_text)
    return {"per_text": per_text, "mean_nll_per_token": tot / tok if tok else None,
            "mean_bpb": tot / math.log(2) / byt if byt else None,
            "per_text_bpb": {t: v["bpb"] for t, v in per_text.items()}}


def fidelity_rule(stats: dict) -> dict:
    """NLL(8) ≤ NLL(4) + 0.02 nats/token and mean KL(8‖4) < 0.2."""
    tok = sum(s["tokens"] for s in stats.values())
    n8 = sum(s["nll8"] for s in stats.values()) / tok
    n4 = sum(s["nll4"] for s in stats.values()) / tok
    kl = sum(s["kl"] for s in stats.values()) / tok
    return {"nll8_per_token": n8, "nll4_per_token": n4, "kl_8v4_per_token": kl,
            "nll_ok": n8 <= n4 + NLL_MARGIN, "kl_ok": kl < KL_MAX}


def flip_rate(a: list, b: list) -> dict:
    """Share of items whose extracted answer differs between two runs
    (None counts as an answer, so a parse failure on one side is a flip)."""
    if len(a) != len(b):
        raise ValueError("flip_rate: runs differ in length")
    flips = sum(1 for x, y in zip(a, b) if x != y)
    n = len(a)
    return {"n": n, "flips": flips, "rate": flips / n if n else None, "ok": (flips / n <= FLIP_MAX) if n else False}


# --------------------------------------------------------------------------
# Amendment 6: a failed family rule attributed to a build by a bf16 reference
# --------------------------------------------------------------------------

class ReferenceError(ValueError):
    """A pinned fidelity reference that cannot be used (malformed pin, missing record, other sha256, rows)."""


def _rel(p: Path, exp_dir: Path) -> str:
    try:
        return Path(p).resolve().relative_to(Path(exp_dir).resolve()).as_posix()
    except ValueError:
        return redact_path(p)


def load_fidelity_reference(path=None, exp_dir=None) -> dict:
    """The pinned reference records, checked; {} when there is no pin file (the registered behaviour).

    Returns {"pin": {"path", "sha256"}, "families": {family: {"record", "record_sha256", "texts",
    "reference_model", "measurement", "arms": {arm: {"tokens", "kl_ref_per_token", "nll_arm_per_token",
    "nll_ref_per_token", "per_text": {T: {"tokens", "kl", "nll_arm", "nll_ref"}}}}}}}: per arm, the per-text
    means of the record and their token-weighted means over the pinned texts. Raises ReferenceError when the
    pin is malformed, names an unknown family, or its record is missing, has another sha256, or lacks a row of
    some arm of the family on some pinned text (or has one twice, or one outside the pinned texts)."""
    exp_dir = Path(exp_dir or common.EXP_DIR)
    path = Path(path) if path is not None else FIDELITY_REFERENCE_FILE
    if not path.is_file():
        return {}
    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReferenceError(f"{path.name}: not JSON ({e})") from None
    if pin.get("schema") != FIDELITY_REFERENCE_SCHEMA:
        raise ReferenceError(f"{path.name}: schema {pin.get('schema')!r}, expected {FIDELITY_REFERENCE_SCHEMA!r}")
    out = {"pin": {"path": _rel(path, exp_dir), "sha256": common.sha256_file(path)}, "families": {}}
    for fam, spec in sorted((pin.get("families") or {}).items()):
        if fam not in FAMILIES:
            raise ReferenceError(f"{path.name}: unknown family {fam!r}")
        try:
            rp = exp_dir / spec["record"]
            want_sha, texts = spec["record_sha256"], [str(t) for t in spec["texts"]]
        except (KeyError, TypeError) as e:
            raise ReferenceError(f"{path.name}: {fam}: missing {e}") from None
        if not rp.is_file():
            raise ReferenceError(f"{fam}: reference record {spec['record']} is missing")
        sha = common.sha256_file(rp)
        if sha != want_sha:
            raise ReferenceError(f"{fam}: reference record {spec['record']} has sha256 {sha[:12]}…, "
                                 f"the pin says {str(want_sha)[:12]}…")
        f = {**REFERENCE_FIELDS, **(spec.get("fields") or {})}
        try:
            body = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ReferenceError(f"{fam}: {spec['record']} is not JSON ({e})") from None
        rows = body.get(spec.get("rows", "reference_kl")) if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ReferenceError(f"{fam}: {spec['record']} has no list {spec.get('rows', 'reference_kl')!r}")
        arms = {}
        for arm in FAMILIES[fam]["arms"]:
            per: dict = {}
            for r in rows:
                if not isinstance(r, dict) or r.get(f["arm"]) != arm:
                    continue
                m = _TEXT_ID.match(str(r.get(f["text"], "")))
                if not m or m.group(1) not in texts or m.group(1) in per:
                    raise ReferenceError(f"{fam}: {arm} row for text {r.get(f['text'])!r} is not one of the pinned "
                                         f"texts {texts}, or it repeats")
                try:
                    per[m.group(1)] = {"tokens": int(r[f["tokens"]]), "kl": float(r[f["kl"]]),
                                       "nll_arm": float(r[f["nll_arm"]]), "nll_ref": float(r[f["nll_ref"]])}
                except (KeyError, TypeError, ValueError) as e:
                    raise ReferenceError(f"{fam}: {arm} {m.group(1)}: bad row ({type(e).__name__}: {e})") from None
            if sorted(per) != sorted(texts):
                raise ReferenceError(f"{fam}: {arm} has rows for {sorted(per)}, the pin names {sorted(texts)}")
            n = sum(v["tokens"] for v in per.values())
            if n <= 0:
                raise ReferenceError(f"{fam}: {arm} rows hold no tokens")
            arms[arm] = {"tokens": n,
                         "kl_ref_per_token": sum(v["kl"] * v["tokens"] for v in per.values()) / n,
                         "nll_arm_per_token": sum(v["nll_arm"] * v["tokens"] for v in per.values()) / n,
                         "nll_ref_per_token": sum(v["nll_ref"] * v["tokens"] for v in per.values()) / n,
                         "per_text": dict(sorted(per.items()))}
        out["families"][fam] = {"record": spec["record"], "record_sha256": sha, "texts": texts,
                                "record_commit": spec.get("record_commit"),
                                "reference_model": spec.get("reference_model"),
                                "measurement": spec.get("measurement"), "arms": arms}
    return out


def reference_classes(reference: dict | None = None) -> dict:
    """{arm: "pass" | "speed-only"} by the pinned references' KL alone (no agreement with a run is checked):
    what the reference says of each build it covers. tools/dry_run.py writes its peers record from it."""
    ref = load_fidelity_reference() if reference is None else reference
    return {arm: ("pass" if a["kl_ref_per_token"] < KL_MAX else SPEED_ONLY)
            for f in (ref.get("families") or {}).values() for arm, a in sorted(f["arms"].items())}


def attribute_fidelity(ref_family: dict, nll_chat: dict) -> dict:
    """Judge each arm of a family whose rule failed by its pinned reference (a pure function).

    nll_chat: {arm: this run's arms.<a>.nll_chat (nll_summary of the chat-wrapped texts)}. The reference is
    usable only if every pinned text of every arm agrees with the run: the same token count and an NLL per
    token within REFERENCE_NLL_TOL. Then per arm: "pass" if KL(reference ‖ arm) < KL_MAX, else "speed-only".
    Returns {"usable", "problems", "threshold", "nll_tolerance", "arms": {arm: {"kl_ref_per_token",
    "nll_arm_per_token", "nll_ref_per_token", "tokens", "fidelity", "agreement": {T: {...}}}}}."""
    problems, arms = [], {}
    for arm, a in sorted(ref_family["arms"].items()):
        run_texts = ((nll_chat or {}).get(arm) or {}).get("per_text") or {}
        agree = {}
        for t, r in a["per_text"].items():
            got = run_texts.get(t)
            if got is None:
                problems.append(f"{arm} {t}: no chat-wrapped NLL in this run")
                continue
            diff = float(got["nll_per_token"]) - r["nll_arm"]
            agree[t] = {"tokens_run": int(got["tokens"]), "tokens_reference": r["tokens"],
                        "nll_run": float(got["nll_per_token"]), "nll_reference": r["nll_arm"], "nll_diff": diff}
            if int(got["tokens"]) != r["tokens"]:
                problems.append(f"{arm} {t}: {got['tokens']} chat-wrapped tokens in this run, {r['tokens']} in the "
                                f"reference record")
            elif abs(diff) > REFERENCE_NLL_TOL:
                problems.append(f"{arm} {t}: NLL {float(got['nll_per_token']):.4f} in this run vs {r['nll_arm']:.4f} "
                                f"in the reference record (> {REFERENCE_NLL_TOL})")
        kl = a["kl_ref_per_token"]
        arms[arm] = {"kl_ref_per_token": kl, "nll_arm_per_token": a["nll_arm_per_token"],
                     "nll_ref_per_token": a["nll_ref_per_token"], "tokens": a["tokens"],
                     "fidelity": "pass" if kl < KL_MAX else SPEED_ONLY, "agreement": agree}
    return {"usable": not problems, "problems": problems, "threshold": KL_MAX, "nll_tolerance": REFERENCE_NLL_TOL,
            "arms": arms}


def apply_family_rule_failure(rec: dict, family: str, fam_arms, reference: dict) -> None:
    """The consequence of a failed NLL(8) / KL(8‖4) rule for the family's arms (both bit widths).

    Registered: every arm gets the problem (verdict `fail`). Amendment 6: if `reference` pins this family and its
    record matches the run (attribute_fidelity), an arm whose KL(reference ‖ arm) < KL_MAX gets no problem (its
    verdict follows its other checks) and an arm at or above KL_MAX gets `speed_only_problems` (verdict
    `speed-only`). No pin for the family, a pin that cannot be read, or one that does not match the run leaves the
    registered consequence, and families.<f>.fidelity_attribution says why."""
    frec = rec["families"].setdefault(family, {})
    ref_fam = (reference.get("families") or {}).get(family)
    if ref_fam is None:
        frec["fidelity_attribution"] = {
            "used": False, "amendment": "Amendment 6",
            "reason": (f"the fidelity reference is unusable: {reference['error']}" if reference.get("error")
                       else "no Amendment 6 fidelity reference is pinned for this family: the registered rule "
                            "applies")}
        for a in fam_arms:
            rec["arms"][a]["problems"].append(FAMILY_RULE_PROBLEM)
        return
    att = attribute_fidelity(ref_fam, {a: rec["arms"][a].get("nll_chat") for a in fam_arms})
    frec["fidelity_attribution"] = {
        "used": att["usable"], "amendment": "Amendment 6",
        "reference": {k: ref_fam.get(k) for k in ("record", "record_sha256", "record_commit", "texts",
                                                   "reference_model", "measurement")}
        | {"pin": (reference.get("pin") or {}).get("path"), "pin_sha256": (reference.get("pin") or {}).get("sha256")},
        **{k: att[k] for k in ("threshold", "nll_tolerance", "problems", "arms")},
    }
    if not att["usable"]:
        for a in fam_arms:
            rec["arms"][a]["problems"] += [FAMILY_RULE_PROBLEM, "the Amendment 6 fidelity reference does not match "
                                                                "this run (families.<f>.fidelity_attribution)"]
        return
    for a in fam_arms:
        x = att["arms"][a]
        rec["arms"][a]["fidelity_reference"] = {"kl_ref_per_token": x["kl_ref_per_token"], "threshold": KL_MAX,
                                                "fidelity": x["fidelity"], "record": ref_fam["record"]}
        if x["fidelity"] == SPEED_ONLY:
            rec["arms"][a].setdefault("speed_only_problems", []).append(
                f"{FAMILY_RULE_PROBLEM}; KL(bf16‖{a}) {x['kl_ref_per_token']:.4f} ≥ {KL_MAX} on the Amendment 6 "
                f"reference")


def batched_prompts(wrapper_ids: list, stream: list) -> tuple[list, list]:
    """(base, prompts) of the batched path (Amendment 5): base = the chat
    wrapper followed by the text token stream (the noise floor and
    decode-vs-prefill run on its positions FLOOR_RANGE, all inside the text);
    prompt i = the wrapper followed by stream[50 i : 50 i + n_i - len(wrapper)],
    so every prompt starts with the wrapper and has exactly the registered
    length n_i = BATCH_LENGTHS[i]."""
    w = [int(i) for i in wrapper_ids]
    stream = [int(i) for i in stream]
    if len(w) >= min(BATCH_LENGTHS):
        raise ValueError(f"batched path: the {len(w)}-token wrapper does not fit the shortest prompt "
                         f"({min(BATCH_LENGTHS)} tokens)")
    lo, hi = FLOOR_RANGE
    if len(w) > lo:
        raise ValueError(f"batched path: the {len(w)}-token wrapper reaches the noise-floor positions from {lo}")
    need = max(max(i * 50 + n - len(w) for i, n in enumerate(BATCH_LENGTHS)), hi + 1 - len(w))
    if len(stream) < need:
        raise ValueError(f"batched path: {len(stream)} base tokens, need {need}")
    prompts = [w + stream[i * 50:i * 50 + n - len(w)] for i, n in enumerate(BATCH_LENGTHS)]
    return w + stream, prompts


def batched_path(model, d: Path, arm: str, base_ids: list | None = None, wrapper_ids: list | None = None) -> dict:
    """B = 8 vs B = 1 teacher-forced, under the G5 noise-floor rule; through
    gate/checks/g5_generation.py (the same code as gate G5): floor_and_decode
    gives floor_KL and floor_dis (prefill 2048 vs 64), parity_bound the G5
    bound, batch_parity the batched-vs-single comparison with mid-run
    admission. The token stream is T1-T4 after the Amendment 5 chat wrapper
    (batched_prompts). `base_ids` replaces the T1-T4 stream and `wrapper_ids`
    the rendered wrapper (tests)."""
    from gate.checks import g5_generation as g5   # sibling area; lazy
    from runner import chat

    wrapper_rec = None
    if wrapper_ids is None:
        from bench import kl_8v4 as K

        wrap = K.chat_wrapper(family_of(arm), d)
        wrapper_ids, wrapper_rec = wrap["prompt_ids"], wrap["record"]
    if base_ids is None:
        from bench import kl_8v4 as K

        texts = K.load_texts(K.gate_text_paths())
        base_ids = []
        for t in ("T1", "T2", "T3", "T4"):
            base_ids.extend(K.tokenize(d / "tokenizer.json", texts[t]["text"])[0])
    base, prompts = batched_prompts(wrapper_ids, base_ids)
    lo, hi = FLOOR_RANGE
    th = json.loads((common.EXP_DIR / "gate" / "thresholds.json").read_text())["G5"]
    eos = tuple(chat.eos_ids(d))
    floor = g5.floor_and_decode(model, {"base": base}, {"base": (lo, hi)},
                                tuple(th["noise_floor_prefill_steps"]), log=lambda *_a, **_k: None)
    bound = g5.parity_bound(floor, th)
    par = g5.batch_parity(model, prompts, list(BATCH_MAX_TOKENS), int(th["batch_B"]), eos=eos)
    ok = (par["mean_kl"] <= bound["kl_max"] and par["top1_dis"] <= bound["dis_max"]
          and par["admitted_mid_run"] > 0 and par["max_live"] <= int(th["batch_B"]))
    return {"noise_floor": floor, "parity": par, "kl_bound": bound["kl_max"], "dis_bound": bound["dis_max"],
            "ok": bool(ok), "lengths": list(BATCH_LENGTHS), "max_tokens": list(BATCH_MAX_TOKENS),
            "floor_range": [lo, hi], "eos_ids": list(eos),
            "wrapper": wrapper_rec if wrapper_rec is not None else {
                "prompt_ids": [int(i) for i in wrapper_ids], "note": "wrapper ids passed by the caller"},
            "text": "Amendment 5: chat wrapper + T1-T4 token stream; every prompt starts with the wrapper"}


def greedy_flip(model, d: Path, arm: str) -> dict:
    """Greedy answer-flip rate, B = 8 vs B = 1, 30 MMLU-ProX full EN items, thinking off."""
    from bench import genutil
    from runner import chat
    from scorers import mc
    from tasks import mmlu_prox
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(Path(d))
    fam = family_of(arm)
    items = mmlu_prox.peer_check_items(common.data_dir() / "MMLU-ProX", common.data_dir() / "MMLU-ProX-Lite",
                                       n=FLIP_ITEMS)
    prompts = [chat.render(fam, tok, it.messages, "none")[1] for it in items]
    eos = chat.eos_ids(d)
    runs = {}
    for B in (8, 1):
        outs = genutil.batch_completions(model, prompts, B, FLIP_MAX_TOKENS, eos)
        answers = []
        for o in outs:
            ids = [i for i in o["completion_ids"] if i not in eos]
            answers.append(mc.extract_mmlu_en(tok.decode(ids)))
        runs[B] = {"answers": answers, "truncated": sum(1 for o in outs if o["finish_reason"] == "length")}
    res = flip_rate(runs[8]["answers"], runs[1]["answers"])
    res.update(items=[it.id for it in items], truncated_B8=runs[8]["truncated"], truncated_B1=runs[1]["truncated"],
               max_tokens=FLIP_MAX_TOKENS)
    return res


def _verdict(arm_rec: dict) -> str:
    """`fail` if anything outside the batched-path check failed or a check
    could not run (`problems`); else `speed-only` if the Amendment 6
    reference attributed its family's failed fidelity rule to this build
    (`speed_only_problems`); else `B=1` if the batched-path check failed,
    i.e. its parity or its greedy flip rate (HYPOTHESIS "Peers are verified,
    not gated": "a peer that fails the batched-path check runs at B = 1; a
    peer that fails anything else is dropped"); else `ok`."""
    if arm_rec["problems"]:
        return "fail"
    if arm_rec.get("speed_only_problems"):
        return SPEED_ONLY
    if (arm_rec.get("batched_path_problems") or arm_rec.get("batched_path", {}).get("ok") is False
            or arm_rec.get("greedy_flip", {}).get("ok") is False):
        return "B=1"
    return "ok"


def _reference_for_run(reference: dict | None) -> tuple[dict, dict]:
    """(reference, its summary for the record). None loads the pinned file; any error reading it (a
    ReferenceError, an unreadable file) is kept as {"error": ...}, which leaves the registered consequence
    wherever a family rule fails: a broken pin never stops the peer check and never passes an arm."""
    if reference is None:
        try:
            reference = load_fidelity_reference()
        except Exception as e:  # noqa: BLE001 - recorded; the registered rule applies
            reference = {"error": f"{type(e).__name__}: {e}"[:400]}
    if reference.get("error"):
        return reference, {"status": "unusable", "error": reference["error"]}
    if not reference:
        return reference, {"status": "absent", "note": "no pin file: the registered rule applies to every family"}
    return reference, {"status": "pinned", "pin": reference["pin"]["path"], "pin_sha256": reference["pin"]["sha256"],
                       "families": {f: {"record": v["record"], "record_sha256": v["record_sha256"]}
                                    for f, v in reference["families"].items()},
                       "note": "Amendment 6: read only for a family whose NLL(8) / KL(8‖4) rule fails"}


def run(arms, checks=ALL_CHECKS, root: Path | None = None, reference: dict | None = None) -> dict:
    """reference: the Amendment 6 fidelity reference (load_fidelity_reference()); None loads the pinned
    tools/fidelity_reference.json, {} means none (the registered rule)."""
    t_start = common.utc_iso()
    root = Path(root or peer_root())
    rec = {"schema": SCHEMA, "host": HOST_LABEL, "t_start": t_start, "arms": {}, "families": {},
           "checks": list(checks)}
    reference, rec["fidelity_reference"] = _reference_for_run(reference)
    texts = None
    if "nll" in checks:
        try:
            from bench import kl_8v4 as K

            texts = K.load_texts(K.gate_text_paths())
            rec["texts"] = {t: {"bytes": v["bytes"], "sha256": v["sha256"]} for t, v in texts.items()}
        except Exception as e:
            rec["texts_error"] = f"{type(e).__name__}: {e}"
    for arm in arms:
        rec["arms"][arm] = static_check(arm, root)
    for fam, spec in FAMILIES.items():
        fam_arms = [a for a in spec["arms"] if a in arms and rec["arms"][a].get("present")]
        if not fam_arms or not ({"load", "nll", "kl", "batch", "flip"} & set(checks)):
            continue
        a8 = next((a for a in fam_arms if ARM_BITS[a] == 8), None)
        a4 = next((a for a in fam_arms if ARM_BITS[a] == 4), None)
        models = {}
        frec = rec["families"].setdefault(fam, {})
        for a in (a8, a4):
            if a is None:
                continue
            d = root / ARM_DIRS[a]
            try:
                from tools import params as P

                m = _load(d)
                models[a] = m
                cp = rec["arms"][a].get("params_config", {})
                lp = P.loaded_params(m, cp.get("num_experts"), cp.get("top_k"), cp.get("tied_embeddings", False))
                lp["rel_diff_vs_config"] = abs(lp["total"] - cp["total"]) / cp["total"] if cp.get("total") else None
                rec["arms"][a]["strict_load"] = {"ok": True, "params_loaded": lp}
                if lp["rel_diff_vs_config"] is None or lp["rel_diff_vs_config"] > PARAM_TOLERANCE:
                    rec["arms"][a]["problems"].append("loaded parameter count not within 2 % of config arithmetic")
            except Exception as e:
                rec["arms"][a]["strict_load"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
                rec["arms"][a]["problems"].append("strict text-only load failed")
        try:
            if texts is not None and ("nll" in checks or "kl" in checks):
                from bench import kl_8v4 as K   # sibling area; lazy

                d_ref = root / ARM_DIRS[a8 or a4]
                m_main = models.get(a8) or models.get(a4)
                m_other = models.get(a4) if (a8 in models and a4 in models and "kl" in checks) else None
                if m_main is not None:
                    main_arm = a8 if a8 in models else a4
                    # Raw text after the context-start token: what the gate's G3 reads (arms.<a>.nll).
                    raw = text_stats(m_main, m_other, d_ref, texts)
                    rec["arms"][main_arm]["nll"] = nll_summary(raw, "nll8")
                    # Amendment 5: the same texts as the assistant turn of the fixed chat wrapper.
                    wrap = K.chat_wrapper(fam, d_ref)
                    frec["wrapper"] = {**wrap["record"], "rendered_from": ARM_DIRS[a8 or a4]}
                    if a8 in models and a4 in models:
                        same = K.chat_wrapper(fam, root / ARM_DIRS[a4])["prompt_ids"] == wrap["prompt_ids"]
                        frec["wrapper"]["prompt_ids_4bit_identical"] = same
                        if not same:
                            raise ValueError("the 8-bit and 4-bit chat wrappers render different prompt ids")
                    st = text_stats(m_main, m_other, d_ref, texts, prompt_ids=wrap["prompt_ids"])
                    rec["arms"][main_arm]["nll_chat"] = nll_summary(st, "nll8")
                    if m_other is not None:
                        rec["arms"][a4]["nll"] = nll_summary(raw, "nll4")
                        rec["arms"][a4]["nll_chat"] = nll_summary(st, "nll4")
                        frec["fidelity_raw_text"] = {**fidelity_rule(raw), "used_for_verdict": False,
                                                     "note": "raw text after the context-start token; "
                                                             "descriptive since Amendment 5"}
                        frec["fidelity"] = {**fidelity_rule(st), "texts": "chat-wrapped (Amendment 5)"}
                        if not (frec["fidelity"]["nll_ok"] and frec["fidelity"]["kl_ok"]):
                            # Registered: both arms fail; Amendment 6: a pinned bf16 reference may attribute it.
                            apply_family_rule_failure(rec, fam, (a8, a4), reference)
        except Exception as e:
            frec["nll_error"] = f"{type(e).__name__}: {e}"[:400]
            for a in fam_arms:
                rec["arms"][a]["problems"].append("NLL / KL could not run")
        if a4 in models:
            del models[a4]
            _release()
        if a8 in models and a8 in BATCHED_PATH_ARMS:
            d8 = root / ARM_DIRS[a8]
            # The batched-path check (its parity and its greedy flip rate): a failure here is B = 1, not a drop;
            # a check that could not run is a problem like any other.
            bpp = rec["arms"][a8].setdefault("batched_path_problems", [])
            if "batch" in checks:
                try:
                    bp = batched_path(models[a8], d8, a8)
                    rec["arms"][a8]["batched_path"] = bp
                    if not bp["ok"]:
                        bpp.append("batched-path parity outside the G5 noise-floor bound")
                except Exception as e:
                    rec["arms"][a8]["batched_path"] = {"ok": None, "error": f"{type(e).__name__}: {e}"[:400]}
                    rec["arms"][a8]["problems"].append("batched-path parity could not run")
            if "flip" in checks:
                try:
                    fr = greedy_flip(models[a8], d8, a8)
                    rec["arms"][a8]["greedy_flip"] = fr
                    if not fr["ok"]:
                        bpp.append(f"greedy flip rate {fr['rate']:.3f} > {FLIP_MAX}")
                except Exception as e:
                    rec["arms"][a8]["greedy_flip"] = {"error": f"{type(e).__name__}: {e}"[:400]}
                    rec["arms"][a8]["problems"].append("greedy flip rate could not run")
        models.clear()
        _release()
    for arm, a in rec["arms"].items():
        a["verdict"] = _verdict(a)
    rec["t_end"] = common.utc_iso()
    return rec


def exit_code(rec: dict) -> int:
    verdicts = [a["verdict"] for a in rec["arms"].values()]
    if any(v == "fail" for v in verdicts):
        return 1
    return 2 if any(v in ("B=1", SPEED_ONLY) for v in verdicts) else 0


def arm_line(arm: str, a: dict) -> str:
    """The console line of one arm: verdict, then every problem and the Amendment 6 attribution."""
    notes = list(a["problems"]) + list(a.get("speed_only_problems", [])) + list(a.get("batched_path_problems", []))
    fr = a.get("fidelity_reference") or {}
    if fr.get("fidelity") == "pass":
        notes.append(f"{FAMILY_RULE_PROBLEM} for the family; by the Amendment 6 reference KL(bf16‖{arm}) "
                     f"{fr['kl_ref_per_token']:.4f} < {fr['threshold']}: fidelity passes")
    return f"{arm:6s} {a['verdict']:10s} " + "; ".join(notes)


def main(argv=None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_036 peer check (BUILD_SPEC §5.9)")
    ap.add_argument("--parity", action="store_true", help="build-time tokenizer/template parity")
    ap.add_argument("--mlx-root", default=None, help="parity: folder holding the mlx-community peer folders")
    ap.add_argument("--upstream-root", default=None, help="parity: folder holding <repo>@<rev8> upstream folders")
    ap.add_argument("--out", default=None, help="parity: output file (default tools/peer_parity_build.json)")
    ap.add_argument("--arms", default="G8,G4,Q36-8,Q36-4,Q38-8,Q38-4")
    ap.add_argument("--checks", default=",".join(ALL_CHECKS))
    ap.add_argument("--out-dir", default=None, help="run host: directory for peers_<UTC>.json (default results/)")
    args = ap.parse_args(argv)

    if args.parity:
        mlx_root = Path(args.mlx_root or common.models_dir())
        up_root = Path(args.upstream_root or common.models_dir() / "upstream")
        rec = build_parity(mlx_root, up_root)
        out = Path(args.out) if args.out else PARITY_FILE
        out.write_text(common.dumps(rec), encoding="utf-8")
        for fam, f in rec["families"].items():
            for arm, p in f.get("peers", {}).items():
                if p.get("present"):
                    print(f"{fam:8s} {arm:6s} ids mismatches {len(p['token_ids']['mismatches'])}/"
                          f"{p['token_ids']['n']}, template mismatches "
                          f"{len(p['template']['render_mismatches'])}/{p['template']['cases']}, "
                          f"bytes equal {sum(p['byte_equal_to_upstream'].values())}/{len(FILES)}")
        print(f"[peer_check] wrote {redact_path(out)}")
        return 0

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARM_DIRS]
    if bad:
        print(f"[peer_check] unknown arms {bad}", file=sys.stderr)
        return 1
    checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
    try:
        common.require_identity()
    except common.IdentityError as e:
        print(f"[peer_check] refusing to run: {e}", file=sys.stderr)
        return 1
    rec = run(arms, checks)
    out_dir = Path(args.out_dir or common.EXP_DIR / "results")
    path = common.write_new_json(out_dir / f"peers_{common.utc_stamp()}.json", rec)
    for arm, a in rec["arms"].items():
        print(arm_line(arm, a).rstrip())
    print(f"[peer_check] wrote {redact_path(path)}")
    return exit_code(rec)


if __name__ == "__main__":
    sys.exit(main())
