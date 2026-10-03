# SPDX-License-Identifier: MIT
"""G0 static checks (HYPOTHESIS Phase 0 G0; BUILD_SPEC §5.3 g0_static.py).

  census                tensor count, parameter count, full-attention layers,
                        every shard's sha256 against its LFS oid (etag)
  strict_load_check     the port consumes every checkpoint tensor and declares
                        no parameter absent from it (catches an afmoe gate)
  template_parity       12 conversations x 10 settings, runtime path vs vendor jinja
  tokenizer_parity      harness ids vs raw `tokenizers` ids, decode round trip,
                        fix_mistral_regex never True
  converted_config_check  quantization block, tensor policy, bf16 scales/biases,
                        unquantised fp32 router
  dtype_asserts         on a K8/K4 forward: router and head logits fp32 and not
                        bf16-exact

Every function returns a plain dict of measured values; pass/fail against
gate/thresholds.json is decided in run_gate.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from pathlib import Path

import numpy as np

from gate import common

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "U32": 4, "I32": 4, "U8": 1, "F8_E4M3": 1}


def _headers(model_dir: Path) -> dict:
    out = {}
    for f in sorted(Path(model_dir).glob("model*.safetensors")):
        for name, meta in common.safetensors_header(f).items():
            out[name] = dict(meta, shard=f.name)
    return out


def _etag(model_dir: Path, shard: str) -> str | None:
    """The etag recorded by `hf download --local-dir` (line 2 of the metadata
    file); for LFS files it is the sha256 of the content."""
    meta = Path(model_dir) / ".cache" / "huggingface" / "download" / f"{shard}.metadata"
    if not meta.is_file():
        return None
    lines = meta.read_text(encoding="utf-8").splitlines()
    return lines[1].strip().strip('"') if len(lines) > 1 else None


def _preflight_hashes(results_dir: Path, head: str | None, repo: str = "Aleph-Alpha/Kolibri-1-BF16") -> dict:
    """Shard sha256 from the newest results/preflight_<UTC>.json of the same
    HEAD (BUILD_SPEC §5.3: preflight --deep hashes are reused by G0):
    record["assets"]["models"][i]["deep"]["sha256"] = {file: hex} for `repo`."""
    for path in reversed(sorted(Path(results_dir).glob("preflight_*.json"))):
        try:
            rec = common.read_json(path)
        except Exception:
            continue
        if head is None or (rec.get("git") or {}).get("head") != head:
            continue
        for m in (rec.get("assets") or {}).get("models", []):
            deep = m.get("deep") or {}
            if m.get("repo") == repo and isinstance(deep.get("sha256"), dict) and deep["sha256"]:
                return {"source": common.redact_path(path),
                        "sha256": {Path(k).name: v for k, v in deep["sha256"].items() if k.endswith(".safetensors")}}
    return {}


def lfs_oids() -> dict:
    """{shard: sha256} committed in tools/kolibri_lfs_oids.json (the LFS oids
    of Aleph-Alpha/Kolibri-1-BF16 at the pinned revision), if present."""
    path = common.EXP_DIR / "tools" / "kolibri_lfs_oids.json"
    if not path.is_file():
        return {}
    rec = common.read_json(path)
    return {k: v["sha256"] for k, v in (rec.get("shards") or {}).items()}


def census(model_dir: Path, expected: dict, results_dir: Path | None = None, hash_shards: bool = True,
           workers: int = 8, oids: dict | None = None) -> dict:
    """expected: tensor_count, param_count, full_attention_layers. Every
    shard's sha256 is compared with its download etag (the LFS oid) and with
    the committed oid list (tools/kolibri_lfs_oids.json) when given."""
    model_dir = Path(model_dir)
    headers = _headers(model_dir)
    params = 0
    for meta in headers.values():
        params += int(np.prod(meta["shape"], dtype=np.int64)) if meta["shape"] else 1
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    full = [i for i, t in enumerate(cfg.get("layer_types", [])) if t == "full_attention"]
    shards = sorted({m["shard"] for m in headers.values()})
    out = {
        "tensors": len(headers),
        "params": params,
        "full_attention_layers": full,
        "n_shards": len(shards),
        "expected": expected,
        "dtypes": sorted({m["dtype"] for m in headers.values()}),
    }
    etags = {s: _etag(model_dir, s) for s in shards}
    if not hash_shards or not all(etags.values()):
        out["shard_sha"] = {"status": "unavailable", "missing_etags": [s for s, e in etags.items() if not e]}
        return out
    pre = _preflight_hashes(results_dir, common.git_head()) if results_dir is not None else {}
    shas = {k: v for k, v in pre.get("sha256", {}).items() if k in etags}
    todo = [s for s in shards if s not in shas]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:  # hashlib releases the GIL
        for s, h in zip(todo, ex.map(lambda s: common.sha256_file(model_dir / s), todo)):
            shas[s] = h
    oids = oids or {}
    mism = [s for s in shards if shas.get(s) != etags[s] or (oids and oids.get(s) != shas.get(s))]
    out["shard_sha"] = {"status": "checked", "source": pre.get("source", "computed"),
                        "n_reused": len(shards) - len(todo), "mismatches": mism,
                        "against": "download etag" + (" and tools/kolibri_lfs_oids.json" if oids else ""),
                        "n_committed_oids": len(oids)}
    return out


def strict_load_check(model_dir: Path, mutant: str | None = None, port_file: Path = common.PORT_FILE) -> dict:
    """Load model_dir lazily with the port's classes; report checkpoint tensors
    the port does not consume and parameters the checkpoint does not provide."""
    common.set_offline_env()
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    module = common.port_module(mutant, port_file)
    model = module.Model(module.ModelArgs.from_dict(cfg))
    import mlx.core as mx

    weights = {}
    for f in sorted(model_dir.glob("model*.safetensors")):
        weights.update(mx.load(str(f)))  # lazy; only shapes are used
    n_ckpt = len(weights)
    numel_ckpt = sum(int(np.prod(v.shape, dtype=np.int64)) for v in weights.values())
    sanitized = model.sanitize(dict(weights))
    quant = cfg.get("quantization")
    if quant:
        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode", "affine"),
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in sanitized)
    params = dict(tree_flatten(model.parameters()))
    unused = sorted(set(sanitized) - set(params))
    missing = sorted(set(params) - set(sanitized))
    shape_mismatch = sorted(k for k in set(params) & set(sanitized) if tuple(params[k].shape) != tuple(sanitized[k].shape))
    numel_params = sum(int(np.prod(v.shape, dtype=np.int64)) for v in params.values())
    error = None
    try:
        model.load_weights(list(sanitized.items()), strict=True)
    except Exception as e:  # the loader's own verdict
        error = f"{type(e).__name__}: {str(e)[:300]}"
    return {
        "checkpoint_tensors": n_ckpt,
        "numel_checkpoint": numel_ckpt,
        "numel_params": numel_params,
        "numel_equal": numel_ckpt == numel_params,
        "quantized": bool(quant),
        "unused_ckpt_tensors": unused[:50],
        "n_unused_ckpt_tensors": len(unused),
        "missing_params": missing[:50],
        "n_missing_params": len(missing),
        "shape_mismatch": shape_mismatch[:50],
        "load_error": error,
    }


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def template_parity(tok_dir: Path, cases_path: Path = common.TEMPLATE_CASES) -> dict:
    """Runtime path (mlx_lm TokenizerWrapper.apply_chat_template, which also
    injects enable_thinking when absent) against the vendor jinja."""
    common.set_offline_env()
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from gate import vendor_template

    cases = common.read_json(cases_path)
    tw = load_tokenizer(Path(tok_dir))
    mismatches, n = [], 0
    runner_render = _runner_render()
    runner_mism, runner_n = [], 0
    for cname, messages in cases["conversations"].items():
        for sname, s in cases["settings"].items():
            msgs, tools = list(messages), None
            if s.get("tools"):
                tools = cases["tools"]
                if msgs[0]["role"] != "system":
                    msgs = [{"role": "system", "content": s["system_if_absent"]}] + msgs
            harness = tw.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **s["kwargs"])
            vendor = vendor_template.render(msgs, tools=tools, add_generation_prompt=True, **s["kwargs"])
            n += 1
            if harness != vendor:
                mismatches.append({"conversation": cname, "setting": sname, "first_diff_char": _first_diff(harness, vendor),
                                   "harness_sha256": common.sha256_bytes(harness.encode()),
                                   "vendor_sha256": common.sha256_bytes(vendor.encode())})
            effort = s["kwargs"].get("reasoning_effort")
            if runner_render is not None and effort in ("none", "low", "medium", "high") and not tools \
                    and set(s["kwargs"]) == {"reasoning_effort"}:
                runner_n += 1
                text = runner_render("kolibri", tw, msgs, effort)[0]
                if text != vendor:
                    runner_mism.append({"conversation": cname, "setting": sname})
    return {
        "n": n,
        "n_conversations": len(cases["conversations"]),
        "n_settings": len(cases["settings"]),
        "mismatches": mismatches,
        "cases_sha256": common.sha256_file(cases_path),
        "vendor_jinja_sha256": common.sha256_file(common.VENDOR_JINJA),
        "runner_chat": {"present": runner_render is not None, "n": runner_n, "mismatches": runner_mism},
    }


def _runner_render():
    """runner.chat.render(family, tokenizer, messages, effort) when present."""
    try:
        from runner import chat
    except Exception:
        return None
    return getattr(chat, "render", None)


def tokenizer_parity(tok_dir: Path, groups: dict, gate_texts: dict) -> dict:
    """groups: {name: [lines]} (tokenizer_lines.py); gate_texts: {id: (text,
    committed ids or None)}. Records counts and group sha256 only."""
    common.set_offline_env()
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from gate import tokenizer_lines

    raw = common.load_raw_tokenizer(tok_dir)
    tw = load_tokenizer(Path(tok_dir))
    hf = tw._tokenizer
    n = mism = rt_fail = 0
    examples = []
    group_info = {}
    for gname, lines in groups.items():
        g_m = 0
        for line in lines:
            want = raw.encode(line, add_special_tokens=False).ids
            got = hf.encode(line, add_special_tokens=False)
            n += 1
            if list(got) != want:
                mism += 1
                g_m += 1
                if len(examples) < 5 and gname != "mmlu":
                    examples.append({"group": gname, "line_sha256": common.sha256_bytes(line.encode())})
            if raw.decode(want, skip_special_tokens=False) != line:
                rt_fail += 1
        group_info[gname] = {"n": len(lines), "sha256": tokenizer_lines.group_digest(lines), "mismatches": g_m}
    text_info = {}
    for tid, (text, ids) in gate_texts.items():
        want = raw.encode(text, add_special_tokens=False).ids
        got = list(hf.encode(text, add_special_tokens=False))
        entry = {"harness_equals_raw": got == want}
        if ids is not None:
            entry["decode_equals_text"] = raw.decode(list(ids), skip_special_tokens=False) == text
        text_info[tid] = entry
        n += 1
        mism += int(got != want)
        rt_fail += int(ids is not None and not entry["decode_equals_text"])
    fmr = (getattr(hf, "init_kwargs", {}) or {}).get("fix_mistral_regex")
    return {"n_lines": n, "mismatches": mism, "roundtrip_fail": rt_fail, "examples": examples,
            "groups": group_info, "gate_texts": text_info, "fix_mistral_regex": fmr,
            "tokenizer_json_sha256": common.sha256_file(Path(tok_dir) / "tokenizer.json")}


def _bits_of(meta_w: dict, meta_s: dict, group_size: int) -> int | None:
    """Bits of an affine-packed tensor from its uint32 weight and its scales."""
    if meta_w["dtype"] != "U32":
        return None
    cols_packed, n_groups = meta_w["shape"][-1], meta_s["shape"][-1]
    return int(round(cols_packed * 32 / (n_groups * group_size)))


def _policy_of(headers: dict, prefix: str, group_size: int) -> str:
    w = headers.get(f"{prefix}.weight")
    if w is None:
        return "absent"
    s = headers.get(f"{prefix}.scales")
    if s is not None:
        return f"{_bits_of(w, s, group_size)}bit"
    return {"BF16": "bf16", "F32": "fp32", "F16": "fp16"}.get(w["dtype"], w["dtype"])


def converted_config_check(converted_dir: Path, bits: int, policy_name: str, thresholds_g0: dict) -> dict:
    d = Path(converted_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    q = cfg.get("quantization") or {}
    headers = _headers(d)
    record = json.loads((d / common.CONVERT_RECORD).read_text(encoding="utf-8")) if (d / common.CONVERT_RECORD).is_file() else {}
    gs = int(q.get("group_size", 64))
    problems = []
    if not (q.get("group_size") == 64 and q.get("bits") == bits and q.get("mode", "affine") == "affine"):
        problems.append(f"quantization block {q}")
    want_dtype = {"bfloat16": "BF16"}.get(thresholds_g0["scales_biases_dtype"], thresholds_g0["scales_biases_dtype"])
    n_quant, wrong_bits = 0, []
    for name, meta in headers.items():
        if name.endswith(".scales"):
            n_quant += 1
            base = name[: -len(".scales")]
            bmeta = headers.get(f"{base}.biases")
            if meta["dtype"] != want_dtype or bmeta is None or bmeta["dtype"] != want_dtype:
                problems.append(f"{base}: scales/biases {meta['dtype']}/{(bmeta or {}).get('dtype')}")
            b = _bits_of(headers[f"{base}.weight"], meta, gs)
            if b != bits:
                wrong_bits.append(base)
    n_layers = cfg["num_hidden_layers"]
    router_dtypes = set()
    for i in range(n_layers):
        p = f"model.layers.{i}"
        for unq in (f"{p}.mlp.gate", f"{p}.input_layernorm", f"{p}.post_attn_norm", f"{p}.post_attention_layernorm",
                    f"{p}.post_ffn_norm", f"{p}.self_attn.q_norm", f"{p}.self_attn.k_norm"):
            if f"{unq}.scales" in headers:
                problems.append(f"{unq} is quantised")
        if f"{p}.mlp.gate.expert_bias.scales" in headers or f"{p}.mlp.gate.expert_bias" not in headers:
            problems.append(f"{p}.mlp.gate.expert_bias missing or quantised")
        router_dtypes.add(headers.get(f"{p}.mlp.gate.weight", {}).get("dtype"))
    want_router = {"float32": "F32"}[thresholds_g0["router_weight_dtype"]]
    if router_dtypes != {want_router}:
        problems.append(f"router weight dtypes {sorted(map(str, router_dtypes))}, want {want_router}")
    observed = {"embed_tokens": _policy_of(headers, "model.embed_tokens", gs), "lm_head": _policy_of(headers, "lm_head", gs)}
    expected_raw = thresholds_g0["tensor_policy"][policy_name]
    expected = {k: (f"{bits}bit" if v == "arm_bits" else v) for k, v in expected_raw.items()}
    rec_policy = record.get("tensor_policy")
    if rec_policy is None and record:
        rec_policy = {"embed_tokens": f"{bits}bit" if record.get("quantize_embeddings") else "bf16",
                      "lm_head": f"{bits}bit" if record.get("quantize_lm_head") else "fp32",
                      "derived_from": "quantize_embeddings / quantize_lm_head flags"}
    rec_cmp = {k: v for k, v in (rec_policy or {}).items() if k in ("embed_tokens", "lm_head")}
    rec_norm = {k: (f"{bits}bit" if v in ("arm_bits", bits, f"{bits}", f"{bits}bit", "quantised", "quantized") else v)
                for k, v in rec_cmp.items()}
    if observed != expected:
        problems.append(f"tensor policy {observed} != signed-off {policy_name} {expected}")
    if rec_norm and rec_norm != expected:
        problems.append(f"record tensor policy {rec_cmp} != signed-off {expected}")
    if wrong_bits:
        problems.append(f"{len(wrong_bits)} tensors not at {bits} bits, e.g. {wrong_bits[:3]}")
    return {"quantization": q, "n_quantised_tensors": n_quant, "observed_policy": observed,
            "expected_policy": expected, "policy_name": policy_name, "record_policy": rec_policy,
            "router_weight_dtypes": sorted(map(str, router_dtypes)), "problems": problems[:30],
            "n_problems": len(problems)}


def dtype_asserts(model, module, ids) -> dict:
    """Router-logit and head-output dtypes and their bf16-exact fractions on
    one forward of `ids` (a 1-D id list) through a loaded model."""
    import mlx.core as mx

    from gate.harness import RouteTap

    with RouteTap(module) as tap:
        out = model(mx.array(np.asarray(ids, dtype=np.int32))[None])
        mx.eval(out)
        logits = [l for l, _ in tap.calls]
        mx.eval(logits)
    router_dtypes = sorted({str(l.dtype).replace("mlx.core.", "") for l in logits})
    router_np = np.concatenate([np.array(l.astype(mx.float32)).reshape(-1) for l in logits])
    head_np = np.array(out.astype(mx.float32)).reshape(-1)
    return {
        "n_tokens": int(len(ids)),
        "router_calls": len(logits),
        "router_logits_dtype": router_dtypes[0] if len(router_dtypes) == 1 else router_dtypes,
        "head_dtype": str(out.dtype).replace("mlx.core.", ""),
        "router_bf16_exact_frac": common.bf16_exact_fraction(router_np),
        "head_bf16_exact_frac": common.bf16_exact_fraction(head_np),
    }
