#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Part of Chronos exp_036 (Miktam, 2026-10-03).
"""Fallback converter: the same output as convert.py, built one layer at a time.

BUILD_SPEC 5.1 (port/convert_streaming.py) and spec item 25(b): if
mlx_lm.convert's whole-model pass does not fit in memory on the run host,
convert.py's memory guard stops it and prints the command with --streaming,
which lands here:

    python port/convert.py --src $EXP036_MODELS/Kolibri-1-BF16 \
        --out $EXP036_MODELS/Kolibri-1-MLX-8bit-g64 --bits 8 --group-size 64 --streaming

The tensors are bit-identical to mlx_lm.convert's (tests/test_convert.py,
test_streaming_bit_identical): every step is the one mlx_lm takes, applied
to one group of tensors at a time instead of to the whole model:

  1. the source shards are opened lazily (mx.load); only one group's arrays
     are ever evaluated;
  2. the port's own sanitize maps and stacks the group's tensors (item 21);
  3. each parameter is cast to bfloat16 where the port's cast_predicate says
     so (mlx_lm.convert's --dtype step);
  4. each module the quant predicate selects (the same wrapper as
     mlx_lm.utils.quantize_model: to_quantized, weight.shape[-1] % group
     size, the port's predicate) is quantised with module.to_quantized,
     i.e. the call nn.quantize makes;
  5. the group is written as one safetensors shard (metadata format "mlx").

Groups: the embedding, each decoder layer, then the final norm with the
head. The shard layout differs from mlx_lm's (one shard per group instead of
~5 GB shards), so file hashes differ; tensor names, dtypes, shapes and bytes
do not. config.json is written with mlx_lm's own save_config, the model card
with its create_model_card, and the index in mlx_lm's format.

Peak memory is about one layer: ~3.1 GB of bf16 weights, its stacked copy
and the quantised result, i.e. well under 10 GB on the real model.
"""

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _group_of(name: str):
    """Sort key and group label of a source tensor name."""
    if name.startswith("model.embed_tokens."):
        return (0, -1), "embed"
    if name.startswith("model.layers."):
        i = int(name.split(".")[2])
        return (1, i), f"layer{i}"
    return (2, 0), "final"  # model.norm, lm_head


def write_streaming(staging, work, bits: int, group_size: int, quantize_embeddings: bool = True,
                    quantize_lm_head: bool = True, port=None) -> dict:
    """Write the converted model into `work` (created; must not exist) from the
    staging directory convert.py builds (symlinked shards, edited config,
    kolibri1.py). Tokenizer files are copied from staging; convert.py then
    restores them from the source and verifies the output, as for mlx_lm.

    Returns {"shards": n, "tensors": m, "total_size": bytes}."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import create_model_card, load_config, save_config

    staging, work = Path(staging), Path(work)
    if work.exists():
        raise SystemExit(f"{work} exists; convert.py removes stale partial outputs first")
    if port is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("exp036_kolibri1_streaming", staging / "kolibri1.py")
        port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(port)

    config = load_config(staging)  # as mlx_lm.load does (merges generation_config eos)
    args = port.ModelArgs.from_dict(config)
    if (args.exp036_quantize_embeddings, args.exp036_quantize_lm_head) != (
        quantize_embeddings, quantize_lm_head
    ):
        raise SystemExit("staging config policy keys disagree with the requested policy")
    # The model object is only a map of module paths to module types. Its own
    # parameters are lazy random initialisers that are never evaluated.
    model = port.Model(args)
    modules = dict(model.named_modules())
    cast = model.cast_predicate
    predicate = port.make_quant_predicate(
        group_size=group_size, quantize_embeddings=quantize_embeddings, quantize_lm_head=quantize_lm_head
    )

    def wrapped_predicate(path, module):
        # mlx_lm.utils.quantize_model's wrapper, verbatim in behaviour.
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        return predicate(path, module)

    # 1. Lazy views of every source tensor, grouped.
    source = {}
    for shard in sorted(staging.glob("model*.safetensors")):
        source.update(mx.load(str(shard)))
    groups = {}
    for name in source:
        key, label = _group_of(name)
        groups.setdefault(key, (label, []))[1].append(name)
    order = sorted(groups)

    work.mkdir(parents=True)
    n_shards = len(order)
    weight_map = {}
    total_size = 0
    total_parameters = 0
    n_tensors = 0
    for k, key in enumerate(order):
        label, names = groups[key]
        # 2. sanitize this group only (it acts on whichever names are present).
        tensors = port.Model.sanitize(model, {n: source[n] for n in names})
        by_module = {}
        for name, value in tensors.items():
            path, param = name.rsplit(".", 1)
            by_module.setdefault(path, {})[param] = value

        out = {}
        for path, params in by_module.items():
            if path not in modules:
                raise SystemExit(f"{label}: {path} is not a module of the port")
            # 3. the --dtype cast.
            params = {
                p: (v.astype(mx.bfloat16) if cast(f"{path}.{p}") and mx.issubdtype(v.dtype, mx.floating) else v)
                for p, v in params.items()
            }
            module = modules[path]
            if wrapped_predicate(path, module):
                # 4. quantise exactly as nn.quantize does: module.to_quantized.
                saved = {p: module[p] for p in params}
                module.update(params)
                quantised = module.to_quantized(group_size=group_size, bits=bits, mode="affine")
                q_params = dict(tree_flatten(quantised.parameters()))
                module.update(saved)  # back to the unevaluated placeholders
                n_logical = q_params["weight"].size * 32 // bits
                if "bias" in q_params:
                    n_logical += q_params["bias"].size
                total_parameters += n_logical
                params = q_params
            else:
                total_parameters += sum(v.size for v in params.values())
            for p, v in params.items():
                out[f"{path}.{p}"] = v

        mx.eval(list(out.values()))
        shard_name = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        mx.save_safetensors(str(work / shard_name), out, metadata={"format": "mlx"})
        for name, value in out.items():
            weight_map[name] = shard_name
            total_size += value.nbytes
        n_tensors += len(out)
        print(f"[exp036] streaming: {label} -> {shard_name} ({len(out)} tensors)", flush=True)
        del out, tensors, by_module
        mx.clear_cache()

    index = {
        "metadata": {"total_size": total_size, "total_parameters": total_parameters},
        "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
    }
    with open(work / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=4)

    # 5. config, tokenizer, model card, *.py, generation_config: as mlx_lm.save.
    config = dict(config)
    quant = {"group_size": group_size, "bits": bits, "mode": "affine"}
    config["quantization"] = quant
    config["quantization_config"] = quant
    save_config(config, config_path=work / "config.json")
    for name in os.listdir(staging):
        path = staging / name
        if name.endswith(".py") or name == "generation_config.json":
            shutil.copyfile(path, work / name)
    create_model_card(work, None)
    # Tokenizer files: convert.py replaces them with the source's bytes next.
    return {"shards": n_shards, "tensors": n_tensors, "total_size": total_size}


def convert_streaming(src, out, bits: int, group_size: int = 64, quantize_embeddings: bool = True,
                      quantize_lm_head: bool = True, force: bool = False) -> dict:
    """BUILD_SPEC 5.1 API: convert.py's whole pipeline (source checks,
    staging, memory guard, tokenizer restore, verification, record) with the
    layer-by-layer writer above. Returns the conversion record."""
    try:  # imported as port.convert_streaming (tests) or run from port/
        from . import convert as _convert
    except ImportError:
        import convert as _convert
    return _convert.convert(src, out, bits, group_size, quantize_embeddings=quantize_embeddings,
                            quantize_lm_head=quantize_lm_head, force=force, streaming=True)


if __name__ == "__main__":
    raise SystemExit("run: python port/convert.py ... --streaming")
