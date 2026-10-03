# SPDX-License-Identifier: MIT
"""Port-side harness of the gate (BUILD_SPEC §5.3, items 12, 29).

* RouteTap wraps the port module's `route` for one call site: it records the
  router logits and ids the MoE block used and, in "forced" mode, adds a
  boost of FORCE_BOOST to the reference's expert ids inside the selection
  scores. The selection is then the reference's, while the routing weights
  still come from the port's own route() code on its own fp32 logits (spec
  item 12). Forcing through route() rather than DecoderLayer.branches'
  force_ids keeps weight-path defects (renormalisation, a route scale,
  biased weights; port mutants 6, 7, 14) visible in the forced check. For
  the unmutated port the two forcing paths are cross-checked (G2 record
  "forcing_crosscheck").
* PortLayers builds one DecoderLayer at a time from a checkpoint directory
  (the BF16 shards, or a converted MLX directory for G2q), loads only that
  layer's tensors (mx.load is lazy), runs it on reference inputs and returns
  the branch outputs of item 29 (r_attn, r_moe before the residual adds).
  Masks and caches come from the port's own Kolibri1Model.make_masks and
  Model.make_cache, so the port's window handling is what is tested.
* embedding_rows / head_logits probe the port's own trunk input and head.
* stream_logits teacher-forces a sequence through a loaded model in chunks.

mlx is imported lazily.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate import common

FORCE_BOOST = 1.0e6  # >> |logit + bias| (bias magnitudes reach about 20, item 11)


class HarnessError(RuntimeError):
    """The harness cannot vouch for what it measured (not a port verdict)."""


class _StopTrunk(Exception):
    pass


class RouteTap:
    """Installed as module.route for the duration of a `with` block."""

    def __init__(self, module):
        self.module = module
        self.force = None  # mx.array [..., k] of expert ids, or None
        self.calls: list = []

    def __enter__(self):
        self.orig = self.module.route
        self.module.route = self
        return self

    def __exit__(self, *exc):
        self.module.route = self.orig
        return False

    def __call__(self, logits_f32, expert_bias_f32, top_k, renormalize=False):
        import mlx.core as mx

        bias = expert_bias_f32
        if self.force is not None:
            ids = mx.broadcast_to(self.force.astype(mx.uint32), logits_f32.shape[:-1] + (self.force.shape[-1],))
            boost = mx.put_along_axis(mx.zeros(logits_f32.shape, dtype=mx.float32), ids,
                                      mx.array(FORCE_BOOST, dtype=mx.float32), axis=-1)
            bias = expert_bias_f32.astype(mx.float32) + boost
        weights, ids = self.orig(logits_f32, bias, top_k, renormalize)
        self.calls.append((logits_f32, ids))
        return weights, ids


def _config(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def _weight_index(model_dir: Path) -> dict:
    """tensor name -> shard file name (from the index, or each shard's header)."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    out = {}
    for f in sorted(model_dir.glob("model*.safetensors")):
        for name in common.safetensors_header(f):
            out[name] = f.name
    return out


def _load_fresh(model_dir: Path, index: dict, names) -> dict:
    """Fresh lazy arrays (mx.load) for `names` only. Nothing is kept between
    calls, so evaluated weights are freed with the layer that used them and
    memory stays at about one layer (BUILD_SPEC §5.3: one layer at a time)."""
    import mlx.core as mx

    names = set(names)
    out = {}
    for shard in sorted({index[n] for n in names}):
        arrays = mx.load(str(Path(model_dir) / shard))  # lazy: read when evaluated
        out.update({k: v for k, v in arrays.items() if k in names})
    return out


class PortLayers:
    """One port DecoderLayer at a time, from `model_dir`.

    mode "fp32": every floating parameter cast to fp32 (exact from bf16);
    mode "bf16": as stored (bf16; router and expert_bias fp32 after sanitize);
    a converted directory is quantised as its config says (G2q)."""

    def __init__(self, model_dir: Path, mode: str = "bf16", mutant: str | None = None,
                 port_file: Path = common.PORT_FILE, module=None):
        common.set_offline_env()
        if mode not in ("fp32", "bf16"):
            raise ValueError(mode)
        self.model_dir = Path(model_dir)
        self.mode = mode
        self.mutant = mutant
        self.module = module or common.port_module(mutant, port_file)
        self.config = _config(self.model_dir)
        self.args = self.module.ModelArgs.from_dict(self.config)
        # The whole model is constructed lazily (its random init is never
        # evaluated); it supplies the port's own args, masks and caches.
        self.model = self.module.Model(self.args)
        self.args = self.model.args  # a mutant may replace them (window_512)
        self.quant = self.config.get("quantization")
        self.index = _weight_index(self.model_dir)

    @property
    def dtype(self):
        import mlx.core as mx

        return mx.float32 if self.mode == "fp32" else mx.bfloat16

    def layer_type(self, i: int) -> str:
        return self.args.layer_types[i]

    def build_layer(self, i: int):
        import mlx.core as mx
        import mlx.nn as nn

        prefix = f"model.layers.{i}."
        sub = _load_fresh(self.model_dir, self.index, [k for k in self.index if k.startswith(prefix)])
        sub = self.model.sanitize(sub)
        sub = {k[len(prefix):]: v for k, v in sub.items()}
        layer = self.module.DecoderLayer(self.args, self.layer_type(i))
        if self.quant:
            nn.quantize(layer, group_size=self.quant["group_size"], bits=self.quant["bits"],
                        mode=self.quant.get("mode", "affine"),
                        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in sub)
        layer.load_weights(list(sub.items()), strict=True)
        if self.mode == "fp32" and not self.quant:
            layer.set_dtype(mx.float32)
        mx.eval(layer.parameters())
        return layer

    def _masks(self, x, cache_list):
        """(full mask, sliding mask) from the port's own mask code."""
        trunk = self.model.model
        if hasattr(trunk, "make_masks"):
            return trunk.make_masks(x, cache_list)
        # Older port without make_masks: its __call__ logic, verbatim.
        cache_list = cache_list or [None] * len(trunk.layers)
        fa = self.module.create_attention_mask(x, cache_list[trunk.fa_idx]) if trunk.fa_idx is not None else None
        swa = (self.module.create_attention_mask(x, cache_list[trunk.swa_idx], window_size=trunk.sliding_window)
               if trunk.swa_idx is not None else None)
        return fa, swa

    def _branches(self, layer, x, mask, cache):
        if hasattr(layer, "branches"):
            r_attn, h_mid, r_moe, h_out, _logits, _ids = layer.branches(x, mask, cache)
            return r_attn, r_moe, h_out
        a = layer.post_attn_norm(layer.self_attn(layer.input_layernorm(x), mask, cache))
        h_mid = x + a
        m = layer.post_ffn_norm(layer.mlp(layer.post_attention_layernorm(h_mid)))
        return a, m, h_mid + m

    def run(self, layer, i: int, h: np.ndarray, force_ids: np.ndarray | None = None,
            chunk: int | None = None, check_call: bool = True) -> dict:
        """Branch outputs of layer i for reference inputs h [S, T, H] (fp32
        numpy); force_ids [S, T, k] forces the selection. Returns numpy
        arrays [S, T, ...]: r_attn, r_moe, h_out, router_logits (fp32), ids.

        chunk=None runs each sequence in one call with no cache (T1-T8);
        chunk=n feeds n positions at a time through the port's own cache
        (T9). check_call compares h_out with the layer's own __call__ (no
        cache only); a mismatch means the harness does not reflect the port."""
        import mlx.core as mx

        S, T, _ = h.shape
        sliding = self.layer_type(i) == "sliding_attention"
        outs = {"r_attn": [], "r_moe": [], "h_out": [], "router_logits": [], "ids": []}
        starts = [0] if chunk is None else list(range(0, T, chunk))
        cache_list, cache = None, None
        if chunk is not None:
            cache = self.model.make_cache()[i]
            trunk = self.model.model
            cache_list = [None] * len(self.args.layer_types)
            idx = trunk.swa_idx if sliding else trunk.fa_idx
            cache_list[idx] = cache
        with RouteTap(self.module) as tap:
            for a in starts:
                b = T if chunk is None else min(T, a + chunk)
                x = mx.array(h[:, a:b]).astype(self.dtype)
                tap.force = None if force_ids is None else mx.array(force_ids[:, a:b].astype(np.uint32))
                tap.calls.clear()
                fa, swa = self._masks(x, cache_list)
                mask = swa if sliding else fa
                r_attn, r_moe, h_out = self._branches(layer, x, mask, cache)
                if len(tap.calls) != 1:
                    raise HarnessError(f"layer {i}: route() called {len(tap.calls)} times (expected 1)")
                logits, ids = tap.calls[0]
                mx.eval(r_attn, r_moe, h_out, logits, ids)
                if check_call and chunk is None:
                    tap.calls.clear()  # the same forcing applies to the __call__ run
                    y = layer(x, mask, None)
                    mx.eval(y)
                    if not bool(mx.array_equal(y, h_out)):
                        diff = float(mx.abs(y.astype(mx.float32) - h_out.astype(mx.float32)).max())
                        raise HarnessError(f"layer {i}: branches' h_out differs from the layer's __call__ (max {diff:.3e})")
                outs["r_attn"].append(common.mx_to_np(r_attn))
                outs["r_moe"].append(common.mx_to_np(r_moe))
                outs["h_out"].append(common.mx_to_np(h_out))
                outs["router_logits"].append(np.array(logits.astype(mx.float32)))
                outs["ids"].append(np.array(ids).astype(np.int64))
        res = {k: np.concatenate(v, axis=1) for k, v in outs.items()}
        res["router_logits_dtype"] = str(logits.dtype).replace("mlx.core.", "")
        if force_ids is not None:
            got = np.sort(res["ids"], axis=-1)
            want = np.sort(force_ids.astype(np.int64), axis=-1)
            if not np.array_equal(got, want):
                raise HarnessError(f"layer {i}: forcing did not take (port selected other experts)")
        return res

    def forcing_crosscheck(self, layer, i: int, h: np.ndarray, force_ids: np.ndarray) -> dict | None:
        """r_moe via the route() boost vs via DecoderLayer.branches(force_ids)
        (the port's own forced path), for the unmutated port."""
        import mlx.core as mx

        if not hasattr(layer, "branches"):
            return None
        tapped = self.run(layer, i, h, force_ids, check_call=False)
        x = mx.array(h).astype(self.dtype)
        fa, swa = self._masks(x, None)
        mask = swa if self.layer_type(i) == "sliding_attention" else fa
        _, _, r_moe, _, _, _ = layer.branches(x, mask, None, force_ids=mx.array(force_ids.astype(np.uint32)))
        own = common.mx_to_np(r_moe)
        e = common.rel_err_rows(tapped["r_moe"].reshape(-1, own.shape[-1]), own.reshape(-1, own.shape[-1]))
        return {"layer": i, "max_rel": float(e.max()), "median_rel": float(np.median(e))}

    # --- embedding and head (fp32 / bf16 modes, from the same directory) ----

    def _trunk_model(self, names: list[str]):
        """The lazily constructed model with only `names` loaded."""
        import mlx.core as mx

        missing = [n for n in names if n not in self.index]
        if missing:
            raise HarnessError(f"tensors not in {self.model_dir.name}: {missing}")
        sub = self.model.sanitize(_load_fresh(self.model_dir, self.index, names))
        if self.quant:
            import mlx.nn as nn

            nn.quantize(self.model, group_size=self.quant["group_size"], bits=self.quant["bits"],
                        mode=self.quant.get("mode", "affine"),
                        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in sub)
        self.model.load_weights(list(sub.items()), strict=False)
        if self.mode == "fp32" and not self.quant:
            for mod in (self.model.model.embed_tokens, self.model.model.norm, self.model.lm_head):
                mod.set_dtype(mx.float32)
        return self.model

    def embedding_rows(self, ids_2d: np.ndarray) -> np.ndarray:
        """What the port's trunk feeds to layer 0 for token ids [S, T]:
        captured at layer 0's input, so any scaling in the trunk counts."""
        import mlx.core as mx

        prefix = "model.embed_tokens."
        names = [k for k in self.index if k.startswith(prefix)]
        model = self._trunk_model(names)
        cls = type(model.layers[0])
        captured = {}
        original = cls.__call__

        def capture(layer_self, x, *a, **kw):
            captured["x"] = x
            raise _StopTrunk()

        cls.__call__ = capture
        try:
            model.model(mx.array(ids_2d.astype(np.int32)))
        except _StopTrunk:
            pass
        finally:
            cls.__call__ = original
        x = captured["x"]
        if self.mode == "fp32":
            x = x.astype(mx.float32)
        mx.eval(x)
        return common.mx_to_np(x)

    def head_chunks(self, h: np.ndarray, rows: int = 512, normed: bool = False):
        """Yield (start, fp32 logits [n, V]) of the port's head on h [N, H]
        (fp32), chunk by chunk so the [N, V] matrix never exists whole.
        normed=False: the port's final norm, then the head (the G2 fp32 head
        check, fed the reference's last-layer output); normed=True: the head
        alone on an already normed state (G2q). The head is
        Model.compute_logits when the port has it, else Model.__call__ with
        the trunk stubbed out."""
        import mlx.core as mx

        names = [k for k in self.index if k.startswith("lm_head.") or (not normed and k.startswith("model.norm."))]
        model = self._trunk_model(names)
        if normed and not hasattr(model, "compute_logits"):
            raise HarnessError("the port has no Model.compute_logits (BUILD_SPEC §5.1 delta 1)")
        for a in range(0, h.shape[0], rows):
            x = mx.array(h[a:a + rows][None])
            if not normed:
                x = model.model.norm(x.astype(self.dtype))
            if hasattr(model, "compute_logits"):
                logits = model.compute_logits(x)
            else:
                trunk = model.model
                model.model = _NormOnly(lambda y: y)
                try:
                    logits = model(None, input_embeddings=x)
                finally:
                    model.model = trunk
            mx.eval(logits)
            if logits.dtype != mx.float32:
                raise HarnessError(f"head output dtype {logits.dtype}")
            yield a, np.array(logits[0])

    def head_logits(self, h_last: np.ndarray, rows: int = 512) -> np.ndarray:
        """All head logits (small inputs only; the gate uses head_chunks)."""
        parts = [blk for _, blk in self.head_chunks(h_last, rows)]
        return np.concatenate(parts, axis=0)


class _NormOnly:
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.norm(input_embeddings)


def stream_logits(model, ids, chunk: int = 2048, cache=None):
    """Teacher-force `ids` through a loaded model in chunks of `chunk`
    positions (the port's own cache); yields (start, logits fp32 numpy [c, V])."""
    import mlx.core as mx

    cache = model.make_cache() if cache is None else cache
    ids = np.asarray(ids, dtype=np.int32)
    for a in range(0, ids.size, chunk):
        out = model(mx.array(ids[a:a + chunk])[None], cache=cache)
        out = out[0].astype(mx.float32)
        mx.eval(out)
        yield a, np.array(out)


def logits_at(model, ids, positions: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Logits rows at `positions` (sorted) of a teacher-forced pass."""
    positions = np.asarray(positions)
    need = int(positions.max()) + 1
    rows = []
    for a, block in stream_logits(model, np.asarray(ids)[:need], chunk):
        sel = positions[(positions >= a) & (positions < a + block.shape[0])]
        if sel.size:
            rows.append(block[sel - a])
    return np.concatenate(rows, axis=0)
