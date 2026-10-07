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

exp_037 (DESIGN §3.0, §3.1 P5b-P5c; build task W3) adds, for whole models:
* ForcedTrunk(RouteTap): installed as the port module's `route` for one
  model, in mode "free" (record the port's own selection), "force" (force
  every layer onto ids handed to the port in ascending order: I, Csort) or
  "force_own_order" (force onto ids in the order given: P5b). Forced
  (weights, ids) pairs come back in the order of the forced ids, the weights
  still from the port's own route() on its own fp32 logits, so weight-path
  mutants (6, 7, 14) stay visible. Each forward checks the route-call count
  (= num_hidden_layers), the row count against the pack rows the driver set,
  and the selected set against the forced set (HarnessError otherwise), and
  records the ids used and the shadow (the port's own selection on the
  unboosted logits + bias, from a second route() call).
* forced_stream / forced_logits: teacher-forcing through ForcedTrunk in
  chunks (an int, or explicit spans such as decode_spans' prefix-then-one-
  token schedule), one cache.
* p5b_exactness and p5c_forcing_active: the P5b and P5c measurements.
PortLayers.run and forcing_crosscheck stay as registered (one call over all
8 sequences, DESIGN §3.4). There is no row assertion anywhere here (§6.3).

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


# ---------------------------------------------------------------------------
# exp_037: forced whole-model runs (DESIGN §3.0, §3.1 P5b-P5c; W3)
# ---------------------------------------------------------------------------

FORCED_TRUNK_MODES = ("free", "force", "force_own_order")


class _Namespace:
    """Attribute access to a port module's globals dict (as returned by
    tests/port_harness.port_namespace), so a model loaded by mlx_lm's own
    model_file path can take a tap like a module object."""

    def __init__(self, ns: dict):
        object.__setattr__(self, "_ns", ns)

    def __getattr__(self, key):
        try:
            return self._ns[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self._ns[key] = value


def num_layers_of(model) -> int:
    """num_hidden_layers of a loaded port model."""
    args = getattr(model, "args", None)
    n = getattr(args, "num_hidden_layers", None)
    return int(n) if n is not None else len(model.layers)


class ForcedTrunk(RouteTap):
    """The port module's `route` for one model, for whole-model runs (DESIGN §3.0).

    Modes:
      "free"             route() as is; records the ids in route()'s own order.
      "force"            every layer forced onto table[layer][rows], handed to
                         the port in ascending order (I; Csort's sorted ids).
      "force_own_order"  forced onto table[layer][rows] in the order given (P5b:
                         a free run's own ids reproduce it bitwise).

    `table` is indexed table[layer][row] -> k ids (an [L, N, k] array, or a list
    of L arrays [N, k], e.g. memory-mapped dump files); `rows` are the pack rows
    (or positions) the driver sets for each forward with begin(rows).

    Forcing adds FORCE_BOOST to the forced ids inside the selection scores and
    calls the original route() (the port's own, possibly mutated), so the
    weights come from the port's own fp32 logits (spec item 12). The returned
    (weights, ids) pairs are then reordered into the forced order: the order is
    the caller's, never argpartition's. A second route() call on the unboosted
    logits + bias records the shadow (the port's own selection); in free mode
    the shadow is the selection itself.

    Bookkeeping per forward (begin(rows) ... end(outputs)): route() must be
    called exactly num_layers times (call index = layer), each call must carry
    exactly len(rows) rows, and in a forced mode the selected set must equal
    the forced set on every row. Any mismatch is a HarnessError: the harness
    cannot vouch for the run. Records are appended only for a forward that
    passed every check."""

    def __init__(self, module, num_layers: int, mode: str = "free", table=None, shadow: bool = True):
        if isinstance(module, dict):
            module = _Namespace(module)
        super().__init__(module)
        if mode not in FORCED_TRUNK_MODES:
            raise ValueError(f"ForcedTrunk mode {mode!r}; known: {FORCED_TRUNK_MODES}")
        self.mode = mode
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if mode == "free":
            if table is not None:
                raise ValueError("free mode takes no table")
        else:
            if table is None:
                raise ValueError(f"mode {mode!r} needs a table of forced ids")
            if len(table) != self.num_layers:
                raise HarnessError(f"forcing table has {len(table)} layers, the model {self.num_layers}")
        self.table = table
        self.shadow = bool(shadow)
        self.orig = None
        self._open = False
        self._rows = None
        self._layer = 0
        self._pending: list = []
        self.clear()

    # --- installation -------------------------------------------------------

    def __enter__(self):
        if self.module.route is self:
            raise HarnessError("ForcedTrunk is already installed")
        if isinstance(self.module.route, RouteTap):
            raise HarnessError("another route tap is installed on this module")
        return super().__enter__()

    def __exit__(self, *exc):
        self._open, self._pending, self._rows = False, [], None
        return super().__exit__(*exc)

    @property
    def installed(self) -> bool:
        return self.module.route is self

    # --- records ------------------------------------------------------------

    def clear(self) -> None:
        """Drop every record (ids, shadow, rows)."""
        self._ids = [[] for _ in range(self.num_layers)]
        self._shadow = [[] for _ in range(self.num_layers)]
        self._rows_rec: list = []
        self.forwards = 0

    def records(self) -> dict:
        """{"ids": [L, T, k] int64 (the ids the port used: route()'s own order in
        free mode, the forced order otherwise), "shadow": [L, T, k] int64 (the
        port's own selection on the unboosted scores, route()'s order; compare
        as sets), "rows": [T] int64 (the rows set for each forward, in order)}."""
        if not self._rows_rec:
            return {"ids": None, "shadow": None, "rows": np.zeros(0, dtype=np.int64)}
        return {"ids": np.stack([np.concatenate(v, axis=0) for v in self._ids]),
                "shadow": np.stack([np.concatenate(v, axis=0) for v in self._shadow]),
                "rows": np.concatenate(self._rows_rec)}

    # --- one forward --------------------------------------------------------

    def begin(self, rows) -> None:
        """Start a forward over `rows` (pack rows, or positions; one per token
        the forward feeds, in the order of the flattened [B, c] input)."""
        if not self.installed:
            raise HarnessError("ForcedTrunk.begin() while not installed (use it as a context manager)")
        if self._open:
            raise HarnessError("ForcedTrunk.begin() before the previous forward's end()")
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            raise HarnessError("ForcedTrunk.begin() with no rows")
        if self.mode != "free":
            n = len(self.table[0])
            if rows.min() < 0 or rows.max() >= n:
                raise HarnessError(f"rows outside the forcing table's {n} rows")
        self._rows, self._layer, self._pending, self._open = rows, 0, [], True

    def abort(self) -> None:
        """Discard the open forward (after an exception inside it)."""
        self._open, self._pending, self._rows = False, [], None

    def _forced(self, layer: int) -> np.ndarray:
        f = np.asarray(self.table[layer][self._rows], dtype=np.int64)
        if f.ndim != 2 or f.shape[0] != self._rows.size:
            raise HarnessError(f"layer {layer}: forcing table rows have shape {f.shape}")
        return np.sort(f, axis=-1) if self.mode == "force" else f

    def __call__(self, logits_f32, expert_bias_f32, top_k, renormalize=False):
        import mlx.core as mx

        if not self._open:
            raise HarnessError("route() called outside a ForcedTrunk forward (begin(rows) first)")
        layer = self._layer
        if layer >= self.num_layers:
            raise HarnessError(f"route() called more than num_hidden_layers = {self.num_layers} times in one forward")
        self._layer += 1
        lead = tuple(int(s) for s in logits_f32.shape[:-1])
        n = int(np.prod(lead)) if lead else 1
        if n != self._rows.size:
            raise HarnessError(f"layer {layer}: route() got {n} rows, the driver set {self._rows.size} "
                               "(rows misaligned with the forward)")
        if self.mode == "free":
            weights, ids = self.orig(logits_f32, expert_bias_f32, top_k, renormalize)
            self._pending.append((layer, ids, None, None))
            return weights, ids
        forced = self._forced(layer)
        if forced.shape[-1] != top_k:
            raise HarnessError(f"layer {layer}: {forced.shape[-1]} forced ids per row, top_k is {top_k}")
        n_experts = int(logits_f32.shape[-1])
        if forced.min() < 0 or forced.max() >= n_experts:
            raise HarnessError(f"layer {layer}: forced ids outside [0, {n_experts})")
        fid = mx.array(forced.astype(np.uint32)).reshape(lead + (top_k,))
        boost = mx.put_along_axis(mx.zeros(logits_f32.shape, dtype=mx.float32), fid,
                                  mx.array(FORCE_BOOST, dtype=mx.float32), axis=-1)
        weights, sel = self.orig(logits_f32, expert_bias_f32.astype(mx.float32) + boost, top_k, renormalize)
        sel = sel.astype(mx.uint32)
        # perm[..., j] = the slot of route()'s output holding forced id j; a
        # forced id route() did not select makes the set check in end() fire.
        match = (sel[..., None, :] == fid[..., :, None]).astype(mx.int32)
        perm = mx.argmax(match, axis=-1)
        w_out = mx.take_along_axis(weights, perm, axis=-1)
        ids_out = mx.take_along_axis(sel, perm, axis=-1)
        shadow = self.orig(logits_f32, expert_bias_f32, top_k, renormalize)[1] if self.shadow else None
        self._pending.append((layer, sel, shadow, forced, ids_out))
        return w_out, ids_out

    def end(self, *outputs) -> None:
        """Finish the forward: evaluate `outputs` with the recorded ids, run the
        count and set checks, and append the records."""
        import mlx.core as mx

        if not self._open:
            raise HarnessError("ForcedTrunk.end() without begin()")
        try:
            if self._layer != self.num_layers:
                raise HarnessError(f"route() called {self._layer} times in one forward "
                                   f"(expected num_hidden_layers = {self.num_layers})")
            arrays = list(outputs)
            for p in self._pending:
                arrays.extend(a for a in p[1:] if a is not None and not isinstance(a, np.ndarray))
            mx.eval(*arrays)
            n = self._rows.size
            got_ids, got_shadow = [None] * self.num_layers, [None] * self.num_layers
            for p in self._pending:
                layer, sel, shadow, forced = p[0], p[1], p[2], p[3]
                sel_np = np.array(sel).astype(np.int64).reshape(n, -1)
                if forced is None:
                    used = sel_np
                else:
                    bad = np.any(np.sort(sel_np, axis=-1) != np.sort(forced, axis=-1), axis=-1)
                    if bad.any():
                        raise HarnessError(f"layer {layer}: forcing did not take on {int(bad.sum())} of {n} rows "
                                           "(selected set != forced set)")
                    used = np.array(p[4]).astype(np.int64).reshape(n, -1)
                    if not np.array_equal(used, forced):
                        raise HarnessError(f"layer {layer}: forced ids not handed to the port in the forced order")
                got_ids[layer] = used
                got_shadow[layer] = used if shadow is None else np.array(shadow).astype(np.int64).reshape(n, -1)
            for layer in range(self.num_layers):
                self._ids[layer].append(got_ids[layer])
                self._shadow[layer].append(got_shadow[layer])
            self._rows_rec.append(self._rows)
            self.forwards += 1
        finally:
            self._open, self._pending = False, []


def records_table(rec: dict, n_rows: int | None = None) -> np.ndarray:
    """A forcing table [L, n_rows, k] from ForcedTrunk.records(): each recorded
    row's ids at that row (e.g. a free run over pack rows, to force the pack
    onto its own ids: Csort, P5b). Rows never recorded hold -1, which a forced
    run refuses (HarnessError), so a gap cannot be forced silently."""
    rows = np.asarray(rec["rows"], dtype=np.int64)
    if rec["ids"] is None or rows.size == 0:
        raise HarnessError("no records to build a table from")
    n = int(rows.max()) + 1 if n_rows is None else int(n_rows)
    if np.unique(rows).size != rows.size:
        raise HarnessError("a row was recorded twice; the table would be ambiguous")
    L, _, k = rec["ids"].shape
    table = np.full((L, n, k), -1, dtype=np.int64)
    table[:, rows] = rec["ids"]
    return table


def chunk_spans(n: int, chunk) -> list:
    """[(a, b)] tiling [0, n): consecutive chunks of `chunk` positions, or
    `chunk` itself when it is already a list of spans (checked to tile [0, n))."""
    if isinstance(chunk, (int, np.integer)):
        c = int(chunk)
        if c <= 0:
            raise ValueError("chunk must be positive")
        return [(a, min(n, a + c)) for a in range(0, n, c)]
    spans = [(int(a), int(b)) for a, b in chunk]
    pos = 0
    for a, b in spans:
        if a != pos or b <= a:
            raise ValueError(f"spans do not tile [0, {n}) in order: {(a, b)} after {pos}")
        pos = b
    if pos != n:
        raise ValueError(f"spans end at {pos}, the sequence has {n} positions")
    return spans


def decode_spans(n: int, start: int, prefill_chunk: int = 2048) -> list:
    """The forced-decode schedule (G5-D32 run iii): positions [0, start) prefilled
    in `prefill_chunk` chunks, then one position per forward up to n."""
    if not 0 <= start <= n:
        raise ValueError(f"start {start} outside [0, {n}]")
    prefix = chunk_spans(start, prefill_chunk) if start else []
    return prefix + [(t, t + 1) for t in range(start, n)]


def forced_stream(model, ft: ForcedTrunk, seq_rows, ids, chunk=2048, cache=None):
    """Teacher-force one sequence `ids` through `model` with ForcedTrunk `ft` as
    its route; yields (start, fp32 logits numpy [c, V]) per forward. seq_rows
    [T] are the pack rows (or positions) of the sequence's tokens: forward
    (a, b) sets rows seq_rows[a:b]. `chunk` is a chunk size or a list of spans
    (chunk_spans, decode_spans); all forwards share one cache
    (model.make_cache() unless given). ft is installed for the duration unless
    the caller has installed it."""
    import mlx.core as mx

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    seq_rows = np.asarray(seq_rows, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("forced_stream needs a non-empty sequence")
    if seq_rows.shape != ids.shape:
        raise HarnessError(f"{seq_rows.size} pack rows for {ids.size} tokens (rows misaligned with the sequence)")
    spans = chunk_spans(ids.size, chunk)
    cache = model.make_cache() if cache is None else cache
    own = not ft.installed
    if own:
        ft.__enter__()
    try:
        for a, b in spans:
            ft.begin(seq_rows[a:b])
            try:
                out = model(mx.array(ids[a:b].astype(np.int32))[None], cache=cache)
                out = out[0].astype(mx.float32)
                ft.end(out)
            except BaseException:
                ft.abort()
                raise
            yield a, np.array(out)
    finally:
        if own:
            ft.__exit__(None, None, None)


def forced_logits(model, ft: ForcedTrunk, seq_rows, ids, chunk=2048, positions=None, cache=None) -> np.ndarray:
    """fp32 logits [T, V] of one sequence teacher-forced through ForcedTrunk
    (forced_stream), or only the rows at `positions` (sorted or not) when given."""
    ids = np.asarray(ids).reshape(-1)
    if positions is None:
        return np.concatenate([blk for _, blk in forced_stream(model, ft, seq_rows, ids, chunk, cache)], axis=0)
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size and (positions.min() < 0 or positions.max() >= ids.size):
        raise ValueError("positions outside the sequence")
    out = None
    for a, blk in forced_stream(model, ft, seq_rows, ids, chunk, cache):
        if out is None:
            out = np.empty((positions.size, blk.shape[-1]), dtype=np.float32)
        sel = np.flatnonzero((positions >= a) & (positions < a + blk.shape[0]))
        out[sel] = blk[positions[sel] - a]
    return out


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Same shape, dtype and bit pattern (unlike ==: -0.0 vs 0.0 differ, NaNs compare)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def free_run(model, module, ids, chunk=None, rows=None) -> tuple[np.ndarray, dict]:
    """(fp32 logits [T, V], ForcedTrunk records) of one free-routing run through
    a ForcedTrunk in free mode. chunk=None: one chunk of len(ids)."""
    ids = np.asarray(ids).reshape(-1)
    rows = np.arange(ids.size) if rows is None else rows
    ft = ForcedTrunk(module, num_layers_of(model), "free")
    logits = forced_logits(model, ft, rows, ids, chunk or ids.size)
    return logits, ft.records()


def shift_ids(ids_rec: np.ndarray) -> np.ndarray:
    """P5c's table: position t gets the ids of position t - 1, layer by layer;
    position 0 keeps its own. ids_rec [L, T, k]."""
    out = np.array(ids_rec, copy=True)
    out[:, 1:] = ids_rec[:, :-1]
    return out


def p5b_exactness(model, module, ids, chunk=None) -> dict:
    """P5b (DESIGN §3.1): the port forced onto its own free-run ids in their own
    order ("force_own_order") reproduces the free run's logits bitwise.
    On the gate: the K8 fp32 port, T1 in one 1,536-token chunk (chunk=None).
    Returns {ok, values, reason}."""
    ids = np.asarray(ids).reshape(-1)
    chunk = chunk or ids.size
    free, rec = free_run(model, module, ids, chunk)
    ft = ForcedTrunk(module, num_layers_of(model), "force_own_order", table=rec["ids"])
    forced = forced_logits(model, ft, np.arange(ids.size), ids, chunk)
    equal = bitwise_equal(free, forced)
    diff = np.abs(free.astype(np.float64) - forced.astype(np.float64))
    shadow = ft.records()["shadow"]
    values = {"n_positions": int(ids.size), "chunk": int(chunk) if isinstance(chunk, (int, np.integer)) else "spans",
              "bitwise_equal": bool(equal), "max_abs_diff": float(diff.max()) if diff.size else 0.0,
              "rows_differing": int(np.any(free != forced, axis=-1).sum()),
              "shadow_set_disagreements": int(np.any(np.sort(shadow, -1) != np.sort(rec["ids"], -1), axis=-1).sum())}
    return {"ok": bool(equal), "values": values,
            "reason": None if equal else (f"forced onto its own ids in their own order, {values['rows_differing']} of "
                                          f"{ids.size} logit rows differ from the free run (max |diff| "
                                          f"{values['max_abs_diff']:.3e})")}


def p5c_forcing_active(model, module, ids, *, min_mean_kl: float, chunk=None) -> dict:
    """P5c (DESIGN §3.1): the port forced onto the ids of position t - 1 (layer by
    layer; position 0 keeps its own) moves the logits: mean KL(free || forced)
    over positions 1..T-1 must be >= min_mean_kl (thresholds.json; 0.10, the
    registered G4 gross-bug bound). KL by common.kl_rows. Returns {ok, values, reason}."""
    ids = np.asarray(ids).reshape(-1)
    if ids.size < 2:
        raise ValueError("P5c needs at least 2 positions")
    chunk = chunk or ids.size
    free, rec = free_run(model, module, ids, chunk)
    ft = ForcedTrunk(module, num_layers_of(model), "force_own_order", table=shift_ids(rec["ids"]))
    forced = forced_logits(model, ft, np.arange(ids.size), ids, chunk)
    kl = common.kl_rows(free[1:], forced[1:])
    mean = float(kl.mean())
    ok = mean >= float(min_mean_kl)
    values = {"n_positions": int(ids.size - 1), "mean_kl": mean, "min_mean_kl": float(min_mean_kl),
              "median_kl": float(np.median(kl)), "max_kl": float(kl.max())}
    return {"ok": bool(ok), "values": values,
            "reason": None if ok else f"forcing onto the ids of t - 1 moved mean KL only {mean:.3e} < {min_mean_kl}"}
