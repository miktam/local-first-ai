"""Total and active parameters (HYPOTHESIS C16, "Arms": recomputed from each
config at preflight and written to the version record; BUILD_SPEC §5.9
peer check: parameter count against config arithmetic, ≤ 2 %).

`config_params(model_dir)` is the config arithmetic: it builds the model's
module tree from config.json with mlx_lm's own class for that model_type (or
the port's `kolibri1.py` for Kolibri) and counts parameter shapes. MLX arrays
are lazy, so nothing is allocated and no weights are read. The count is the
text model mlx_lm runs (vision and audio towers are not part of it).

`loaded_params(model)` counts a loaded, possibly quantised model logically:
a quantised weight of b bits counts 32/b values per stored uint32; its scales
and biases are not parameters.

Active = total − routed-expert parameters × (1 − top_k / n_experts). The
input embedding is counted in `active`; `active_excl_input_embedding`
subtracts it when it is not tied to the head (the Kolibri card's 3.46B is of
that kind).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common

PORT_FILE = common.EXP_DIR / "port" / "kolibri1.py"
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
TOPK_KEYS = ("num_experts_per_tok", "top_k_experts", "moe_top_k", "num_experts_per_token")


def _text_cfg(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc else cfg


def _first(d: dict, keys) -> int | None:
    for k in keys:
        if d.get(k) is not None:
            return int(d[k])
    return None


def _model_classes(cfg: dict, model_dir: Path):
    if cfg.get("model_type") == "kolibri1":
        port = Path(model_dir) / "kolibri1.py"
        port = port if port.is_file() else PORT_FILE
        spec = importlib.util.spec_from_file_location("exp036_kolibri1_params", port)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Model, mod.ModelArgs
    from mlx_lm.utils import _get_classes

    return _get_classes(cfg)


def _count(flat, n_experts: int | None) -> tuple[int, int, int]:
    """(total, routed expert params, input embedding params)."""
    total = routed = embed = 0
    for name, size, shape in flat:
        total += size
        if n_experts and shape and shape[0] == n_experts and (
            "switch" in name or ".experts." in name
        ):
            routed += size
        if name.endswith("embed_tokens.weight"):
            embed += size
    return total, routed, embed


def _summary(total: int, routed: int, embed: int, n_experts, top_k, tied: bool) -> dict:
    active = total - routed + (routed * top_k // n_experts if n_experts and top_k else routed)
    return {
        "total": total, "active": active,
        "active_excl_input_embedding": active - (0 if tied else embed),
        "routed_expert_params": routed, "input_embedding_params": embed,
        "num_experts": n_experts, "top_k": top_k, "tied_embeddings": tied,
    }


def config_params(model_dir) -> dict:
    common.set_offline_env()
    from mlx.utils import tree_flatten

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    Model, Args = _model_classes(cfg, model_dir)
    model = Model(Args.from_dict(cfg))
    flat = [(k, v.size, tuple(v.shape)) for k, v in tree_flatten(model.parameters())]
    tc = _text_cfg(cfg)
    n_exp, top_k = _first(tc, EXPERT_KEYS), _first(tc, TOPK_KEYS)
    tied = bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    out = _summary(*_count(flat, n_exp), n_exp, top_k, tied)
    out["model_type"] = cfg.get("model_type")
    out["method"] = "config arithmetic: mlx_lm module tree built from config.json (lazy, no weights)"
    return out


def loaded_params(model, n_experts: int | None = None, top_k: int | None = None, tied: bool = False) -> dict:
    """Logical parameter count of a loaded model (quantised weights unpacked)."""
    from mlx.utils import tree_flatten

    qbits = {}
    for name, mod in model.named_modules():
        bits = getattr(mod, "bits", None)
        if isinstance(bits, int) and hasattr(mod, "scales"):
            qbits[name] = bits
    flat = []
    for name, v in tree_flatten(model.parameters()):
        prefix, _, leaf = name.rpartition(".")
        if prefix in qbits:
            if leaf in ("scales", "biases"):
                continue
            if leaf == "weight":
                shape = tuple(v.shape[:-1]) + (v.shape[-1] * 32 // qbits[prefix],)
                size = 1
                for s in shape:
                    size *= s
                flat.append((name, size, shape))
                continue
        flat.append((name, v.size, tuple(v.shape)))
    return _summary(*_count(flat, n_experts), n_experts, top_k, tied)
