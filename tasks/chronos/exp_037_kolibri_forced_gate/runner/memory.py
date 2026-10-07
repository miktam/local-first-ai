"""Memory rule, batch size B and the Metal limit (HYPOTHESIS "Pilot and the
rule that fixes n", rule 1 and "K8 working-set rule"; BUILD_SPEC §5.4
runner/memory.py; plan_rules.json).

  need(B) = weight_bytes + 2 · B · (prompt_max + cap) · kv_bytes_per_token
            + B · fixed_window_bytes + 4 GiB
  B       = the largest of {16, 8, 4, 2, 1} with need(B) <= 0.9 · L
            (AIME cells at most 8); 0 means not even B = 1 fits.
  L       = max(mx.device_info()["max_recommended_working_set_size"],
                iogpu.wired_limit_mb · 2^20)

The factor 2 covers the transient copy when the batch KV cache grows or
admits a sequence (every row is padded to the longest live row). weight_bytes
is the summed size of the arm directory's *.safetensors files. The kit never
calls sudo; sysctl_advice() only prints the line for Andrei.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from runner.common import ARMS, arm_dir, load_rules, sysctl, task_kind

GiB = 1 << 30
MiB = 1 << 20


def _device_info() -> dict:
    import mlx.core as mx

    return mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()


def effective_limit(
    sysctl_reader: Callable[[str], str | None] = sysctl,
    device_info: Callable[[], dict] = _device_info,
) -> tuple[int, dict]:
    """(L_bytes, sources). Reads iogpu.wired_limit_mb via `sysctl -n`, no sudo."""
    di = device_info()
    rec = int(di.get("max_recommended_working_set_size", 0))
    raw = sysctl_reader("iogpu.wired_limit_mb")
    try:
        mb = int(str(raw).strip()) if raw is not None else 0
    except ValueError:
        mb = 0
    wired = mb * MiB
    L = max(rec, wired)
    return L, {
        "max_recommended_working_set_size": rec,
        "iogpu_wired_limit_mb": mb,
        "iogpu_bytes": wired,
        "chosen": "iogpu.wired_limit_mb" if wired > rec else "max_recommended_working_set_size",
    }


def arm_weight_bytes(model_dir: Path) -> int:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    return sum(f.stat().st_size for f in files)


def kolibri_kv_from_config(cfg: dict) -> tuple[int, int]:
    """(growing bytes per token, fixed bytes per sequence) for Kolibri in bf16:
    10 full layers x 2 (k, v) x 4 kv heads x 128 x 2 B = 20,480 B/token, and
    40 sliding layers x 513 x 2 x 4 x 128 x 2 B = 42,024,960 B (spec item 20)."""
    types = cfg["layer_types"]
    n_full = sum(1 for t in types if t == "full_attention")
    n_slide = len(types) - n_full
    per_layer_token = 2 * int(cfg["num_key_value_heads"]) * int(cfg["head_dim"]) * 2
    return n_full * per_layer_token, n_slide * int(cfg["sliding_window"]) * per_layer_token


def kv_params(arm: str, rules: dict | None = None, config: dict | None = None) -> tuple[int, int]:
    """(kv_bytes_per_token, fixed_window_bytes) for an arm.

    Order: tools/kv_bytes.py on the arm's config (the kit's one definition)
    if both exist; Kolibri's own config arithmetic; plan_rules.json values."""
    rules = rules or load_rules()
    family = ARMS[arm]["family"]
    if config is None:
        p = arm_dir(arm) / "config.json"
        if p.is_file():
            config = json.loads(p.read_text(encoding="utf-8"))
    if config is not None:
        try:
            from tools import kv_bytes as kvb  # sibling area; lazy

            g, f = kvb.kv_bytes_per_token(config)
            return int(g), int(f)
        except ModuleNotFoundError:
            pass
        if family == "kolibri" and "layer_types" in config:
            return kolibri_kv_from_config(config)
    return int(rules["kv_bytes_per_token"][family]), int(rules["fixed_window_bytes"][family])


def need_bytes(
    weight_bytes: int,
    B: int,
    prompt_max: int,
    cap: int,
    kv_bytes_per_token: int,
    fixed_window_bytes: int,
    *,
    overhead_gib: float = 4.0,
    transient_factor: float = 2.0,
) -> int:
    return int(
        weight_bytes
        + transient_factor * B * (prompt_max + cap) * kv_bytes_per_token
        + B * fixed_window_bytes
        + overhead_gib * GiB
    )


def prompt_max_for(task: str, rules: dict) -> int:
    pm = rules["prompt_max_tokens"]
    return int(pm.get(task, pm[task_kind(task)]))


def choose_B(
    arm: str,
    task: str,
    cap: int,
    L: int,
    *,
    weight_bytes: int | None = None,
    prompt_max: int | None = None,
    kv_bytes_per_token: int | None = None,
    fixed_window_bytes: int | None = None,
    rules: dict | None = None,
) -> int:
    """Largest B in plan_rules B_choices with need(B) <= limit_fraction · L;
    AIME cells capped at B_max_aime; 0 if even B = 1 does not fit."""
    rules = rules or load_rules()
    if weight_bytes is None:
        weight_bytes = arm_weight_bytes(arm_dir(arm))
    if kv_bytes_per_token is None or fixed_window_bytes is None:
        g, f = kv_params(arm, rules)
        kv_bytes_per_token = g if kv_bytes_per_token is None else kv_bytes_per_token
        fixed_window_bytes = f if fixed_window_bytes is None else fixed_window_bytes
    if prompt_max is None:
        prompt_max = prompt_max_for(task, rules)
    budget = float(rules["limit_fraction"]) * L
    for B in sorted(rules["B_choices"], reverse=True):
        if task_kind(task) == "aime" and B > int(rules["B_max_aime"]):
            continue
        need = need_bytes(
            weight_bytes, B, prompt_max, cap, kv_bytes_per_token, fixed_window_bytes,
            overhead_gib=float(rules["memory_overhead_gib"]),
            transient_factor=float(rules["kv_transient_factor"]),
        )
        if need <= budget:
            return B
    return 0


def sysctl_advice(
    k8_cells: Iterable,
    L: int,
    *,
    weight_bytes: int | None = None,
    rules: dict | None = None,
    arm: str = "K8",
) -> str | None:
    """The `sudo sysctl iogpu.wired_limit_mb=114688` line, with the B it would
    gain, when L' = 112 GiB would raise some K8 cell's B, or when need(1) >
    0.9 · L for some K8 cell. None otherwise. k8_cells: (task, cap) pairs or
    dicts with task, cap and optionally prompt_max."""
    rules = rules or load_rules()
    if weight_bytes is None:
        weight_bytes = arm_weight_bytes(arm_dir(arm))
    L2 = int(rules["sysctl_advice_mb"]) * MiB
    gains, unfit = [], []
    for c in k8_cells:
        task, cap, pm = (c["task"], c["cap"], c.get("prompt_max")) if isinstance(c, dict) else (c[0], c[1], None)
        b_now = choose_B(arm, task, cap, L, weight_bytes=weight_bytes, prompt_max=pm, rules=rules)
        b_new = choose_B(arm, task, cap, L2, weight_bytes=weight_bytes, prompt_max=pm, rules=rules) if L2 > L else b_now
        if b_now == 0:
            unfit.append(f"{arm} {task} cap {cap}: does not fit at B = 1 (B at 112 GiB: {b_new})")
        elif b_new > b_now:
            gains.append(f"{arm} {task} cap {cap}: B {b_now} -> {b_new}")
    if not gains and not unfit:
        return None
    line = f"sudo sysctl iogpu.wired_limit_mb={int(rules['sysctl_advice_mb'])}"
    why = "; ".join(unfit + gains)
    return f"{line}    # 112 GiB, leaves ~16 GiB for macOS; resets at reboot. {why}"
