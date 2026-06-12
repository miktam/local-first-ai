### Phase A — Model Selection Summary (Exp 016)

**Goal**: Select the best “smart tier” model for the MBP (M5 Max 128 GB) using MLX that offers a meaningful quality upgrade over `gemma4:26b` while maintaining good speed (>20 tok/s).

#### Recommended Models

| Priority              | Model                                      | Type                          | Approx. Memory | Recommendation          | Notes |
|-----------------------|--------------------------------------------|-------------------------------|----------------|-------------------------|-------|
| **Control**           | `mlx-community/gemma-4-26b-a4b-it-4bit`   | MoE (26B total / 4B active)  | ~13–15 GB     | Must run               | Direct baseline vs mini’s gemma4:26b |
| **Primary**           | `mlx-community/Qwen3.5-122B-A10B-4bit`    | MoE (122B total / ~10B active) | ~50 GB      | **Strongly recommended** | Best balanced smart tier model |
| **Strong Coding Option** | `mlx-community/Qwen3-Coder-Next-4bit`  | MoE (80B total / 3B active)  | ~40 GB        | **Test as well**       | Coding-specialized, potentially strongest on benchmark task |
| **Strong Alternative**| `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` | MoE (~109B total / 17B active) | ~55–65 GB | Optional               | Excellent generalist |

#### Final Recommendation

- **Run first**: `Qwen3.5-122B-A10B-4bit` as the main balanced candidate.
- **Also test**: `Qwen3-Coder-Next-4bit` — strong contender due to its coding specialization.
- **Control**: Always run `gemma-4-26b-a4b-it-4bit` first.
- **Optional**: `Llama-4-Scout-17B-16E-Instruct-4bit` if time allows.

#### Why These Models?

- All are ready-to-use 4-bit MLX models from `mlx-community`.
- They fit comfortably in 128 GB with good headroom.
- `Qwen3.5-122B-A10B-4bit` offers the best balance of general reasoning and coding capability.
- `Qwen3-Coder-Next-4bit` is purpose-built for coding and may deliver the highest quality on the specific benchmark task.
- All should comfortably exceed 20 tok/s on the M5 Max.

#### Expected Outcome

- **H1** (capability threshold): Very likely to pass.
- **H2** (quality differential): High probability that either the Qwen3.5-122B or Qwen3-Coder-Next will score noticeably higher than the gemma-4 control.

#### Suggested Testing Order

1. Control model (`gemma-4-26b-a4b-it-4bit`)
2. `Qwen3.5-122B-A10B-4bit` (primary balanced candidate)
3. `Qwen3-Coder-Next-4bit` (coding specialist)
4. `Llama-4-Scout-17B-16E-Instruct-4bit` (optional)