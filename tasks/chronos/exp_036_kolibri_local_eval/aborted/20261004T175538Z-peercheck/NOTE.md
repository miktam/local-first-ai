# RUNBOOK step 9 re-run after Amendment 5: Gemma 4 fails the fidelity rule on chat-wrapped texts (2026-10-04)

## The Amendment 5 checks pass (mbp)

- The sync was append-only (+67) and equals `amendments/5_measurement_20261004T164502Z.md`. Pushed as `77815fe`.
- Step 3: 1234 passed + 8 `build-host only:` skips = 1242.
- `hash_tree --check`: 17 of 17. runner, bench and tools match via Amendment 5.

## Step 9, re-run

The mbp Claude session ran it at Andrei's request, 17:42:58–17:55:38Z, as the RUNBOOK gives it. The exit code was 1. The record is `results/peers_20261004T175538Z.json`, committed with this note.

```text
G8     fail NLL(8) / KL(8‖4) rule failed; greedy flip rate 0.033 > 0.02
G4     fail NLL(8) / KL(8‖4) rule failed
Q36-8  B=1  batched-path parity outside the G5 noise-floor bound; greedy flip rate 0.033 > 0.02
Q36-4  ok
Q38-8  ok
Q38-4  ok
```

- **Q36-8** is `B=1` as Amendment 5 intends. Both its problems are under `batched_path_problems`. It is not a drop.
- **G8 and G4** fail on the family fidelity rule. G8's flip (1 in 30) alone would be `B=1`.

## Chat-wrapped fidelity (`families.<f>.fidelity`, used for the verdict)

| family | NLL(8) | NLL(4) | NLL(8) ≤ NLL(4) + 0.02 | KL(8‖4) | KL < 0.2 |
|---|---|---|---|---|---|
| gemma4 | 2.938 | 2.910 | **no**: 2.938 > 2.930, misses by 0.008 | **0.323** | **no** |
| qwen3_6 | 1.882 | 1.935 | yes | 0.095 | yes |
| qwen3_8 | 1.787 | 1.848 | yes | 0.069 | yes |

- Gemma's chat-wrapped NLL is in the main session's expected 2–4 nats/token range. The measurement now behaves.
- Per text, G8 / G4: T1 4.10 / 4.03, T2 3.20 / 3.20, T3 2.16 / 2.14, T4 2.28 / 2.20, T5 3.34 / 3.32, T6 2.84 / 2.85.
- The 8-bit build is 0.028 nats/token **worse** than the 4-bit build. In both Qwen families, 8-bit is better than 4-bit by 0.05–0.06.
- The wrapper's prompt ids are identical for both Gemma builds (`prompt_ids_4bit_identical: true`). The prefill is `empty_channel`, with `enable_thinking` false.
- The raw-text values (NLL 10.21 / 10.66, KL 4.57) remain in `fidelity_raw_text`. Since Amendment 5 they are descriptive only.

## A decision is needed

Under the registered rule, this is a `fail` for both Gemma builds. The registered consequences are:
- **HYPOTHESIS, settled case 7:** if G4 is dropped, H1 is NOT RUN; a dropped family leaves H8's peer median.
- **RUNBOOK step 9:** H3 and H6 then use the remaining MoE peer (Qwen3.6).

What a failed G4 does to H1 or H8 beyond that is for Andrei to decide by amendment, before step 13.

The alternative is for the main session to judge whether the G8 result (8-bit worse than 4-bit, KL 0.32) points to a problem with the G8 build or the 8-bit path, rather than a property of Gemma 4.

The mbp waits at step 9. Nothing else was run.
