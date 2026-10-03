# Peer chat templates

The runner renders every peer prompt with these files (`runner/chat.py`), never with whatever template a model folder happens to carry. Each is the upstream `chat_template.jinja`, copied **unmodified** (byte for byte), so the file carries no header; provenance is here, in `MANIFEST.json` (sha256, bytes, source) and in `NOTICE`.

| File | Source | Revision | Bytes | sha256 | Same as the mlx-community builds |
|---|---|---|---|---|---|
| `gemma4.jinja` | google/gemma-4-26B-A4B-it | `20da991ab4afab98e8f910c4a2e8f4fbefc404ad` | 17466 | `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0` | yes (8-bit and 4-bit) |
| `qwen3_6.jinja` | Qwen/Qwen3.6-35B-A3B | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 7764 | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` | yes |
| `qwen3_8.jinja` | Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 8952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | yes |

Licence: Apache-2.0, per each upstream repository's metadata. Copyright stays with the upstream authors (Google; the Qwen team, Alibaba Cloud). The Apache-2.0 text is `../../LICENSE-APACHE-2.0`.

Fetched on the build host (the mini) on 2026-10-03, about 18:14 UTC, by naming each file positionally.

**Gemma 4, a known difference.** Upstream `main` at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (header dated 2026-07-09) carries a newer template: tool-call argument handling, turn-closure and thinking-order fixes, and `enable_thinking | default(false)` in place of `enable_thinking is defined and enable_thinking`. For the prompts the kit sends (one user message, or a system message plus one user message, with `enable_thinking` passed explicitly as True or False) the two versions render byte-identical prompts; this was checked on the mini on 2026-10-03. The pre-registered pin is `20da991a` (HYPOTHESIS, New assets), which is also what the mlx-community builds ship, so that is the committed copy.

**Kwargs the runner passes** (HYPOTHESIS, Reasoning; C9):
- Gemma 4: `enable_thinking=True`. The template prefills an empty, closed thought channel only when thinking is off, so with thinking on the model may or may not open `<|channel>thought`.
- Qwen3.6: `enable_thinking=True`; the prompt ends with an open `<think>\n`.
- Qwen3.8: `enable_thinking=True, reasoning_effort="xhigh"`. `xhigh` is the template's own default (`reasoning_effort|default('xhigh')`), passed explicitly so that nothing depends on a default; the render is identical to the default render (`tests/test_runner_chat.py`).
