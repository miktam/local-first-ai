# Test fixtures

Two copies of the Kolibri 1 chat template, used by `../test_chat_template.py`.
Both are configuration files of the model, published by Aleph Alpha GmbH under the Apache License 2.0. They are copied here unmodified. The licence text is `../../LICENSE-APACHE-2.0`, and attribution is in `../../NOTICE`.

| File | Source | Bytes | sha256 |
|---|---|---|---|
| `kolibri1_chat_template.tokenizer_config.jinja` | The `chat_template` string in `tokenizer_config.json` of Aleph-Alpha/Kolibri-1-BF16 at revision `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, written out verbatim with no trailing newline. The Hugging Face API metadata of Aleph-Alpha/Kolibri-1 (FP8) at `e52eb4627d11516b0c01de49210ab5a4e4061444` carries the same string. | 6236 | `9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97` |
| `kolibri1_chat_template.vendor.jinja` | `tests/kolibri1_chat_template.jinja` of aleph-alpha-inference 1.0.0, commit `049a6a7bd2405b27d6d280d256bd3d585191c7ae`. It is the same template with a leading `{#- … -#}` comment, which renders to nothing. | 6478 | `51b9ae6f83e7a30d428fccc75da306653801be73d4ed936e838667d5de7a6617` |

Neither repo ships a separate `chat_template.jinja`, so the string in `tokenizer_config.json` is the template that transformers and mlx_lm apply.

If the model's template changes, replace the first file and rerun the tests. Do not edit either file by hand.
