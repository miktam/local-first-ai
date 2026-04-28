# H4 closed without further test execution

## Status
The script ran successfully but found no input-token fields it
recognised in 17 session JSONL files. Strict reading of test contract:
"could_not_run". Substantive closure is on behavioural grounds below.

## Why H4 is closed without patching the jq filter

The strong form of H4 (OpenClaw silently caps every Ollama request
to 4096 input tokens regardless of contextWindow) is contradicted by
direct user behavioural evidence. The user has documented practice of
running `/compact` whenever Nestor becomes unresponsive — a workflow
that only makes sense if long prompts are actually being shipped to
Ollama and triggering the cliff observed in H6. If the strong form
of H4 were true, the model would never receive prompts long enough
to cause a runaway, and `/compact` would be unnecessary.

## Why the weak form is unevaluable

Most session JSONL files on the system carry `.deleted` or `.reset`
suffixes (a consequence of OpenClaw's session lifecycle and the
user's frequent `/compact` invocations). The remaining active
sessions are too few for statistical analysis of input-token
distribution.

## What H6 has already established

H6 directly measured that long prompts (15k, 25k, 35k tokens) reach
Ollama and produce the predicted prefill scaling. This is sufficient
to rule out the H4 (input cap) explanation regardless of weak/strong
form. Any residual H4 question is moot for the 003-Alpha root cause.

## What the failed jq filter actually tells us

The session schema exposes record types — `["type","id","parentId",
"timestamp","provider","modelId"]` and similar — but the first three
lines of any session are session-level metadata, not turn records.
Token-count fields (if present) would live on `assistant_message` or
`model_response` records deeper in the file. A future investigation
that needs OpenClaw turn-level instrumentation should:
  1. List record types: `jq -r '.type' session.jsonl | sort | uniq -c`
  2. Inspect the keys of each non-metadata type
  3. Locate token-count fields in the relevant type
  4. Patch the H4 jq filter to target that type and field

For 003-Alpha specifically, this is unnecessary.
