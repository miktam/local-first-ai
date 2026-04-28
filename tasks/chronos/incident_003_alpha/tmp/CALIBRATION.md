# Tokenizer calibration

Empirical chars-per-token for `gemma4-think:26b` on Latin filler:

  source: evidence/2026-04-28T07-34-59Z-H5/repeat_1.json (pre-reset)
  prompt bytes: 160000
  prompt_eval_count: 23305
  ratio: 6.866 chars/token

`generate_filler_prompt` uses 7 chars/token (rounded up) so target
token counts are reached or slightly exceeded, never undershot.
