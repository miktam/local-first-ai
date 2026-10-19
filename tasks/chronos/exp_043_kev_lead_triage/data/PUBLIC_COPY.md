# Public copy of the exp_043 data

The synthetic writers were told to make contact details clearly fake, "for example `+34 600 000 1xx` or `name@example.com`". They kept to padded numbers. A padded number only looks fake, though. `+34 600 000 NNN` has the format of an ordinary Spanish mobile number, and we found no range reserved for fiction in Spain.

On 2026-10-19, after first publication (`370341c`), every padded number in this folder was moved to [Ofcom's drama range](https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama), which is never allocated:
- `+34 600 000 NNN` → `+44 7700 900NNN`, keeping the last three digits, the same everywhere a number appears;
- nine padded Belgian, French and Colombian numbers → `+44 7700 900700`–`708`;
- `+44 7700 900123` was already in Ofcom's range and is unchanged.

Seven messages spell their padded number out in words ("seis cero cero, cero cero cero, uno cero uno", "six cero cero, …"), which no digit scan sees; exp_045's voice path surfaced the pattern. They are now Ofcom numbers spelled the same way ("cero siete siete cero cero, nueve cero cero, uno cero uno"): W2-23, W2-50, W8-05, W8-28, W8-34, W8-46, W8-52.

Earlier commits still hold the padded numbers. The test fixture in `kit/` is left as run.

**Results are unaffected:**
- Phone numbers in digits are masked to `[phone]` before Kev reads a message. For every message except the seven spelled out in words, Kev's masked input is identical before and after (checked by code). The masker does not see numbers in words, so Kev read those seven in the run; the recorded predictions come from the originals.
- The code-side facts are identical for every message.
- No label touches a phone number.
- `leads43.py score test` on this copy reproduces `results/test/score.json` exactly.

**Hashes:** the `.sha256` files and the hashes in the records (`71014cc8…`, `614141b1…`, `8495fbd6…`) are of the original, sealed files. This copy's hashes are:

- `agreed.jsonl`: `e334d0ee999fb8d9ec9fe840b44906dac823e9a4113dca43716fcbd9aefcf487`
- `dev.jsonl`: `f94a3794189af0b50e1c8287243d9dada0edaa0c4946704a9a6b21625f72da0f`
- `test.jsonl`: `1c055277b672cafe6ddb163b5d7cd98cf7b8067bd6c580c49a8b6b5d3eeb652f`
