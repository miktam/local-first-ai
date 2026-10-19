# Public copy of the exp_043 data

The synthetic writers were told to make contact details clearly fake, "for example `+34 600 000 1xx` or `name@example.com`". They kept to padded numbers. A padded number only looks fake, though. `+34 600 000 NNN` has the format of an ordinary Spanish mobile number, and we found no range reserved for fiction in Spain.

On 2026-10-19, after first publication (`370341c`), every padded number in this folder was moved to [Ofcom's drama range](https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama), which is never allocated:
- `+34 600 000 NNN` → `+44 7700 900NNN`, keeping the last three digits, the same everywhere a number appears;
- nine padded Belgian, French and Colombian numbers → `+44 7700 900700`–`708`;
- `+44 7700 900123` was already in Ofcom's range and is unchanged.

Earlier commits still hold the padded numbers. The test fixture in `kit/` is left as run.

**Results are unaffected:**
- Phone numbers are masked to `[phone]` before Kev reads a message. For every message, Kev's masked input and the code-side facts are identical before and after (checked by code).
- No label touches a phone number.
- `leads43.py score test` on this copy reproduces `results/test/score.json` exactly.

**Hashes:** the `.sha256` files and the hashes in the records (`71014cc8…`, `614141b1…`, `8495fbd6…`) are of the original, sealed files. This copy's hashes are:

- `agreed.jsonl`: `b1ffda490c1fbc7b6f50ea039ae067a0a387182ffc9dfe12f41bf1a96602d3d7`
- `dev.jsonl`: `fb03b82cd855bcfb622ba9fa6e647977c9046f8119955059bc152db1f6e284be`
- `test.jsonl`: `fba36c5a3293fb8a94027c0af9863b9292a1bfa958ef528811d598f40c6fd6ca`
