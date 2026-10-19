# Public copy of the exp_044 data

The synthetic writers were told to make contact details clearly fake, "for example `+34 600 000 1xx` or `name@example.com`". This public copy differs from the sealed originals in three ways, all made on 2026-10-19.

1. **Realistic numbers.** Four of the eight writers invented realistic-looking numbers: 26 in all. The first scan looked only for international numbers and found 11. A stricter check (`leads44.py validate`, `realistic_contacts`) found 15 more in national formats: Spanish mobiles without +34, French 06 numbers and a Belgian number. All 26 were replaced, and the public history was rewritten so that no commit holds them.
2. **Padded numbers are not fake either.** The brief's own example, and the `+34 600 000 9NN` numbers first used as replacements, have the format of ordinary Spanish mobile numbers. We found no range reserved for fiction in Spain. Every such number is now in [Ofcom's drama range](https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama), which is never allocated: `+34 600 000 NNN` → `+44 7700 900NNN`, keeping the last three digits, the same everywhere a number appears. Earlier commits still hold the padded numbers. The test fixtures in `kit/` are left as run.
3. **A name.** One synthetic signature used a real public figure's full name. It is shortened to the first name.

**Results are unaffected:**
- Phone numbers are masked to `[phone]` before Kev reads a message. For every message, Kev's masked input and the code-side facts are identical before and after these edits, under rev 3.1 and rev 4 (checked by code).
- No label touches a phone number or a name.
- `leads44.py compare test` on this copy reproduces `results/test/compare.json` exactly.

**Hashes:** the `.sha256` files and the hashes in the records (`57aa2dd9…`, `d237a277…`, `bf54b4d0…`) are of the original, sealed files. This copy's hashes are:

- `agreed.jsonl`: `18a4d24430a4dd11f70ce3edb1a587b05362d82ba241367362eac14b652d3b90`
- `dev.jsonl`: `04910eace46d8c2e0713cdcc39b6536bb80a05f10762f7c25ec4cf32a2042414`
- `test.jsonl`: `bfaff0421eea52a57443104630e6948e59f60d57ffe853744db8693cdb2887c2`
