# Public copy of the exp_044 data

The synthetic writers were told to make contact details clearly fake, "for example `+34 600 000 1xx` or `name@example.com`". This public copy differs from the sealed originals in four ways, all made on 2026-10-19.

1. **Realistic numbers.** Four of the eight writers invented realistic-looking numbers: 26 in all. The first scan looked only for international numbers and found 11. A stricter check (`leads44.py validate`, `realistic_contacts`) found 15 more in national formats: Spanish mobiles without +34, French 06 numbers and a Belgian number. All 26 were replaced, and the public history was rewritten so that no commit holds them.
2. **Padded numbers are not fake either.** The brief's own example, and the `+34 600 000 9NN` numbers first used as replacements, have the format of ordinary Spanish mobile numbers. We found no range reserved for fiction in Spain. Every such number is now in [Ofcom's drama range](https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama), which is never allocated: `+34 600 000 NNN` → `+44 7700 900NNN`, keeping the last three digits, the same everywhere a number appears. Earlier commits still hold the padded numbers. The test fixtures in `kit/` are left as run.
3. **Numbers written out in words.** Seven messages spell their number out ("plus three four six hundred zero zero zero one one five"; X6-04 "sex nio sex, ett två tre, fyra fem sex", 696 123 456, a valid-format Spanish mobile). No digit scan sees these; exp_045's voice path turned one back into digits. They are now Ofcom numbers spelled the same way ("plus four four seven seven zero zero nine zero zero one one five"; X6-04 → 07700 900714): X1-18, X1-23, X1-48, X1-57, X1-60, X1-69, X6-04.
4. **A name.** One synthetic signature used a real public figure's full name. It is shortened to the first name.

**Results are unaffected:**
- Phone numbers in digits are masked to `[phone]` before Kev reads a message. For every message except the seven in item 3, Kev's masked input is identical before and after these edits, under rev 3.1 and rev 4 (checked by code). The masker does not see numbers in words, so Kev read those seven numbers in the run; the recorded predictions come from the originals.
- The code-side facts are identical for every message.
- No label touches a phone number or a name.
- `leads44.py compare test` on this copy reproduces `results/test/compare.json` exactly.

**Hashes:** the `.sha256` files and the hashes in the records (`57aa2dd9…`, `d237a277…`, `bf54b4d0…`) are of the original, sealed files. This copy's hashes are:

- `agreed.jsonl`: `47fb4a057ef0613a8d7d8fefbc24db02890318d699d11b53a94601c7c6e0a241`
- `dev.jsonl`: `65ef5ac9da625b1eb0b16e83adb171b85102e3ff8c8c923b167625c10caf18b5`
- `test.jsonl`: `514b6386dfba45464089a472d579097df58c3380964c1203aab9a39deddae808`
