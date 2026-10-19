# Public copy of the exp_044 data

The synthetic writers were told to use clearly fake contact details; some invented realistic-looking numbers instead. In this public copy, 26 such numbers are replaced with clearly fake ones (`+34 600 000 9NN`), the same replacement everywhere a number appears. One synthetic signature used a real public figure's full name; it is shortened to the first name.

**Results are unaffected:**
- Phone numbers are masked to `[phone]` before Kev reads a message, so Kev's input is identical.
- The code-side fact "has a phone" is unchanged.
- No label touches a phone number or a name.

**Second pass (2026-10-19, same day):** a stricter check (`leads44.py validate`, `realistic_contacts`) found 15 more in national formats the first scan missed: Spanish mobiles without +34, French 06 numbers, and a Belgian number. They are replaced the same way (`+34 600 000 912`–`926`). The check now runs at writing time, so future sets are caught before anything is published.

**Hashes:** the `.sha256` files and the hashes in the records (`57aa2dd9…`, `d237a277…`, `bf54b4d0…`) are of the original, sealed files. This copy's hashes are:

- `agreed.jsonl`: `98654ca9c354876c93d41c6ed4fd576e5c4ef88740dd840e23e59d524f14415f`
- `dev.jsonl`: `c2cd65128d1585b80193552ba266cb2304992820c5f472a6220cadc866aa7e3a`
- `test.jsonl`: `0a80f4fd9c488cfee69d9d56025bef6f283bd556e29839148cfbda9e74ff43ad`
