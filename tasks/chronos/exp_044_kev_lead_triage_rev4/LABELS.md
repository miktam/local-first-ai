# exp_044 lead labels: definitions (shared by writers, checkers and the scorer)

*exp_043's definitions unchanged, plus `budget_eur`. Spam's action is now `review`, the name the code has used since exp_043; the decision is the same. Urgency is unchanged (Andrei, 2026-10-19).*

Each lead is one inbound message to a Costa del Sol estate agency (WhatsApp, portal enquiry form, e-mail, or a transcribed voice note). Each label is decided **from the message text alone**. Never guess beyond the text.

| Field | Values | Rule |
|---|---|---|
| `intent` | `buy`, `rent_long`, `rent_holiday`, `sell`, `valuation`, `invest`, `info`, `spam` | The sender's main goal. See the intent notes below. |
| `property_type` | `apartment`, `penthouse`, `townhouse`, `villa`, `plot`, `commercial`, `unspecified` | The kind of property the message is about. `unspecified` if none is named. A "flat" or "piso" is an apartment; a "house" or "Haus" with no other detail is a villa. |
| `timeline` | `now`, `months`, `later`, `unspecified` | `now`: within about a month, urgent, or already visiting or ready to sign. `months`: within the next months or this year. `later`: next year or beyond. `unspecified`: no timing given. |
| `financing` | `true`, `false`, `null` | `true`: needs a mortgage or loan. `false`: says cash, or no mortgage needed. `null`: not mentioned. |
| `foreign_buyer` | `true`, `false`, `null` | `true`: clearly a foreigner or non-resident in Spain. `false`: clearly a Spanish resident. `null`: cannot tell. Writing in a foreign language alone does **not** decide it. |
| `qualified` | `true`, `false` | `true` iff the goal is clear **and** at least two of budget, area and timing are stated. Always `false` for `info` and `spam`. |
| `urgency` | `0`, `1`, `2`, `3` | How fast an agent should reply. See the urgency scale below. |
| `action` | `call_now`, `send_listings`, `ask_details`, `book_valuation`, `answer_question`, `review` | The agency's next step. Apply the action rules below **in order**; the first that fits decides. |
| `budget_eur` | a number, or `null` | The largest budget the message states, in euros: the total price for buying, selling, investing or a valuation; the monthly rent for rentals. A range counts at its upper end ("800k to 1M" → 1000000). Written in words counts too ("medio millón" → 500000). `null` if no budget is stated. Ignore prices that are not the sender's budget, such as a supplier's fee in a sales pitch. |

**Intent notes:**
- `rent_long` is 6 months or more; `rent_holiday` is weeks.
- `valuation` means they want a price estimate. Use `sell` when they want to list or sell; if they say both, `sell`.
- `invest` means yield, development, land banking or several units.
- `buy` includes a holiday or second home they want to own, even when they describe it with holiday words ("for our summers", "flying in for the week").
- `info` is a question about law, taxes, process or an area, with no transaction asked for.
- `spam` covers suppliers, advertising, scams, other agencies prospecting and anything that is not a client. A business acting for a genuine client (a relocation firm renting for staff, a lawyer or family office buying for a client, a developer looking for land) is a client, not spam.

**Urgency scale:**
- `0`: no rush. Info, spam, or a timeline of later or unspecified with no hot signal.
- `1`: within a day. A real client with a months timeline.
- `2`: within hours. Qualified with timeline `now`, or a specific property asked about.
- `3`: right now. Visiting today or tomorrow, an offer or a signature imminent, or another agency competing.

**Action rules:**
1. `spam` → `review`.
2. `info` → `answer_question`.
3. `valuation`, or `sell` when they ask what the property is worth → `book_valuation`.
4. Qualified, with timeline `now` and a phone number or a request to call → `call_now`.
5. Qualified → `send_listings`. For `sell` without a valuation request → `book_valuation`.
6. Otherwise → `ask_details`.

**Clarifications (2026-10-19, after the writers and before any checker label):**
- `budget_eur` for a holiday rental is the amount stated for the stay ("9.000 for the fortnight" → 9000), because no monthly figure exists. All eight writers did this. Long-term rentals stay monthly.
- `timeline`: a date in January 2027 or later is next year, so `later`, even if it is only weeks away. This is the existing rule taken literally; most writers applied it, but the Swedish batch called December–February `months`. Where writer and checker disagree, timeline is excluded for that lead, as pre-registered.

