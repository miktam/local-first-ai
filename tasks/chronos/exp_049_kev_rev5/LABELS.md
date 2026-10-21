# exp_049 lead labels: definitions (shared by writers, checkers and the scorer)

*Version 3. exp_044's definitions, with three changes:*
- *a new label, `on_behalf`;*
- *the timeline rule stated against an explicit date;*
- *exp_044's two clarifications folded in.*

*The urgency scale and the action rules are unchanged (Andrei, 2026-10-19).*

**Every message arrives on Wednesday 21 October 2026.** Each lead is one inbound message to a Costa del Sol estate agency: WhatsApp, a portal enquiry form, an e-mail, or a transcribed voice note. Each label is decided **from the message text alone**. Never guess beyond the text.

| Field | Values | Rule |
|---|---|---|
| `intent` | `buy`, `rent_long`, `rent_holiday`, `sell`, `valuation`, `invest`, `info`, `spam` | The sender's main goal. See the intent notes below. |
| `property_type` | `apartment`, `penthouse`, `townhouse`, `villa`, `plot`, `commercial`, `unspecified` | The kind of property the message is about. `unspecified` if none is named. A "flat" or "piso" is an apartment; a "house" or "Haus" with no other detail is a villa. |
| `timeline` | `now`, `months`, `later`, `unspecified` | **Today is 21 October 2026.** `now`: within about a month (by about 21 November 2026), urgent, or already visiting or ready to sign. `months`: later in 2026, by 31 December. `later`: 2027 or beyond, **even if only weeks away** ("from January" is `later`). `unspecified`: no timing given. For a holiday rental, the timeline is when the stay starts. |
| `financing` | `true`, `false`, `null` | `true`: uses a mortgage or loan, already approved or not. `false`: says cash, or no mortgage needed. `null`: not mentioned. |
| `foreign_buyer` | `true`, `false`, `null` | `true`: clearly a foreigner or non-resident in Spain. `false`: clearly a Spanish resident. `null`: cannot tell. Writing in a foreign language alone does **not** decide it. |
| `on_behalf` | `true`, `false` | `true`: the sender writes for someone else who is the actual client. For example a lawyer, notary or family office for a buyer; a relocation firm or employer for staff; an assistant for an executive; a relative for a family member. `false`: the sender is the client, or is selling something to the agency. A pitch is never `on_behalf`. |
| `qualified` | `true`, `false` | `true` iff the goal is clear **and** at least two of budget, area and timing are stated. Always `false` for `info` and `spam`. |
| `urgency` | `0`, `1`, `2`, `3` | How fast an agent should reply. See the urgency scale below. |
| `action` | `call_now`, `send_listings`, `ask_details`, `book_valuation`, `answer_question`, `review` | The agency's next step. Apply the action rules below **in order**; the first that fits decides. |
| `budget_eur` | a number, or `null` | The largest budget the message states, in euros. That is the total price for buying, selling, investing or a valuation; the monthly rent for long-term rentals; the amount stated for the stay for holiday rentals ("9.000 for the fortnight" → 9000). A range counts at its upper end ("800k to 1M" → 1000000). Written in words counts too ("medio millón" → 500000). `null` if no budget is stated. Ignore prices that are not the sender's budget, such as a supplier's fee in a sales pitch. |

**Intent notes:**
- `rent_long` is 6 months or more; `rent_holiday` is weeks.
- `valuation` means they want a price estimate **and have not decided to sell**: they are still deciding, need the figure for paperwork or insurance, or are curious. `sell` means they have decided to sell or want to list, even if they also ask what it is worth.
- `invest` means yield, development, land banking or several units.
- `buy` includes a holiday or second home they want to own, even when they describe it with holiday words ("for our summers", "flying in for the week").
- `info` is a question about law, taxes, process or an area, with no transaction asked for.
- `spam` covers suppliers, advertising, scams, other agencies prospecting and anything that is not a client. **A business or person acting for a genuine client is a client, not spam.** That includes a relocation firm renting for staff, a lawyer or family office buying for a client, and a developer looking for land. Label their intent by what the client wants, with `on_behalf: true`.

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
