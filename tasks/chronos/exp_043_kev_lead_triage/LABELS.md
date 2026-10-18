# exp_043 lead labels: definitions (shared by writers, checkers and the scorer)

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
| `action` | `call_now`, `send_listings`, `ask_details`, `book_valuation`, `answer_question`, `discard` | The agency's next step. Apply the action rules below **in order**; the first that fits decides. |

**Intent notes:**
- `rent_long` is 6 months or more; `rent_holiday` is weeks.
- `valuation` means they want a price estimate. Use `sell` when they want to list or sell; if they say both, `sell`.
- `invest` means yield, development, land banking or several units.
- `info` is a question about law, taxes, process or an area, with no transaction asked for.
- `spam` covers suppliers, advertising, scams, other agencies prospecting and anything that is not a client.

**Urgency scale:**
- `0`: no rush. Info, spam, or a timeline of later or unspecified with no hot signal.
- `1`: within a day. A real client with a months timeline.
- `2`: within hours. Qualified with timeline `now`, or a specific property asked about.
- `3`: right now. Visiting today or tomorrow, an offer or a signature imminent, or another agency competing.

**Action rules:**
1. `spam` → `discard`.
2. `info` → `answer_question`.
3. `valuation`, or `sell` when they ask what the property is worth → `book_valuation`.
4. Qualified, with timeline `now` and a phone number or a request to call → `call_now`.
5. Qualified → `send_listings`. For `sell` without a valuation request → `book_valuation`.
6. Otherwise → `ask_details`.
