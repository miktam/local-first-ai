# exp_050 lead labels: definitions (shared by writers, checkers and the scorer)

*Version 4. exp_049's version 3, dated Friday 23 October 2026 instead of 21 October, plus **the sender's criteria** (areas, minimum bedrooms, required features), **the contacts** in the message, and **the fit rule**, which code applies to the labels and the agency's inventory. The urgency scale and the action rules are unchanged.*

**Every message arrives on Friday 23 October 2026.** Each lead is one inbound message to a Costa del Sol estate agency: WhatsApp, a portal enquiry form, an e-mail, or a transcribed voice note. Each label is decided **from the message text alone**. Never guess beyond the text.

| Field | Values | Rule |
|---|---|---|
| `intent` | `buy`, `rent_long`, `rent_holiday`, `sell`, `valuation`, `invest`, `info`, `spam` | The sender's main goal. See the intent notes below. |
| `property_type` | `apartment`, `penthouse`, `townhouse`, `villa`, `plot`, `commercial`, `unspecified` | The kind of property the message is about. `unspecified` if none is named. A "flat" or "piso" is an apartment; a "house" or "Haus" with no other detail is a villa. |
| `timeline` | `now`, `months`, `later`, `unspecified` | **Today is 23 October 2026.** `now`: within about a month (by about 23 November 2026), urgent, or already visiting or ready to sign. `months`: later in 2026, by 31 December. `later`: 2027 or beyond, **even if only weeks away** ("from January" is `later`). `unspecified`: no timing given. For a holiday rental, the timeline is when the stay starts. |
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

## Version 4: the sender's criteria, the contacts and the fit rule

The agency in exp_050 **sells homes and lets them long-term (6 months or more)** in eight areas. It does **no holiday lets**. Its inventory is `inventory.md` in this folder. Writers read it; checkers don't need it, because every label below is decided from the message alone.

**Criteria labels**, only when `intent` is `buy`, `invest`, `rent_long` or `rent_holiday`. For any other intent they are `[]`, `null` and `[]`.

| Field | Values | Rule |
|---|---|---|
| `areas` | a list of `san_pedro`, `nueva_andalucia`, `guadalmina`, `marbella_centre`, `puerto_banus`, `elviria`, `estepona`, `benahavis`, `other` | Every area the sender would accept. `[]` if none is named. "Near X" counts as X. **"Marbella" with no district** means the six Marbella areas: `san_pedro`, `nueva_andalucia`, `guadalmina`, `marbella_centre`, `puerto_banus`, `elviria`. The old town and the Golden Mile are `marbella_centre`. Any place outside the eight (Mijas, Fuengirola, Sotogrande, Málaga, Benalmádena…) adds `other`. |
| `min_bedrooms` | a number, or `null` | The fewest bedrooms the sender accepts: "3 bedrooms" → 3, "at least 2" → 2, "2 or 3 bedrooms" → 2. `null` if not stated. |
| `features` | a list of `sea_view`, `pool`, `garden`, `terrace`, `parking`, `walk_to_beach`, `near_schools`, `golf` | Features the sender **requires**: stated as a need or a want without hedging. "Ideally", "would be nice", "a bonus", "not essential" → not required, so not listed. A communal pool is `pool`; a balcony or terrace is `terrace`; a garage or parking space is `parking`; "walking distance to the beach" is `walk_to_beach`; schools for the children nearby is `near_schools`; on or by a golf course is `golf`. |

**`contacts`:** a list of every phone number and e-mail address exactly as written in the message, digits or words. `[]` if none.

**The fit rule.** Code applies it to the labels; nobody labels a home by hand. A listing **fits** an enquiry when all of these hold:
1. **Operation:** `buy` or `invest` → a listing for sale; `rent_long` → a long let. No other intent fits any listing, and `rent_holiday` fits none, since the agency does no holiday lets.
2. **Type**, if stated: `apartment` accepts an apartment or a penthouse; `penthouse`, `townhouse`, `villa`, `plot` and `commercial` accept only their own type. `unspecified` accepts an apartment, penthouse, townhouse or villa, but not a plot or commercial premises.
3. **Area**, if any is named: the listing's area is in `areas`. If `areas` is only `other`, nothing fits.
4. **Budget**, if stated: the price is at most `budget_eur` × 1.10. That is the sale price for a sale, or the monthly rent for a long let.
5. **Bedrooms**, if stated: at least `min_bedrooms`. Plots and commercial premises have none.
6. **Features:** every feature in `features` is among the listing's.

The **fitting set** is every listing that fits. A reply may offer only listings from it.
