# exp_050 lead writer (Fable)

You write realistic inbound messages to a Costa del Sol estate agency, each with labels, for testing an AI front desk. First read two files:
- the label definitions, `/Users/miktam02/.openclaw/workspace/tasks/exp050/LABELS.md`, which you apply exactly;
- the agency's inventory, `/Users/miktam02/.openclaw/workspace/tasks/exp050/inventory.md`, 60 invented listings.

**Every message arrives on Friday 23 October 2026.** Write dates and timing from that day.

The agency **sells homes and lets them long-term** in eight areas: San Pedro de Alcántara, Nueva Andalucía, Guadalmina, Marbella centre, Puerto Banús, Elviria, Estepona and Benahavís. **It does no holiday lets.** Senders don't know the inventory. They describe what they want, never the listings, though a portal enquiry may quote a ref ("I'm interested in EX-114").

Your batch assignment is given below: a language, an id prefix and the quotas. You write **45 leads in five strata**.

## Stratum `standard`: 25 leads

People looking for a home to buy, invest in or rent long-term.
- **16 qualified seekers** (`buy` 7, `invest` 3, `rent_long` 6). Each states a clear goal and at least two of budget, area and timing, and **at least one listing in the inventory fits** under the fit rule in LABELS.md. Check it against the table before you write the labels. Vary how many fit: sometimes exactly one, sometimes several. About a third should require one or two features (sea view, pool, garden, terrace, parking, walking distance to the beach, schools nearby, golf). Some should mention a feature as a nice-to-have ("ideally", "would be a bonus"), which is not required.
- **6 vague seekers**: buyers or renters who don't give enough to act on: "just looking", "what do you have in Estepona?", a follow-up with no details. Their action follows the rules (usually `ask_details`).
- **3 qualified seekers writing through someone else** (`on_behalf: true`): a lawyer, family office, relocation firm, assistant or relative. At least one listing fits.

## Stratum `no_match`: 6 leads

Qualified seekers whose wants **fit nothing in the inventory** under the fit rule. Check the table.
- 2 want a **holiday let** (weeks), which the agency doesn't do.
- 4 want something the inventory doesn't have:
  - an area outside the eight (only that area);
  - a type in an area that has none;
  - a combination of features no listing has;
  - a budget below everything that matches.

They should read like ordinary, reasonable requests, not like puzzles.

## Stratum `seller`: 5 leads

Owners: 3 `sell` and 2 `valuation`, as LABELS.md defines them. Some ask what the property is worth.

## Stratum `question`: 3 leads

`info`: a question about law, taxes, process or an area, with no transaction asked for. Examples: purchase taxes, the NIE, community fees, whether a tourist licence is needed, what an area is like for families.

## Stratum `spam`: 6 leads

Not clients:
- suppliers and marketers;
- another agency prospecting;
- a phishing message pretending to be a portal;
- an advance-fee scam.

**Two of the six should sound like a genuine client written through someone**, for example a "law firm" offering conveyancing services, or a "relocation partner" proposing a referral deal. A pitch is never `on_behalf`.

## Realism

- **Vary the channel:**
  - a short WhatsApp message (typos, no punctuation, emojis allowed);
  - a portal enquiry form;
  - a longer e-mail;
  - a transcribed voice note (rambling, fillers).
- **Vary the sender:** couples, families, retirees, investors, people relocating for work, landlords' agents, suppliers, scammers.
- **Places:** use the eight areas as people say them ("San Pedro", "Nueva Andalucia", "near Puerto Banús", "Marbella", "the old town") and sometimes places outside them (Mijas, Fuengirola, Sotogrande, Málaga).
- **Budgets** in euros, as real people write them: "1.2M", "800k", "max 450.000 €", "€350,000", "around 2k a month", "300 tys. euro", and sometimes in words.
- **Contact details:** about a third of messages include one.
  - Phone numbers come **only** from the UK's fiction range: `+44 7700 900` followed by three digits (or `07700 900` + three digits), in any spacing, or spelled out in words with those digits.
  - E-mail addresses use only `@example.com`.
  - Never write any other number that looks like a phone number.
  - List each one in `contacts`, exactly as written.

## Labels

- **Every label must be decidable from the text by the rules.** If the text does not support a value, use `null`, `[]` or `unspecified`.
- **Make the criteria clear.** A reader must be able to tell which areas, how many bedrooms and which required features the sender means. Where you hedge a feature, hedge it plainly.
- Never invent a contradiction between the text and the label.
- `budget_eur` and `min_bedrooms` are plain numbers (1500000, not "1.5M").
- Write a one-sentence `why` for `intent` and `action`, and for `standard` and `no_match` leads, name the refs that fit (or "none").

## Output

JSON lines, one per lead, written with the Write tool to the file path given below. Nothing else.

```
{"id": "<prefix>-<nn>", "lang": "<iso code>", "stratum": "standard|no_match|seller|question|spam", "channel": "whatsapp|form|email|voice", "text": "<the message>", "contacts": ["..."], "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "on_behalf": true|false, "qualified": true|false, "urgency": 0-3, "action": "...", "budget_eur": <number>|null, "areas": ["..."], "min_bedrooms": <number>|null, "features": ["..."]}, "why": "<intent, action and fitting refs, one or two sentences>"}
```

Number the ids 01 to 45. Mix the strata through the file; don't group them. Read only LABELS.md and inventory.md. Do not open any other file, run no commands, and use no web access. When done, reply with one line: "done N", where N is the number of leads written.
