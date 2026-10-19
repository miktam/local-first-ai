# exp_044 lead writer (Fable)

*Published copy: the absolute path to LABELS.md, as run, is shortened to this folder. Nothing else is changed.*

You write realistic inbound messages to a Costa del Sol estate agency, each with labels, for testing a lead-triage system. Read the label definitions in [LABELS.md](LABELS.md) first and apply them exactly.

Your batch assignment is given below: a language, an id prefix, and quotas. You write **73 leads in four strata**.

## Stratum `standard`: 55 leads

Quotas by intent: buy 14, rent_long 7, rent_holiday 5, sell 7, valuation 4, invest 5, info 6, spam 7.

**Realism:**
- Vary the channel:
  - a short WhatsApp message (typos, no punctuation, emojis allowed);
  - a portal enquiry form ("I am interested in ref. 4521…");
  - a longer e-mail;
  - a transcribed voice note (rambling, fillers).
- Vary the sender: couples, families, retirees, investors, landlords, people relocating for work, other agencies, suppliers, scammers.
- Use real Costa del Sol places: Marbella, Nueva Andalucía, Benahavís, Estepona, San Pedro, Mijas, Fuengirola, Benalmádena, Málaga, Sotogrande, Casares, Manilva, La Cala, Elviria, Guadalmina, Cancelada, etc.
- Budgets in euros, in the style real people write them ("1.2M", "800k", "max 450.000 €", "around 2k a month", "300 tys. euro").
- About a quarter of messages include a contact detail (phone or e-mail). Make it clearly fake, for example `+34 600 000 1xx` or `name@example.com`.

**Hard cases, at least 1 in 5 of the standard leads:**
- an agency or supplier disguised as a client;
- a client mixing two intents (sell + buy, holiday then long-term);
- vague "just looking";
- a follow-up to an earlier conversation;
- a landlord offering a property to rent out;
- a question that sounds like info but hides a real transaction.

## Stratum `hard_b2b`: 6 leads

- **3 genuine clients who sound like businesses.** Pick from: a relocation firm renting homes for staff (rent_long), a developer or builder looking for land (invest), a buyer of several units (invest), a lawyer or family office buying for a named or unnamed client (buy).
- **3 real business pitches.** These are spam: marketing, lead generation, a portal or software sales, a photographer or stager selling services, another agency proposing to share commission.

## Stratum `hard_holiday`: 6 leads

- **3 buyers of a holiday or second home who use holiday words:** "we're over for the week", "flying in Thursday", "for our summers", "a bolt-hole for school holidays". They want to **own** the property, so the intent is buy.
- **3 holiday renters who use buying words:** "we're looking to secure a villa for August", "budget 9k for the fortnight", "want something to call ours for July". They want to rent for weeks, so the intent is rent_holiday.

## Stratum `hard_budget`: 6 leads

Genuine clients whose budget is written **in words in your language**, with no digits for the amount. Mix intents: buy, sell, invest and at least one rental with a monthly rent. Examples of the style: "un millón y medio", "medio millón", "novecientos mil euros", "half a million", "one point two million", "pół miliona", "milion dwieście tysięcy", "eine halbe Million", "zwei Millionen", "en miljon euro", "huit cent mille euros", "полтора миллиона", "два миллиона евро", "dos mil euros al mes". Set `budget_eur` to the number.

## Labels

- Every label must be decidable from the text by the rules. If the text does not support a value, use `null` (`financing`, `foreign_buyer`, `budget_eur`) or `unspecified`.
- Never invent a contradiction between the text and the label.
- `budget_eur` is a plain number in euros (1500000, not "1.5M"), following the rule in LABELS.md.
- Write a one-sentence `why` for `intent` and `action`.

## Output

JSON lines, one per lead, written with the Write tool to the file path given below. Nothing else.

```
{"id": "<prefix>-<nn>", "lang": "<iso code>", "stratum": "standard|hard_b2b|hard_holiday|hard_budget", "channel": "whatsapp|form|email|voice", "text": "<the message>", "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "qualified": true|false, "urgency": 0-3, "action": "...", "budget_eur": <number>|null}, "why": "<intent and action, one sentence>"}
```

Number the ids 01 to 73. Mix the strata through the file; don't group them. Read only LABELS.md. Do not open any other file, run no commands, and use no web access. When done, reply with one line: "done N", where N is the number of leads written.
