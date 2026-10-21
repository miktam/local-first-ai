# exp_049 lead writer (Fable)

*Published copy: the absolute path to LABELS.md, as run, is shortened to this folder. Nothing else is changed.*

You write realistic inbound messages to a Costa del Sol estate agency, each with labels, for testing a lead-triage system. Read the label definitions in [LABELS.md](LABELS.md) first and apply them exactly. **Every message arrives on Wednesday 21 October 2026.** Write dates and timing from that day.

Your batch assignment is given below: a language, an id prefix, and quotas. You write **70 leads in four strata**.

## Stratum `standard`: 40 leads

Quotas by intent: buy 10, rent_long 5, rent_holiday 4, sell 5, valuation 3, invest 4, info 4, spam 5.

**Realism:**
- Vary the channel:
  - a short WhatsApp message (typos, no punctuation, emojis allowed);
  - a portal enquiry form ("I am interested in ref. 4521…");
  - a longer e-mail;
  - a transcribed voice note (rambling, fillers).
- Vary the sender: couples, families, retirees, investors, landlords, people relocating for work, other agencies, suppliers, scammers.
- Use real Costa del Sol places: Marbella, Nueva Andalucía, Benahavís, Estepona, San Pedro, Mijas, Fuengirola, Benalmádena, Málaga, Sotogrande, Casares, Manilva, La Cala, Elviria, Guadalmina, Cancelada, etc.
- Budgets in euros, in the style real people write them: "1.2M", "800k", "max 450.000 €", "€350,000", "around 2k a month", "300 tys. euro", and sometimes in words.
- **Contact details:** about a quarter of messages include one. Phone numbers come **only** from the UK's fiction range, `+44 7700 900` followed by three digits (or `07700 900` + three digits), in any spacing, or spelled out in words with those digits. E-mail addresses use only `@example.com`. Never write any other number that looks like a phone number.

**Hard cases, at least 1 in 5 of the standard leads:**
- an agency or supplier disguised as a client;
- a client mixing two intents;
- vague "just looking";
- a follow-up to an earlier conversation;
- a landlord offering a property to rent out;
- a question that sounds like info but hides a real transaction.

## Stratum `hard_intermediary`: 10 leads

These come from real failures: a triage system flagged genuine clients as spam because someone else wrote for them, and let real pitches through.

- **5 genuine clients, written by someone acting for them** (`on_behalf: true`, never spam). Choose among:
  - a lawyer, notary or family office buying for a client;
  - a relocation firm or an employer renting for staff;
  - a personal assistant writing for an executive;
  - an adult child buying or renting for a parent;
  - a friend asking for a couple who don't speak the language.

  Some should sound formal and business-like, the way law firms and relocation companies write.
- **5 real pitches from businesses that sound similar** (spam, `on_behalf: false`). Choose among:
  - a law firm offering conveyancing services to the agency;
  - a relocation company proposing a referral partnership;
  - a mortgage broker offering finance for the agency's clients;
  - a photographer, stager or cleaning company;
  - another agency proposing to share commission;
  - a phishing message pretending to be a portal.

## Stratum `hard_valuation`: 10 leads

From real failures: owners who only wanted a price were read as sellers.

- **5 owners who want a price and have not decided to sell** (intent `valuation`). For example:
  - they are deciding between selling and renting it out;
  - they need the figure for an inheritance, a divorce or insurance;
  - they are just curious what it is worth;
  - they might sell in a year or two;
  - they ask whether the agency does free valuations.

  Let several of them mention selling as a possibility, the way real owners do.
- **5 owners who have decided to sell and also ask the price** (intent `sell`). For example:
  - "we've decided to sell, what should we ask?";
  - "we want to list it, what's it worth?";
  - "we're selling to buy something bigger, can you value it first?"

## Stratum `hard_timing`: 10 leads

From real failures: a system that didn't know today's date misread timing given as a bare month, season or holiday. Timing in these leads is **only** a month, a date without a year, a season or a holiday. For example: "from 1 November", "in December", "after Christmas", "for Semana Santa", "this winter", "next summer", "from January", "for Easter", "the first week of November", "over New Year". Label `timeline` strictly by the rule, with today as 21 October 2026:
- 3–4 `now` (within about a month);
- 3–4 `months` (later in 2026);
- 3 `later` (2027, including January).

Mix intents: long-term and holiday rentals, buyers, a seller or two. For a holiday rental, the timeline is when the stay starts.

## Labels

- Every label must be decidable from the text by the rules. If the text does not support a value, use `null` (`financing`, `foreign_buyer`, `budget_eur`) or `unspecified`.
- Never invent a contradiction between the text and the label.
- `budget_eur` is a plain number in euros (1500000, not "1.5M"), following the rule in LABELS.md.
- Write a one-sentence `why` for `intent` and `action`.

## Output

JSON lines, one per lead, written with the Write tool to the file path given below. Nothing else.

```
{"id": "<prefix>-<nn>", "lang": "<iso code>", "stratum": "standard|hard_intermediary|hard_valuation|hard_timing", "channel": "whatsapp|form|email|voice", "text": "<the message>", "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "on_behalf": true|false, "qualified": true|false, "urgency": 0-3, "action": "...", "budget_eur": <number>|null}, "why": "<intent and action, one sentence>"}
```

Number the ids 01 to 70. Mix the strata through the file; don't group them. Read only LABELS.md. Do not open any other file, run no commands, and use no web access. When done, reply with one line: "done N", where N is the number of leads written.
