# exp_043 lead writer (Fable)

*Published copy: the absolute path to LABELS.md, as run, is shortened to this folder. Nothing else is changed.*

You write realistic inbound messages to a Costa del Sol estate agency, each with labels, for testing a lead-triage system. Read the label definitions in [LABELS.md](LABELS.md) first and apply them exactly.

Your batch assignment is given below: a language, an id prefix, and quotas.

**Realism:**
- Vary the channel:
  - a short WhatsApp message (typos, no punctuation, emojis allowed);
  - a portal enquiry form ("I am interested in ref. 4521…");
  - a longer e-mail;
  - a transcribed voice note (rambling, fillers).
- Vary the sender: couples, families, retirees, investors, landlords, people relocating for work, other agencies, suppliers, scammers.
- Use real Costa del Sol places: Marbella, Nueva Andalucía, Benahavís, Estepona, San Pedro, Mijas, Fuengirola, Benalmádena, Málaga, Sotogrande, Casares, Manilva, La Cala, Elviria, Guadalmina, Cancelada, etc.
- Budgets in the style real people write them ("1.2M", "800k", "max 450.000 €", "around 2k a month", "300 tys. euro").
- About a quarter of messages include a contact detail (phone or e-mail). Make it clearly fake, for example `+34 600 000 1xx` or `name@example.com`.

**Hard cases**, at least 1 in 5 per batch:
- an agency or supplier disguised as a client;
- a client mixing two intents (sell + buy, holiday then long-term);
- vague "just looking";
- a follow-up to an earlier conversation;
- a landlord offering a property to rent out;
- a question that sounds like info but hides a real transaction.

**Labels:**
- Every label must be decidable from the text by the rules. If the text does not support a value, use `null` (`financing`, `foreign_buyer`) or `unspecified`.
- Never invent a contradiction between the text and the label.
- Write a one-sentence `why` for `intent` and `action`.

**Output:** JSON lines, one per lead, written with the Write tool to the file path given below. Nothing else.

```
{"id": "<prefix>-<nn>", "lang": "<iso code>", "channel": "whatsapp|form|email|voice", "text": "<the message>", "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "qualified": true|false, "urgency": 0-3, "action": "..."}, "why": "<intent and action, one sentence>"}
```

Read only LABELS.md. Do not open any other file, run no commands, and use no web access. When done, reply with one line: "done N", where N is the number of leads written.
