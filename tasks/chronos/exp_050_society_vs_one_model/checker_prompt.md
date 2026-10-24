# exp_050 lead checker (Opus, blind)

You label inbound messages to a Costa del Sol estate agency. Another writer has labelled them too; you do not see those labels. Read the label definitions in `/Users/miktam02/.openclaw/workspace/tasks/exp050/LABELS.md` first and apply them exactly, including version 4's criteria (`areas`, `min_bedrooms`, `features`). **Every message arrives on Friday 23 October 2026.** Decide **from the message text alone**.

Your batch file (JSON lines) gives `id` and `text` per lead. Read only LABELS.md and your batch file. Do not open any other file, run no commands, and use no web access.

For each lead, output one JSON line with every label. Set `doubt: true` if a careful reader could reasonably pick a different `intent`, `action` or criterion (type, areas, bedrooms, budget or required features):

```
{"id": "<copied exactly>", "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "on_behalf": true|false, "qualified": true|false, "urgency": 0-3, "action": "...", "budget_eur": <number>|null, "areas": ["..."], "min_bedrooms": <number>|null, "features": ["..."]}, "doubt": true|false}
```

`budget_eur` and `min_bedrooms` are plain numbers (1500000, not "1.5M"), following LABELS.md, or `null`. The criteria are `[]`, `null` and `[]` for any intent other than buy, invest, rent_long or rent_holiday.

Write all lines with the Write tool to the output path given below, then reply with one line: "done N".
