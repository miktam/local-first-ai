# exp_043 lead checker (Fable, blind)

*Published copy: the absolute path to LABELS.md, as run, is shortened to this folder. Nothing else is changed.*

You label inbound messages to a Costa del Sol estate agency. Another writer has labelled them too; you do not see those labels. Read the label definitions in [LABELS.md](LABELS.md) first and apply them exactly. Decide **from the message text alone**.

Your batch file (JSON lines) gives `id` and `text` per lead. Read only LABELS.md and your batch file. Do not open any other file, run no commands, and use no web access.

For each lead, output one JSON line with every label, and `doubt: true` if a careful reader could reasonably pick a different `intent` or `action`:

```
{"id": "<copied exactly>", "labels": {"intent": "...", "property_type": "...", "timeline": "...", "financing": true|false|null, "foreign_buyer": true|false|null, "qualified": true|false, "urgency": 0-3, "action": "..."}, "doubt": true|false}
```

Write all lines with the Write tool to the output path given below, then reply with one line: "done N".
