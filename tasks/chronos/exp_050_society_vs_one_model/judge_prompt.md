# exp_050 reply judge (Opus, blind)

You review replies that a Costa del Sol estate agency's front desk would send to inbound enquiries. Each line of your batch file has a neutral `jid`, the `enquiry` and the `reply`. You don't know how a reply was produced, and you don't have the agency's inventory or rules, so judge only what a careful agent could judge from the enquiry and the reply. The agency sells homes and lets them long-term (6 months or more). It does no holiday lets. Today is 23 October 2026.

For each line, answer:
- `answers_request`: does the reply respond to what the sender actually asked for? (true/false)
- `asks_again`: does it ask for something the sender already gave? (true/false)
- `problem`: the most serious problem, or `none`:
  - `wrong_or_odd_fact`: a statement that contradicts the enquiry or looks wrong;
  - `misleading`: a promise or claim the agency may not be able to keep;
  - `language`: not in the language the enquiry is written in;
  - `tone`: rude, robotic or off;
  - `garbled`: broken text or characters;
  - `other`.
- `send_as_is`: would a careful agent send this reply unchanged? (true/false)
- `note`: one short sentence if `problem` is not `none` or `send_as_is` is false; otherwise "".

Read only this prompt and your batch file. Do not open any other file, run no commands, and use no web access. Write one JSON line per item, in the same order, with the Write tool, to the output path given below:

```
{"jid": "<copied exactly>", "answers_request": true|false, "asks_again": true|false, "problem": "none|wrong_or_odd_fact|misleading|language|tone|garbled|other", "send_as_is": true|false, "note": "..."}
```

Then reply with one line: "done N".
