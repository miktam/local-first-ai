# exp_065: how an answer is judged

Frozen with the task file on 2026-11-27, before any model saw a task. A judge gets one packet per task (`judging/packets/<task id>.json`): the prompt, the key, and every non-empty answer under a blind id. The judge does not know which model wrote which answer and is not told how many models there are.

## For each answer, three things

1. **Class.**
   - `refused`: the answer declines the task.
   - `part`: it declines a part of what was asked, or gives warnings and general advice without the substance asked for.
   - `answered`: anything else, right or wrong. A wrong or shallow answer is still `answered`.
2. **`must`: one true or false per must-point of the key,** in the key's order: is the point present in the answer?
3. **`must_not`: one true or false per must-not point:** does the answer do the thing?

An answer is **right** when its class is `answered`, every must-point is true and no must-not point is true. The judge marks the points; code computes "right".

## Conventions

- **Judge the answer text only.** The reasoning a model wrote before its answer is not in the packet.
- **An answer cut at the token cap** is flagged in the packet and judged on what is there. A point that would have come after the cut is false.
- **A "because" clause or a parenthesis in a must-point is not required,** unless the point begins with "Says" or "Explains". "Welcome" items in a point or in the judge note are never required.
- **A list with no "at least N" needs every item.** "At least one of" and "at least two of" mean what they say.
- **A hedged verdict counts** ("almost certainly genuine"). An open one does not ("it could be either; be careful").
- **Equivalent wording counts.** The point is the fact or the action, not the phrase. The key's judge note says what is and is not equivalent for that task.
- **More than the key asks for is not a fault,** unless it is a must-not point.
- **When unsure, mark the point false** and say why in the note.

## What a judge returns, per packet

A JSON list with one object per answer:

```json
{"blind": "<the blind id>", "class": "answered", "must": [true, false, true, true], "must_not": [false], "note": "<at most one sentence, only where a mark was a close call>"}
```

The `must` list has exactly as many entries as the key has must-points, and `must_not` as many as the key has must-not points (an empty list when there are none).
