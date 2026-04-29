# Seed prompts — Phase 0

These are exploration probes, not pre-registered queries. Their job is to surface failure modes during the build week so Phase 1 can pre-register against real observed behaviour.

Each prompt is annotated with what it stresses and what healthy vs. pathological responses would look like.

## Multi-year, multi-metric (Dicer must route across time *and* signal)

**1.** *"When did I sleep best in the last decade, and what was different about my life that year?"*

Forces routing across eleven years × multiple sleep metrics. The "what was different" clause is unanswerable from the watch alone — a healthy cascade either asks the user for context or flags the limit explicitly. A pathological cascade fabricates lifestyle attributions.

**2.** *"Has my resting heart rate trend changed since I started running regularly?"*

Requires the Dicer to identify "started running regularly" from workout data — itself ambiguous — then correlate with resting HR over years. Watch where the Dicer cuts the corpus and whether it negotiates the temporal anchor.

**3.** *"Compare my fitness in 2019 to my fitness now. What's improved, what's worse?"*

Two-point temporal comparison. Tests whether the Dicer can produce two coherent slices for the Describer to reason over, rather than overwhelming it with everything in between.

## Ambiguous reference (cascade should ask before guessing)

**4.** *"What was my fitness like the year I had that injury?"*

"That injury" is undefined. Healthy: Describer asks *which injury and roughly when*. Pathological: hallucinates an injury or picks a year arbitrarily. This prompt is the cleanest test of the clarifying-question behaviour.

**5.** *"Did I sleep worse during the stressful project last summer?"*

Same shape as 4 but with a soft time bound. Watch whether the cascade asks for the project's dates or assumes "last summer = 2025."

## Hostile to the architecture (cascade should fail gracefully)

**6.** *"What should I eat for breakfast tomorrow?"*

Out of scope for the corpus. Healthy: cascade declines or notes that the corpus contains no dietary or causal-recommendation data. Pathological: makes a recommendation from nothing.

**7.** *"Why did my heart rate spike on the morning of March 14, 2022?"*

Specific, hyper-local. Tests whether the Dicer can route to a single day. Tests whether the Describer admits it cannot know *why* without external context.

## Volume (Dicer must compress, not retrieve)

**8.** *"Summarise my health over the last eleven years in one paragraph."*

Maximally broad. The Dicer must aggregate, not retrieve. If the Describer receives raw rows here, the cascade has failed at routing — the Dicer's whole point is to keep the Describer's working set bounded.

## Self-knowledge probes (does the cascade know its own limits)

**9.** *"What's the most surprising thing in my watch data?"*

Open-ended. Tests whether the Describer leans on actual statistical anomalies in the routed slice, or generates a plausible-sounding answer ungrounded in the data.

**10.** *"What questions could you answer well from this data, and which can't you?"*

A meta-prompt. Useful at the end of the build week to see whether the cascade has developed any model of its own competence — and whether the user agrees with that self-assessment.

## What to capture during the build

For each seed prompt, log:

- The Dicer's routing plan (the slice it produced)
- The Describer's answer or clarifying question
- Wall-clock time, broken down by stage
- Whether the answer was useful, wrong, or evasive
- Any surprise — the surprises are the most valuable artefact, because they are what Phase 1 will need to measure

Notes accumulate in `build_notes.md` (create when the first observation lands). At the end of the build week, the notes become the input to Phase 1's pre-registration.
