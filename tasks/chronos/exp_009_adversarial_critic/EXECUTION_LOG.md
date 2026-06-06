# Experiment 009 — Execution Log

*Append-only. Each run is a dated entry. Never edit past entries.*

---

## Run 001 — 2026-06-06 (first execution)

**Target project:** CasaSol (`~/REPOS/casasol`)

### Pre-run checklist

- [ ] Ollama running: `ollama ps` — confirm gemma4:26b loaded
- [ ] In workspace root: `cd ~/.openclaw/workspace`
- [ ] Results dir exists: `tasks/chronos/exp_009_adversarial_critic/results/`

---

### Step 1 — Option B (gemma4:26b local)

**Command:**
```
python3 tasks/chronos/exp_009_adversarial_critic/critic.py --project ~/REPOS/casasol
```

**Results:**

| Field | Value |
|-------|-------|
| Duration | 84.7 s |
| Input tokens | 8,966 |
| Output tokens | 902 |
| Tokens/sec | 28.7 |
| Result file | `results/gemma4_26b_20260606T131205.json` |

**Raw terminal output:**
```
=== Adversarial Critic (Option B — gemma4:26b) ===
  Project: /Users/miktam02/REPOS/casasol

  Collecting context...
  Context size: 32621 chars

  Calling gemma4:26b...
  Done in 84.7s
  Tokens — input: 8966  output: 902  total: 9868  speed: 28.7 tok/s

  Saved → gemma4_26b_20260606T131205.json

  [INVESTOR]
    ⚠  No clear unit economics for the 'witnessing' pipeline vs. the value of the extracted data.
    ⚠  Lack of evidence regarding the cost of hardware deployment (Mac Mini) across a distributed agency network.
    ⚠  No quantification of the 'demand signal retention' value to the agency's bottom line.
  [DPO]
    ⚠  No formal Data Protection Impact Assessment (DPIA) for the VLM-based extraction process.
    ⚠  Missing procedures for handling Data Subject Access Requests (DSAR) for data held in the 'witnessed corpus'.
    ⚠  No documented 'Data Processing Agreement' (DPA) template for the agency-to-CasaSol relationship.
  [ENGINEER]
    ⚠  No mention of how the 'Redactor' handles unstructured data within the 'witnessed' images.
    ⚠  Lack of a formal schema registry or versioning for the 15-field Reducer output.
    ⚠  No strategy for handling model drift or performance degradation in the 'Reducer' over time.

  Top actions:
    →  Implement automated PII leakage testing (exp_007) to validate the Redactor's efficacy.
    →  Define the unit economics of the 'witnessing' labor vs. the proprietary data value.
    →  Secure a formal DPIA for the in-person image capture and VLM extraction workflow.
```

---

### Step 2 — Option A (Claude via /critic)

**Command:** ran inline in this session (context already loaded)

**Results:**

| Field | Value |
|-------|-------|
| Duration | n/a (inline) |
| Input tokens (estimated) | ~4,750 (~19,000 context chars ÷ 4) |
| Output tokens (estimated) | ~1,170 (~900 words × 1.3) |
| Result file | `results/claude_20260606_134500.json` |

**Top findings:**
- HIGH: VLM witnessing pipeline not in any commit — the moat's second pillar is documentation only
- HIGH: No DPA template — hard gate on any pilot SOW
- HIGH: No revenue model / pricing anywhere
- Severity: High 3 · Medium 9 · Low 3

---

### Step 3 — Comparison

**Command:**
```
python3 tasks/chronos/exp_009_adversarial_critic/compare.py
```

**Token comparison:**

| Metric | Claude (est) | gemma4:26b (exact) |
|--------|-------------|-------------------|
| Input tokens | ~4,750 | 8,966 |
| Output tokens | ~1,170 | 902 |
| Total tokens | ~5,920 | 9,868 |
| Speed | n/a | 28.7 tok/s |
| Duration | n/a (inline) | 84.7 s |

*Note: gemma4 context was 32,621 chars (8,966 tokens); Claude context was ~19,000 chars (~4,750 est). The gap is mostly system-prompt tokenization overhead in Ollama.*

---

### Step 4 — Human scoring (Andrei fills in)

**Total high-severity issues flagged by Claude:** 3

**For each Claude high-severity issue — did gemma4 also flag it?**

| # | Claude issue (short) | gemma4 match? (Y/N) | Notes |
|---|----------------------|---------------------|-------|
| H1 | VLM witnessing pipeline not in any commit | N | gemma4 flagged replicability ("just a VLM API"), never noticed code doesn't exist |
| H2 | No DPA template | Y | Exact match: "No documented DPA template for agency-CasaSol relationship" |
| H3 | No revenue model / pricing | Partial | gemma4 said "no unit economics for witnessing" — same gap, narrower economic frame |

**gemma4 false positives (flagged by gemma4, not Claude):**

| # | gemma4 issue | Verdict (FP/Valid) |
|---|-------------|-------------------|
| 1 | Filesystem firewall = single point of failure / no crypto isolation | Valid (real gap, Claude missed) |
| 2 | Manual witnessing labor hard to scale (headcount) | Valid (real gap, Claude missed) |
| 3 | Redactor PII audit rate not measured | Valid (relates to exp_007, Claude missed) |
| 4 | Gemma 4 model family dependency risk | Valid (Claude flagged model pinning, different angle) |
| 5 | Hardware deployment cost across agency network | Valid (real gap) |
| 6 | Demand signal quantification missing | Overlap with H3 partial |
| 7 | Redactor handling unstructured image data | Valid |
| 8 | Bystander Art. 6/9 at capture | Overlap with Claude's blur/Art.9 concern |

**Scores:**

| Metric | Result | Threshold | Pass? |
|--------|--------|-----------|-------|
| Issue overlap rate | ~50% (1 certain + 1 partial / 3) | ≥60% | FAIL |
| False positive rate | ~50% (~9 of 18 gemma4 issues not in Claude) | ≤30% | FAIL |

**Overall verdict:** FAIL — gemma4 matched the DPO lens well but missed the most critical engineering finding (VLM not implemented) and generated too many investor/engineer issues outside Claude's frame.

**Nuance:** gemma4's "false positives" are mostly *valid issues Claude missed*, not noise. The high FP rate means gemma4 is exploring different territory, not malfunctioning. For use as a primary QA gate (replacing Claude), FAIL. For use as a complementary signal alongside Claude, useful.

---

### Step 5 — Observations

**Key finding:** gemma4 and Claude converged strongly on the DPO/compliance layer (DPA, DSAR, DPIA — 3/3 matches) but diverged sharply on the engineering layer. The decisive failure: gemma4 never noticed that the VLM witnessing pipeline — the primary moat component — doesn't exist in any commit. It reasoned about *replicability* of witnessing, not about *existence*. Claude caught it by cross-referencing BUILD_LOG claims ("witnessing reframe is core MVP") against the actual commit history (no VLM code). That is a deeper inferential move: not pattern-matching on what's written, but detecting the gap between documented intent and implemented reality.

**False positives are not noise:** gemma4's issues outside Claude's frame (filesystem crypto isolation, labor scalability, Redactor audit rate, hardware cost) are largely valid. Claude simply didn't raise them. This is two critics exploring different search spaces, not one critic being wrong.

**localfirstai.eu post angle:** "We tried to use a local 26B model as our digital QA. Here's exactly where it fell short — and what the result tells you about what 'reasoning' actually means for local models."

The hook: compliance gaps → both critics agree. Implementation-vs-documentation gaps → only the frontier model noticed. The question the post asks: is that gap worth the API cost?

---

*Next run: append a new "Run 002" section below.*
