# Clef 27B — model notes

*An open decision model from Cloudflare (`Cloudflare_clef-Q8_0.gguf`, about 29 GB at Q8_0). Like Kev-4B ([kev-4b.md](./kev-4b.md)) it picks among given options and returns a probability for each, and it never writes text. We run it on the Mac mini through llama.cpp b11711's `/v1/systemone`. One fact per line, dated and sourced; newest at the bottom of each section.*

---

## What it is good at

- 2026-10-18 · As an answerability gate in front of an untrained 4B voice, it raised correct refusals from 0.474 to 0.921, with 0 false abstentions in 101. [exp_042](../../tasks/chronos/exp_042_answerability_gate_retrieval/RESULT.md)
- 2026-10-21 · Behind the same triage rules as Kev-4B, on 566 synthetic enquiries in seven languages, it read the request right 0.991 of the time (Kev 0.919), "qualified" 0.936 (0.821) and timeline 0.833 (0.712). No genuine enquiry was flagged as spam, and it caught 81 of 81 spam. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)
- 2026-10-21 · Its yes/no answers are decisive: 58 of 566 "qualified" scores fell between 0.3 and 0.7, against 191 of Kev's. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)

## Speed and cost

- 2026-10-18 · About 12 s per gate question on the mini, against about 2 s for Kev-4B. [exp_042](../../tasks/chronos/exp_042_answerability_gate_retrieval/RESULT.md)
- 2026-10-21 · One call answering 12 triage questions: median 13.5 s, against 1.7 s for Kev-4B. It needs about 29 GB loaded. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)
- 2026-10-21 · It ran out of GPU memory once, when an 18 GB Ollama model loaded mid-run. Afterwards both stayed loaded together without errors. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)

## Where it is weak

- 2026-10-21 · It reads questions literally. Asked "will the sender need a mortgage?", it answers no when the mortgage is already approved: 10 of 130 stated cases, where Kev got all 130 right. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)
