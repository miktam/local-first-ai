# Clef-Flash 9B — model notes

*An open decision model from Cloudflare (`Cloudflare_clef-flash-Q8_0.gguf`, about 10 GB at Q8_0), the smaller sibling of Clef 27B ([clef-27b.md](./clef-27b.md)). It picks among given options and never writes text. We run it on the Mac mini through llama.cpp b11711's `/v1/systemone`, as `ai.casasol.triage-flash9b`. Since 2026-10-21 it answers the facts half of CasaSol's lead triage. One fact per line, dated and sourced; newest at the bottom of each section.*

---

## What it is good at

- 2026-10-21 · Behind the same rules as Kev-4B, on 566 synthetic enquiries, it read "qualified" right 0.901 of the time (Kev 0.821), timeline 0.773 (0.712), property type 0.963 (0.887) and the request 0.965 (0.919). That is 50–70 % of Clef 27B's gain at about a third of the size. [exp_048](../../tasks/chronos/exp_048_flash_hybrid/RESULT.md)
- 2026-10-21 · Narrow yes/no questions: "writing on behalf of someone else?" 0.991 right on 350 fresh enquiries (Kev 0.957 on dev), and "only wants a price?" 24/24 on a valuation stratum (Kev 18/24). [exp_049](../../tasks/chronos/exp_049_kev_rev5/RESULT.md)
- 2026-10-21 · Told today's date in the question, it read timing given as a bare month ("from 1 November", "for Easter") right 0.870 of the time, against 0.444 without the date. [exp_049](../../tasks/chronos/exp_049_kev_rev5/RESULT.md)

## Speed and cost

- 2026-10-21 · Median 3.8 s per enquiry for 12 triage questions in one call, about 10 GB loaded. Run alongside Kev-4B on the same GPU, the two calls take about 5.5 s. [exp_048](../../tasks/chronos/exp_048_flash_hybrid/RESULT.md)

## Where it is weak

- 2026-10-21 · Alone, it read 9 of 485 genuine enquiries as spam, nearly all clients writing through a lawyer, relocation firm or developer. Its business-pitch scores also run higher than Kev's, so the pitch cut tuned for Kev fires more often. [exp_048](../../tasks/chronos/exp_048_flash_hybrid/RESULT.md)
- 2026-10-21 · It answered "on behalf of a client?" with yes (0.86) for an advance-fee scam that claimed a client. [exp_049](../../tasks/chronos/exp_049_kev_rev5/RESULT.md)
