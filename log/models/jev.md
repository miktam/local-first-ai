# Jev — model notes

*TypeSafe's decision model, released September 2026: it chooses among given options, scores or routes, and doesn't write text. It is a cloud API, so **we don't run it**: inference here stays on our own hardware. We learn the category through its open followers, Kev-4B among them ([kev-4b.md](./kev-4b.md)), and through what others publish. Every line here is a published claim, attributed; none is our measurement.*

---

- 2026-09 · Trained with a method TypeSafe calls RLCD ("calibrated decisions"). It claims to be 20–200× faster and 40–400× cheaper than small frontier LLMs, with output tokens free. *(TypeSafe, via Latent Space AINews, Sep 15–18 2026)*
- 2026-09 · Within two days, six open "clones" appeared, among them Kev-0.5B (a LoRA adapter and a small readout head on Qwen2.5-0.5B, running on a MacBook Pro). *(Latent Space AINews, Sep 17–18 2026)*
- 2026-09 · Uses named by early adopters: routing, approval gates, trace scoring, tool selection, citation selection and escalation. *(Latent Space AINews, Sep 15–18 2026)*
- 2026-10 · Artificial Analysis's Decision Index: decision models are competitive on classification and routing, but weaker and often overconfident on knowledge and reasoning. Jev 1.13 leads it, at $42 per million tasks and 0.13 s. *(Latent Space AINews, Oct 13–14 2026)*

## What our own Kev measurements say about the category

- 2026-10-19 · "Calibrated" is a claim to check, not a property to assume: Kev-4B's raw confidence was ordered but not calibrated (ECE 0.262), which matches the Decision Index's "often overconfident". [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
