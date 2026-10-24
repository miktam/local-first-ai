"""Figures for 'We Blamed Our Rules. The 4B Model Was the Ceiling.' (exp_047, exp_048, exp_049)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-21-the-model-was-the-ceiling.md"


def figures(repo: Path) -> dict:
    s = json.load(open(repo / "tasks/chronos/exp_048_flash_hybrid/results/test/score.json"))
    f, sp = s["fields"], s["spam"]
    arms = ("kev", "flash", "hybrid", "clef27b")
    series = [("Kev-4B", "--s1", "dot"), ("Clef-Flash 9B", "--s2", "dot"), ("Kev + Flash, split", "--ink", "ring"), ("Clef 27B", "--s3", "dot")]
    names = [("intent", "request"), ("action", "next step"), ("qualified", "qualified"), ("timeline", "timeline"), ("financing", "financing, where stated")]
    rows = [{"label": f"{lab} ({f[k]['n']})", "values": [f[k][a] for a in arms]} for k, lab in names]
    fields = V.dot_rows(
        rows, series,
        "Share read right on the same 566 synthetic enquiries (exp_043's and exp_044's test sets), one pass per reader, behind "
        "the same rules and cuts; the cuts were tuned for Kev. The split was recombined offline from recorded answers. Paired, "
        f"Clef 27B against Kev: request {f['intent']['n']} enquiries, 42 only Clef right against 1 only Kev (exp_047). "
        "Labels from two Fable readers who agreed.",
        lo=0.70, hi=1.0, dec=3, label="Four readers behind the same rules, field by field")
    spam = V.bar_pairs(
        [{"title": "Genuine enquiries flagged as spam", "note": f"of {sp['kev']['genuine_n']}, fewer is better",
          "values": [sp[a]["false_spam"] for a in arms], "top": 10, "dec": 0},
         {"title": "Spam caught", "note": f"of {sp['kev']['spam_n']}, more is better",
          "values": [sp[a]["caught"] for a in arms], "top": 81, "dec": 0},
         {"title": "Seconds per enquiry (median)", "note": "the split's time was not measured in the test pass",
          "values": [s["median_ms"]["kev"] / 1000, s["median_ms"]["flash"] / 1000, None, s["median_ms"]["clef27b"] / 1000],
          "top": 14, "dec": 1, "unit": " s"}],
        [("Kev-4B", "--s1"), ("Clef-Flash 9B", "--s2"), ("Kev + Flash, split", "--ink2"), ("Clef 27B", "--s3")],
        "Two of Flash's nine false spams come from the pitch cut, which was tuned for Kev. Live, the split takes about 5.5\u00a0s: "
        "both models share one GPU.", label="Spam safety and speed of the four readers")
    c = json.load(open(repo / "tasks/chronos/exp_049_kev_rev5/results/test/compare.json"))
    ht, st = c["hard_timing"]["timeline"], c["standard"]["timeline"]
    dated = V.slope(
        [("month or holiday", ht["rev42"], ht["rev5"], "--s1"), ("ordinary", st["rev42"], st["rev5"], "--s2")],
        f"Timeline read right, before and after one sentence ('Today is 21 October 2026') in the timeline question. exp_049's "
        f"fresh test set: {ht['n']} enquiries written to give timing only as a month, date, season or holiday "
        f"({ht['rev5_only_right']} better, {ht['rev42_only_right']} worse), and {st['n']} ordinary ones. Flash answers the "
        "timeline in both versions; rev 5 also added two yes/no questions to the same call. One pass per version.",
        "without the date", "with the date", dec=3, top=1.0, bottom=0.3,
        series=[("timing as a month or holiday", "--s1"), ("ordinary", "--s2")], label="Telling the model the date")
    return {"ceiling-fields": fields + spam, "ceiling-dated-timeline": dated}
