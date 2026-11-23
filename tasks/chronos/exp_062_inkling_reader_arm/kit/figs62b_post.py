#!/usr/bin/env python3
"""Figures for the exp_062 Addendum B post, generated from the run's score.json and detail.json (counts only).
    python3.14 -I figs.py <run folder> <out folder>      writes fig1.html, fig2.html, fig3.html and test pages
"""
import json
import re
import sys
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
score = json.loads((run / "score.json").read_text())
detail = json.loads((run / "detail.json").read_text())
A = score["arms"]
V = ["V1", "V2", "V3", "V4", "V5"]
PASS = {"V1": "1 pass", "V2": "2 passes", "V3": "3 passes", "V4": "4 passes", "V5": "5 passes"}


def bar(x0, y, w, h, fill, title):
    if w < 1:
        return f'<line x1="{x0}" x2="{x0}" y1="{y}" y2="{y + h}" stroke="{fill}" stroke-width="3"><title>{title}</title></line>'
    r = min(4, w)
    return (f'<path d="M{x0},{y} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y} {x0 + w:.1f},{y + r} V{y + h - r} '
            f'Q{x0 + w:.1f},{y + h} {x0 + w - r:.1f},{y + h} H{x0} Z" fill="{fill}"><title>{title}</title></path>')


def panel(ttl, note, rows, vmax, fmt, fill="var(--s1)", x0=78, wmax=150, rh=26):
    """rows: [(label, value, fill or None)]"""
    h = len(rows) * rh + 8
    aria = f"{ttl}: " + ", ".join(f"{l} {fmt(v)}" for l, v, _ in rows)
    s = [f'<div><div class="ttl">{ttl}</div><div class="note">{note}</div><svg viewBox="0 0 300 {h}" role="img" aria-label="{aria}">',
         f'<line x1="{x0}" x2="{x0}" y1="2" y2="{h - 2}" stroke="var(--base)" stroke-width="1"/>']
    for i, (l, v, f) in enumerate(rows):
        y = 6 + i * rh
        w = wmax * v / vmax
        s.append(f'<text x="{x0 - 8}" y="{y + 11.5}" text-anchor="end" font-size="12" fill="var(--ink2)">{l}</text>')
        s.append(bar(x0, y, w, 14, f or fill, f"{ttl}: {l} {fmt(v)}"))
        s.append(f'<text x="{x0 + w + 6:.1f}" y="{y + 11.5}" font-size="12.5" fill="var(--ink)">{fmt(v)}</text>')
    s.append("</svg></div>")
    return "".join(s)


# Figure 1: the research model at one to five passes: right answers, tokens written, tokens a second
f1 = ['<figure class="lfa-viz" role="group" aria-label="The research model at one to five loop passes: right answers, tokens written, tokens a second">',
      '<div class="lfa-multi">',
      panel("Right answers, of 96", "higher is better", [(PASS[a], A[a]["right"], None) for a in V], 96, lambda v: str(v)),
      panel("Tokens written, thousands", "all 96 decisions, reasoning included", [(PASS[a], A[a]["completion_tokens"] / 1000, None) for a in V], 110,
            lambda v: f"{v:.0f}", fill="var(--base)"),
      panel("Tokens a second", "the price of a pass", [(PASS[a], A[a]["median_tokens_per_second"], None) for a in V], 52, lambda v: f"{v:.1f}", fill="var(--s2)"),
      '</div><figcaption class="cap">One file, <code>TML-d24-varloop-research</code> at 4-bit, with the number of loop passes set when it is loaded. '
      'The same 96 decisions at every setting, temperature 0, on a Mac mini. More passes: more right answers, about the same amount written, each token slower. '
      '96 decisions separate one pass from three or more; they do not separate three, four and five.</figcaption></figure>']
(out / "fig1.html").write_text("".join(f1))

# Figure 2: wrong decisions by pass count, with and without a turn cut at the cap
cut = {a: set(detail["written"][a]["decisions_with_a_cut_turn"]) for a in V}
rows2 = []
for a in V:
    wrong = [w["id"] for w in detail["wrong"][a]]
    c = sum(1 for i in wrong if i in cut[a])
    rows2.append((PASS[a], len(wrong) - c, c))
x0, sc, rh = 96, 19, 30
h = len(rows2) * rh + 8
aria = "; ".join(f"{l}: {n} wrong with no cut turn, {c} wrong with a turn cut at the cap" for l, n, c in rows2)
s = ['<figure class="lfa-viz" role="group" aria-label="Wrong decisions of the research model at each pass count, split by whether a turn was cut at the token cap">',
     '<div class="leg"><span><i style="background:var(--s1)"></i>a wrong step: a lookup skipped, an alert wrong or missing, a call where none was due</span>'
     '<span><i style="background:var(--s2)"></i>a turn cut at the 2,048-token cap</span></div>',
     f'<svg viewBox="0 0 660 {h}" role="img" aria-label="{aria}">',
     f'<line x1="{x0}" x2="{x0}" y1="2" y2="{h - 2}" stroke="var(--base)" stroke-width="1"/>']
for i, (l, n, c) in enumerate(rows2):
    y = 6 + i * rh
    s.append(f'<text x="{x0 - 10}" y="{y + 13}" text-anchor="end" font-size="13" fill="var(--ink)">{l}</text>')
    x = x0
    if n:
        w = n * sc
        if c:
            s.append(f'<rect x="{x}" y="{y}" width="{w - 1}" height="18" fill="var(--s1)"><title>{l}: {n} wrong with no cut turn</title></rect>')
        else:
            s.append(bar(x, y, w, 18, "var(--s1)", f"{l}: {n} wrong with no cut turn"))
        x += w + (1 if c else 0)
    if c:
        s.append(bar(x, y, c * sc, 18, "var(--s2)", f"{l}: {c} wrong with a turn cut at the cap"))
        x += c * sc
    txt = f"{n + c} wrong" + (f": {n} and {c}" if n and c else (", every one with a cut turn" if c else ""))
    s.append(f'<text x="{x + 8}" y="{y + 13.5}" font-size="13" font-weight="600" fill="var(--ink)">{txt}</text>')
s.append('</svg><figcaption class="cap">Each wrong decision of the 96, at each pass count. Orange: the decision has a turn in which the model wrote 2,048 tokens of reasoning and was cut off. '
         'From three passes up no decision is wrong without one. A decision counted orange can have a second fault; the text says which do.</figcaption></figure>')
(out / "fig2.html").write_text("".join(s))

# Figure 3: Inkling 2 Mini against gemma4:26b, all 96 and by group
groups = [("All 96 decisions", None), ("20 plain jobs", "1"), ("20 jobs with a complication", "2"), ("20 hardest jobs", "3"), ("36 events in three conversations", "stream")]
f3 = ['<figure class="lfa-viz" role="group" aria-label="Inkling 2 Mini against gemma4:26b: right answers on all 96 decisions and by group">',
      '<div class="leg"><span><i style="background:var(--base)"></i>gemma4:26b, on Ollama</span><span><i style="background:var(--s1)"></i>Inkling 2 Mini, on llama.cpp</span></div>',
      '<div class="lfa-multi">']
for ttl, t in groups:
    def val(a):
        return (A[a]["right"], A[a]["decisions"]) if t is None else (A[a]["by_tier"][t]["right"], A[a]["by_tier"][t]["n"])
    (g, n), (k, _) = val("G"), val("I")
    f3.append(panel(ttl, f"right, of {n}", [("gemma", g, "var(--base)"), ("Inkling", k, "var(--s1)")], n, lambda v: str(v)))
f3.append('</div><figcaption class="cap">The same 96 decisions for both models, each on its own runtime. The events of one conversation depend on each other, '
          'so the last panel carries less weight than 36 separate jobs would.</figcaption></figure>')
(out / "fig3.html").write_text("".join(f3))

# standalone test pages, light and dark
css = re.search(r"<!-- lfa-viz-css -->(.*?)<!-- /lfa-viz-css -->", Path.home().joinpath("local-first-ai-blog/layouts/partials/extend_head.html").read_text(), re.S).group(1)
for n in (1, 2, 3):
    fig = (out / f"fig{n}.html").read_text()
    for theme, bg in (("light", "#ffffff"), ("dark", "#1d1e20")):
        (out / f"test{n}-{theme}.html").write_text(
            f'<!doctype html><html data-theme="{theme}"><head><meta charset="utf-8">{css}</head>'
            f'<body style="background:{bg};margin:24px;max-width:720px;font-size:17px">{fig}</body></html>')
print(rows2)
print({a: (A[a]["right"], A[a]["completion_tokens"], A[a]["median_tokens_per_second"]) for a in V})
