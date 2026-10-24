#!/usr/bin/env python3
"""
exp_050 — the four figures for the localfirstai.eu post, as inline HTML + SVG built from the records (no hand-typed
numbers except the dev-free registered summary, which is read from runs/exp050_test_compare.json).

  EXP50_SCORER=corrected .venv/bin/python3 scripts/figures50.py > fig.html

Figure 3 (every lead, three runs) and Figure 4 (Kev's confidence) are computed from the test predictions; Figure 2 reads
the registered compare; Figure 1 is a diagram. Colors: the team = categorical slot 1, one model = slot 2, validated on
the blog's light (#ffffff) and dark (#1d1e20) surfaces. Light and dark follow PaperMod's :root[data-theme="dark"].
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EXP50_SCORER", "corrected")
from scripts.leads50 import D, ROOT as R, _inv_by_ref, load, outcome  # noqa: E402

STYLE = """<style>
.lfa-viz{--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;--base:#c3c2b7;--same:#dedcd4;
--team:#2a78d6;--team-soft:#d6e6fa;--team-mute:#86b6ef;--one:#eb6834;--one-soft:#fbe0d5;--code:#f1f0ec;--code-line:#c3c2b7;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);margin:1.6em 0}
:root[data-theme="dark"] .lfa-viz{--surface:#1d1e20;--ink:#f0efec;--ink2:#c3c2b7;--muted:#a3a29b;--grid:#2c2c2a;--base:#4a4a46;
--same:#3a3b3e;--team:#3987e5;--team-soft:#1f3552;--team-mute:#184f95;--one:#d95926;--one-soft:#4a2a1c;--code:#2a2b2e;--code-line:#4a4a46}
.lfa-viz .cap{font-size:.86em;color:var(--ink2);margin-top:.5em;line-height:1.4;font-weight:400;text-align:left}
.lfa-viz .leg{display:flex;flex-wrap:wrap;gap:.4em 1.2em;font-size:.84em;color:var(--ink2);margin:.2em 0 .7em}
.lfa-viz .leg i{display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.4em;vertical-align:-.05em}
.lfa-viz svg{display:block;width:100%;height:auto}
.lfa-viz svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.lfa-flow{display:grid;grid-template-columns:1fr;gap:.9em}
.lfa-lane{border:1px solid var(--grid);border-radius:8px;padding:.7em .8em;background:var(--surface)}
.lfa-lane h4{margin:0 0 .5em;font-size:.95em}
.lfa-steps{display:flex;flex-wrap:wrap;gap:.35em;align-items:stretch}
.lfa-step{flex:1 1 9.5em;border-radius:6px;padding:.45em .55em;font-size:.82em;line-height:1.3;border:1px solid var(--code-line);background:var(--code)}
.lfa-step b{display:block;font-size:1.02em;color:var(--ink)}
.lfa-step span{color:var(--ink2)}
.lfa-step em{display:block;font-style:normal;font-size:.78em;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:.15em}
.lfa-step.m-team{background:var(--team-soft);border-color:var(--team)}
.lfa-step.m-one{background:var(--one-soft);border-color:var(--one)}
.lfa-out{font-size:.82em;color:var(--ink2);margin-top:.45em}
.lfa-multi{display:grid;grid-template-columns:repeat(auto-fit,minmax(15em,1fr));gap:.8em 1.4em}
.lfa-waffles{display:flex;flex-wrap:wrap;gap:1em 2em}
.lfa-waffles>div{flex:1 1 16em;max-width:24em}
.lfa-waffles h4{margin:0 0 .3em;font-size:.92em}
.lfa-waffles p{margin:.35em 0 0;font-size:.84em;color:var(--ink2);line-height:1.35}
</style>"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def bar_path(x0, y, w, h, r=4):
    """Horizontal bar: square at the baseline (x0), 4px rounded data-end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    x1 = x0 + w
    return (f"M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + h - r} Q{x1},{y + h} {x1 - r},{y + h} H{x0} Z")


def col_path(x, y_base, w, h, r=4):
    """Vertical column: square at the baseline, 4px rounded top."""
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    yt = y_base - h
    return (f"M{x},{y_base} V{yt + r} Q{x},{yt} {x + r},{yt} H{x + w - r} Q{x + w},{yt} {x + w},{yt + r} V{y_base} Z")


# --------------------------------------------------------------------------------------------------------------------
def fig1() -> str:
    team = [("model", "Kev-4B + Clef-Flash 9B", "closed questions: what does the sender want, how soon, qualified?"),
            ("code", "The agency's rules", "decide the next step"),
            ("model", "Clef-Flash 9B", "areas, bedrooms, must-have features"),
            ("code", "Match against the inventory", "up to 3 homes that fit, or none"),
            ("model", "gemma4:26b", "1–3 sentences, no numbers allowed"),
            ("code", "Print listings, final check", "facts from the record; language, numbers, contacts")]
    one = [("model", "gemma4:26b, thinking about a minute", "gets the enquiry, the rules written out and the whole inventory; "
            "returns the next step, the homes, the reply and whether a person should handle it")]

    def lane(title, steps, cls):
        s = "".join(f'<div class="lfa-step{" " + cls if k == "model" else ""}"><em>{(str(n) + " · ") if len(steps) > 1 else ""}{k}</em><b>{esc(t)}</b><span>{esc(d)}</span></div>'
                    for n, (k, t, d) in enumerate(steps, 1))
        return (f'<div class="lfa-lane"><h4>{esc(title)}</h4><div class="lfa-steps">{s}</div>'
                f'<div class="lfa-out">→ ready to send, or handed to a person</div></div>')
    return (f'<figure class="lfa-viz" role="group" aria-label="How the two builds work">'
            f'<div class="leg"><span><i style="background:var(--team-soft);border:1px solid var(--team)"></i>model step, the team</span>'
            f'<span><i style="background:var(--one-soft);border:1px solid var(--one)"></i>model step, one model</span>'
            f'<span><i style="background:var(--code);border:1px solid var(--code-line)"></i>code step</span></div>'
            f'<div class="lfa-flow">{lane("The team of small models", team, "m-team")}{lane("One model", one, "m-one")}</div>'
            f'<figcaption class="cap">The same writer model, gemma4:26b, sits in both. In the team it writes only a few '
            f'sentences; the decisions come from the two decision models and code.</figcaption></figure>')


# --------------------------------------------------------------------------------------------------------------------
def fig2(reg: dict) -> str:
    a = reg["arms"]
    med = {}
    for arm in ("G", "M"):
        ms = sorted(p["ms"] for p in load(R / "runs" / f"exp050_test_{arm}_r1" / "pred.jsonl") if p.get("ms"))
        med[arm] = ms[len(ms) // 2] / 1000
    panels = [
        ("Replies that should go out, sent and right", "%", 100 * a["G r1"]["right_ready_of_should"], 100 * a["M r1"]["right_ready_of_should"], 100, "higher is better"),
        ("Same decision in all three runs", "%", 100 * reg["hypotheses"]["H3"]["G"], 100 * reg["hypotheses"]["H3"]["M"], 100, "higher is better"),
        ("Wrong replies sent", "%", 100 * a["G r1"]["wrong_sent"], 100 * a["M r1"]["wrong_sent"], 25, "lower is better"),
        ("With a 4B model in place of the 26B: wrong replies sent", "%", 100 * a["Gs r1"]["wrong_sent"], 100 * a["Ms r1"]["wrong_sent"], 25, "lower is better"),
        ("Seconds per enquiry (median)", " s", med["G"], med["M"], 80, "lower is better"),
    ]
    out = []
    for title, unit, g, m, top, note in panels:
        dec = 1 if "wrong" in title.lower() else 0
        W, lab, bw, bh, gap = 300, 92, 160, 16, 14
        y0 = 8
        rows = ""
        for i, (name, v, var) in enumerate((("Team", g, "--team"), ("One model", m, "--one"))):
            y = y0 + i * (bh + gap)
            w = bw * v / top
            val = f"{v:.{dec}f}{' %' if unit == '%' else unit}"
            rows += (f'<text x="{lab - 8}" y="{y + bh * .75}" text-anchor="end" font-size="12" fill="var(--ink2)">{name}</text>'
                     f'<path d="{bar_path(lab, y, w, bh)}" fill="var({var})"><title>{esc(title)}: {name} {val}</title></path>'
                     f'<text x="{lab + w + 6}" y="{y + bh * .75}" font-size="12.5" font-weight="600" fill="var(--ink)">{val}</text>')
        H = y0 + 2 * bh + gap + 10
        base = f'<line x1="{lab}" x2="{lab}" y1="{y0 - 4}" y2="{y0 + 2 * bh + gap + 4}" stroke="var(--base)" stroke-width="1"/>'
        out.append(f'<div><div style="font-size:.86em;font-weight:600;line-height:1.3">{esc(title)}</div>'
                   f'<div style="font-size:.76em;color:var(--muted);margin-bottom:.2em">{note}</div>'
                   f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}: team {g:.1f}, one model {m:.1f}">{base}{rows}</svg></div>')
    return (f'<figure class="lfa-viz"><div class="leg"><span><i style="background:var(--team)"></i>Team of small models</span>'
            f'<span><i style="background:var(--one)"></i>One model, thinking</span></div>'
            f'<div class="lfa-multi">{"".join(out)}</div>'
            f'<figcaption class="cap">247 synthetic enquiries, registered scoring, run 1 (stability across runs 1–3). '
            f'Each chart has its own scale.</figcaption></figure>')


# --------------------------------------------------------------------------------------------------------------------
def fig3(rows: dict) -> str:
    ids = list(rows)                                  # test.jsonl order, the same for both builds
    panels, summary = [], {}
    for arm, var, title in (("G", "--team", "Team of small models"), ("M", "--one", "One model, thinking")):
        O = {k: {i: outcome(rows[i], p, _inv_by_ref(k)) for i, p in ((x["id"], x) for x in load(R / "runs" / f"exp050_test_{arm}_r{k}" / "pred.jsonl"))}
             for k in (1, 2, 3)}
        cols, cell, gp = 19, 12, 2
        svg, changed, wrong_all, wrong_some = "", 0, 0, 0
        for n, i in enumerate(ids):
            x, y = (n % cols) * (cell + gp), (n // cols) * (cell + gp)
            ch = not (O[1][i]["decision"] == O[2][i]["decision"] == O[3][i]["decision"])
            w = sum(O[k][i]["wrong_sent"] for k in (1, 2, 3))
            changed += ch; wrong_all += w == 3; wrong_some += 0 < w < 3
            svg += f'<g><rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="var({var if ch else "--same"})"/>'
            if w == 3:
                svg += f'<circle cx="{x + cell / 2}" cy="{y + cell / 2}" r="3.2" fill="var(--ink)"/>'
            elif w:
                svg += f'<circle cx="{x + cell / 2}" cy="{y + cell / 2}" r="2.6" fill="none" stroke="var(--ink)" stroke-width="1.4"/>'
            svg += "</g>"
        W, H = cols * (cell + gp) - gp, -(-len(ids) // cols) * (cell + gp) - gp
        summary[arm] = (changed, wrong_all, wrong_some)
        line = (f"{changed} of {len(ids)} decisions changed between runs. "
                + (f"Wrong replies on {wrong_all + wrong_some} leads, the same {wrong_all} in every run."
                   if not wrong_some else f"Wrong replies on {wrong_all + wrong_some} different leads, {wrong_all} of them in every run."))
        panels.append(f'<div><h4>{title}</h4><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}: {esc(line)}">{svg}</svg><p>{line}</p></div>')
    return (f'<figure class="lfa-viz"><div class="leg">'
            f'<span><i style="background:var(--same)"></i>same decision in all 3 runs</span>'
            f'<span><i style="background:var(--team)"></i><i style="background:var(--one);margin-left:-.2em"></i>decision changed</span>'
            f'<span><svg width="12" height="12" style="display:inline;width:12px;vertical-align:-1px;margin-right:.4em"><circle cx="6" cy="6" r="3.2" fill="var(--ink)"/></svg>wrong reply sent in all 3 runs</span>'
            f'<span><svg width="12" height="12" style="display:inline;width:12px;vertical-align:-1px;margin-right:.4em"><circle cx="6" cy="6" r="2.6" fill="none" stroke="var(--ink)" stroke-width="1.4"/></svg>in 1 or 2 runs</span></div>'
            f'<div class="lfa-waffles">{"".join(panels)}</div>'
            f'<figcaption class="cap">Every test enquiry is one square, in the same position in both grids. '
            f'Corrected scoring. The ids are in the published predictions.</figcaption></figure>'), summary


# --------------------------------------------------------------------------------------------------------------------
def fig4(rows: dict) -> str:
    P = load(R / "runs" / "exp050_test_G_r1" / "pred.jsonl")
    bands = [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    stats = []
    for lo, hi in bands:
        b = [p for p in P if lo <= p["intent_conf"] < hi]
        stats.append((lo, hi, len(b), sum(p["intent"] == rows[p["id"]]["labels"]["intent"] for p in b) / len(b)))
    W, H, left, bottom, top = 420, 262, 42, 48, 54
    pw = W - left - 10
    ph = H - bottom - top
    cw = pw / len(bands)
    bw = min(24 * 1.6, cw * .5)
    svg = ""
    for t in (0, 50, 100):
        y = top + ph * (1 - t / 100)
        svg += (f'<line x1="{left}" x2="{W - 10}" y1="{y}" y2="{y}" stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{left - 6}" y="{y + 4}" text-anchor="end" font-size="12" fill="var(--muted)">{t} %</text>')
    for k, (lo, hi, n, acc) in enumerate(stats):
        x = left + k * cw + (cw - bw) / 2
        h = ph * acc
        hand = lo == 0
        fill = "var(--team)" if hand else "var(--team-mute)"
        lab = f"below {hi:.1f}" if hand else f"{lo:.1f}–{min(hi, 1):.1f}"
        svg += (f'<path d="{col_path(x, top + ph, bw, h)}" fill="{fill}"><title>Kev confidence {lab}: right {acc:.0%} of {n} leads</title></path>'
                f'<text x="{x + bw / 2}" y="{top + ph - h - 6}" text-anchor="middle" font-size="13" font-weight="{600 if hand else 400}" fill="var(--ink)">{100 * acc:.0f} %</text>'
                f'<text x="{x + bw / 2}" y="{top + ph + 16}" text-anchor="middle" font-size="12" fill="var(--ink2)">{lab}</text>'
                f'<text x="{x + bw / 2}" y="{top + ph + 32}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{n} leads</text>')
    xcut = left + cw
    svg += (f'<line x1="{xcut}" x2="{xcut}" y1="{top - 50}" y2="{top + ph}" stroke="var(--ink2)" stroke-width="1"/>'
            f'<text x="{left + cw / 2}" y="{top - 40}" text-anchor="middle" font-size="12" fill="var(--ink2)">handed to</text>'
            f'<text x="{left + cw / 2}" y="{top - 26}" text-anchor="middle" font-size="12" fill="var(--ink2)">a person</text>'
            f'<text x="{(xcut + W - 10) / 2}" y="{top - 26}" text-anchor="middle" font-size="12" fill="var(--ink2)">the team decides</text>'
            f'<line x1="{left}" x2="{W - 10}" y1="{top + ph}" y2="{top + ph}" stroke="var(--base)" stroke-width="1"/>')
    return (f'<figure class="lfa-viz"><svg style="max-width:560px" viewBox="0 0 {W} {H}" role="img" aria-label="Kev-4B intent accuracy by its own confidence: '
            f'{", ".join(f"{s[3]:.0%} at {s[0]:.1f}" for s in stats)}">{svg}</svg>'
            f'<figcaption class="cap">How often Kev-4B read the sender\'s request right, by its own confidence (test set, run 1). '
            f'Below 0.5 the team hands the lead to a person, though Kev was right on {100 * stats[0][3]:.0f}\u00a0% of those {stats[0][2]}.</figcaption></figure>')


if __name__ == "__main__":
    rows = {r["id"]: r for r in load(D / "test.jsonl")}
    reg = json.load(open(R / "runs" / "exp050_test_compare.json"))
    f3, summ = fig3(rows)
    parts = {"style": STYLE, "fig1": fig1(), "fig2": fig2(reg), "fig3": f3, "fig4": fig4(rows)}
    print(json.dumps({k: v for k, v in parts.items()}, ensure_ascii=False))
    print(json.dumps({"fig3_summary": summ}), file=sys.stderr)
