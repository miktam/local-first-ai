#!/usr/bin/env python3
"""The image for the X post: figure 1 as one 1600x900 SVG, from score.json. python3.14 -I ximg.py <run> <out dir>"""
import json, sys
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
A = json.loads((run / "score.json").read_text())["arms"]
V = ["V1", "V2", "V3", "V4", "V5"]
LAB = ["1 pass", "2 passes", "3 passes", "4 passes", "5 passes"]
THEMES = {"dark": dict(bg="#1d1e20", ink="#f0efec", ink2="#c3c2b7", muted="#a3a29b", axis="#5a5a55", s1="#3987e5", s2="#d95926", grey="#8a8982"),
          "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#6f6d68", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", grey="#a9a89e")}
F = "font-family=\"Arial, Helvetica, sans-serif\""

def panel(x, ttl, note, vals, vmax, fmt, fill, c):
    x0, wmax, y0, rh = x + 150, 235, 330, 76
    s = [f'<text x="{x}" y="250" {F} font-size="33" font-weight="700" fill="{c["ink"]}">{ttl}</text>',
         f'<text x="{x}" y="288" {F} font-size="24" fill="{c["muted"]}">{note}</text>',
         f'<line x1="{x0}" x2="{x0}" y1="{y0 - 14}" y2="{y0 + 5 * rh - 22}" stroke="{c["axis"]}" stroke-width="2"/>']
    for i, v in enumerate(vals):
        y, w = y0 + i * rh, wmax * v / vmax
        s.append(f'<text x="{x0 - 16}" y="{y + 30}" text-anchor="end" {F} font-size="27" fill="{c["ink2"]}">{LAB[i]}</text>')
        s.append(f'<path d="M{x0},{y} H{x0 + w - 8:.1f} Q{x0 + w:.1f},{y} {x0 + w:.1f},{y + 8} V{y + 32} Q{x0 + w:.1f},{y + 40} {x0 + w - 8:.1f},{y + 40} H{x0} Z" fill="{fill}"/>')
        s.append(f'<text x="{x0 + w + 14:.1f}" y="{y + 31}" {F} font-size="31" font-weight="700" fill="{c["ink"]}">{fmt(v)}</text>')
    return "".join(s)

for name, c in THEMES.items():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><rect width="1600" height="900" fill="{c["bg"]}"/>',
         f'<text x="70" y="96" {F} font-size="44" font-weight="700" fill="{c["ink"]}">More loop passes: more right answers, about as much written</text>',
         f'<text x="70" y="146" {F} font-size="27" fill="{c["ink2"]}">Thinking Machines\' TML-d24-varloop-research at 1 to 5 passes · 96 tool-work decisions · Mac mini, 4-bit, temperature 0</text>',
         panel(70, "Right answers, of 96", "higher is better", [A[a]["right"] for a in V], 96, str, c["s1"], c),
         panel(585, "Tokens written, thousands", "all 96 decisions, reasoning included", [A[a]["completion_tokens"] / 1000 for a in V], 110, lambda v: f"{v:.0f}", c["grey"], c),
         panel(1100, "Tokens a second", "the price of a pass", [A[a]["median_tokens_per_second"] for a in V], 52, lambda v: f"{v:.1f}", c["s2"], c),
         f'<text x="70" y="770" {F} font-size="25" fill="{c["ink2"]}">Made-up jobs with simulated tools, one run per setting. 96 decisions separate 1 pass from 3 or more;</text>',
         f'<text x="70" y="806" {F} font-size="25" fill="{c["ink2"]}">they do not separate 3, 4 and 5. The maker says results away from 3 passes may understate the model.</text>',
         f'<text x="70" y="858" {F} font-size="25" font-weight="700" fill="{c["ink"]}">localfirstai.eu</text>',
         f'<text x="1530" y="858" text-anchor="end" {F} font-size="23" fill="{c["muted"]}">llama.cpp b12631 · ggml-org IQ4_NL · 2,048 tokens a turn</text>', "</svg>"]
    (out / f"loop-passes-{name}.svg").write_text("".join(s))
print("ok")
