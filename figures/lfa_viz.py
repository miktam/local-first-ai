"""
Figures for the localfirstai.eu posts: inline HTML + SVG, built only from files published in this repository.

One style block (CSS) for the whole site, written into the blog's head partial by build.py. Every component returns a
<figure class="lfa-viz"> string. Colours follow the dataviz method: categorical slots 1–3 (blue, orange, aqua),
validated all-pairs on the blog's surfaces (#ffffff light, #1d1e20 dark; aqua on white is 2.8:1, so every mark carries
a visible label). Light and dark follow PaperMod's :root[data-theme="dark"]. Marks: bars at most 24 px thick (we use 16),
a 4 px rounded data end, square at the baseline; hairline grids; text in ink, never in the series colour.
"""
from __future__ import annotations

import html
import math

CSS = """.lfa-viz{--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;--base:#c3c2b7;--same:#dedcd4;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s1-soft:#d6e6fa;--s2-soft:#fbe0d5;--s1-mute:#86b6ef;
--team:var(--s1);--one:var(--s2);--team-soft:var(--s1-soft);--one-soft:var(--s2-soft);--team-mute:var(--s1-mute);
--code:#f1f0ec;--code-line:#c3c2b7;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);margin:1.6em 0}
:root[data-theme="dark"] .lfa-viz{--surface:#1d1e20;--ink:#f0efec;--ink2:#c3c2b7;--muted:#a3a29b;--grid:#2c2c2a;--base:#4a4a46;
--same:#3a3b3e;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s1-soft:#1f3552;--s2-soft:#4a2a1c;--s1-mute:#184f95;
--code:#2a2b2e;--code-line:#4a4a46}
.lfa-viz .cap{font-size:.86em;color:var(--ink2);margin-top:.5em;line-height:1.4;font-weight:400;text-align:left}
.lfa-viz .leg{display:flex;flex-wrap:wrap;gap:.4em 1.2em;font-size:.84em;color:var(--ink2);margin:.2em 0 .7em}
.lfa-viz .leg i{display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.4em;vertical-align:-.05em}
.lfa-viz svg{display:block;width:100%;height:auto}
.lfa-viz svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.lfa-viz .ttl{font-size:.86em;font-weight:600;line-height:1.3}
.lfa-viz .note{font-size:.76em;color:var(--muted);margin-bottom:.2em}
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
.lfa-waffles p{margin:.35em 0 0;font-size:.84em;color:var(--ink2);line-height:1.35}"""

SLOTS = ("--s1", "--s2", "--s3")


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt(v: float, dec: int = 0, unit: str = "") -> str:
    s = f"{v:,.{dec}f}"
    if unit == "%":
        return s + " %"
    return s + unit


def hbar(x0, y, w, h, r=4):
    """Horizontal bar, square at the baseline x0, 4 px rounded data end; negative widths grow left."""
    if abs(w) < 0.01:
        return ""
    r = min(r, abs(w), h / 2)
    if w > 0:
        x1 = x0 + w
        return f"M{x0:.2f},{y:.2f} H{x1 - r:.2f} Q{x1:.2f},{y:.2f} {x1:.2f},{y + r:.2f} V{y + h - r:.2f} Q{x1:.2f},{y + h:.2f} {x1 - r:.2f},{y + h:.2f} H{x0:.2f} Z"
    x1 = x0 + w
    return f"M{x0:.2f},{y:.2f} H{x1 + r:.2f} Q{x1:.2f},{y:.2f} {x1:.2f},{y + r:.2f} V{y + h - r:.2f} Q{x1:.2f},{y + h:.2f} {x1 + r:.2f},{y + h:.2f} H{x0:.2f} Z"


def vcol(x, y_base, w, h, r=4):
    """Vertical column, square at the baseline, 4 px rounded top."""
    if h <= 0.01:
        return ""
    r = min(r, w / 2, h)
    yt = y_base - h
    return f"M{x:.2f},{y_base:.2f} V{yt + r:.2f} Q{x:.2f},{yt:.2f} {x + r:.2f},{yt:.2f} H{x + w - r:.2f} Q{x + w:.2f},{yt:.2f} {x + w:.2f},{yt + r:.2f} V{y_base:.2f} Z"


def legend(items) -> str:
    """items: [(label, css var)]. Present whenever two or more series share a chart."""
    return '<div class="leg">' + "".join(f'<span><i style="background:var({v})"></i>{esc(l)}</span>' for l, v in items) + "</div>"


def figure(body: str, caption, label: str = "") -> str:
    """Wrap a chart body; with caption=None the bare body comes back, so several charts can share one figure."""
    if caption is None:
        return body
    return (f'<figure class="lfa-viz" role="group" aria-label="{esc(label or caption)}">{body}'
            f'<figcaption class="cap">{caption}</figcaption></figure>')


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson interval for k of n (descriptive; computed here, not read from a record)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def step_dec(ticks) -> int:
    """Decimals that print every tick exactly (0.25 needs 2, 0.2 needs 1)."""
    for d in range(0, 5):
        if all(abs(round(x, d) - x) < 1e-9 for x in ticks):
            return d
    return 4


def nice_ticks(lo, hi, n=4):
    span = hi - lo or 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


# ---------------------------------------------------------------------------------------------------------------------
def bar_pairs(panels, series, caption, label=""):
    """Small multiples, one panel per measure, each on its own scale; a bar per series.
    panels: [{"title", "note", "values": [..per series..], "top", "unit", "dec"}]; series: [(name, css var)]."""
    out = []
    for p in panels:
        lab = max(60, 6.6 * max(len(n) for n, _ in series) + 14)
        W, bw, bh, gap = 300, 300 - lab - 56, 16, 12
        y0 = 8
        rows = ""
        for i, ((name, var), v) in enumerate(zip(series, p["values"])):
            y = y0 + i * (bh + gap)
            if v is None:
                rows += (f'<text x="{lab - 8}" y="{y + bh * .75}" text-anchor="end" font-size="12" fill="var(--ink2)">{esc(name)}</text>'
                         f'<text x="{lab + 4}" y="{y + bh * .75}" font-size="12" fill="var(--muted)">not measured</text>')
                continue
            w = bw * v / p["top"]
            val = fmt(v, p.get("dec", 0), p.get("unit", ""))
            rows += (f'<text x="{lab - 8}" y="{y + bh * .75}" text-anchor="end" font-size="12" fill="var(--ink2)">{esc(name)}</text>'
                     f'<path d="{hbar(lab, y, w, bh)}" fill="var({var})"><title>{esc(p["title"])}: {esc(name)} {esc(val)}</title></path>'
                     f'<text x="{lab + w + 6}" y="{y + bh * .75}" font-size="12.5" font-weight="600" fill="var(--ink)">{esc(val)}</text>')
        n = len(series)
        H = y0 + n * bh + (n - 1) * gap + 10
        base = f'<line x1="{lab}" x2="{lab}" y1="{y0 - 4}" y2="{y0 + n * bh + (n - 1) * gap + 4}" stroke="var(--base)" stroke-width="1"/>'
        out.append(f'<div><div class="ttl">{esc(p["title"])}</div><div class="note">{esc(p.get("note", ""))}</div>'
                   f'<svg style="max-width:360px" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(p["title"])}">{base}{rows}</svg></div>')
    leg = legend(series) if len(series) > 1 else ""
    return figure(f'{leg}<div class="lfa-multi">{"".join(out)}</div>', caption, label)


def hbars(rows, caption, unit="", dec=0, top=None, label="", highlight=None, var="--s1"):
    """One series of horizontal bars, one per category. rows: [(category, value, note or '')]."""
    top = top or max(v for _, v, _ in rows) * 1.12
    W, lab, bw, bh, gap = 600, 190, 330, 16, 10
    svg, y = "", 6
    for cat, v, note in rows:
        fill = f"var({var})" if (highlight is None or cat in highlight) else "var(--s1-mute)"
        w = bw * v / top
        val = fmt(v, dec, unit)
        svg += (f'<text x="{lab - 10}" y="{y + bh * .75}" text-anchor="end" font-size="13" fill="var(--ink2)">{esc(cat)}</text>'
                f'<path d="{hbar(lab, y, w, bh)}" fill="{fill}"><title>{esc(cat)}: {esc(val)}</title></path>'
                f'<text x="{lab + w + 6}" y="{y + bh * .75}" font-size="13" font-weight="600" fill="var(--ink)">{esc(val)}'
                + (f'<tspan font-weight="400" fill="var(--muted)"> {esc(note)}</tspan>' if note else "") + "</text>")
        y += bh + gap
    H = y + 2
    svg = f'<line x1="{lab}" x2="{lab}" y1="2" y2="{H - 6}" stroke="var(--base)" stroke-width="1"/>' + svg
    return figure(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def dot_ci(rows, caption, xlab="", unit="", dec=2, label="", series=None, xmin=None, xmax=None, zero=0.0, refs=(), signed=True):
    """Estimates with intervals on one shared axis, a reference line at zero.
    rows: [{"label", "est", "lo", "hi", "var", "note"}]; series: legend items if colours carry identity."""
    lo = min(r["lo"] for r in rows) if xmin is None else xmin
    hi = max(r["hi"] for r in rows) if xmax is None else xmax
    pad = (hi - lo) * .08
    lo, hi = min(lo - pad, zero), max(hi + pad, zero)
    W, right, rh = 660, 132, 34
    lab = max(140, 7.0 * max(len(r["label"]) for r in rows) + 18)
    pw = W - lab - right
    sg = (lambda v: "+" if v > 0 else "") if signed else (lambda v: "")

    def X(v):
        return lab + (v - lo) / (hi - lo) * pw
    H = 28 + len(rows) * rh + 30 + (18 if xlab else 0)
    svg = ""
    ticks = nice_ticks(lo, hi, 5)
    tdec = step_dec(ticks)
    for t in ticks:
        if lo - 1e-9 <= t <= hi + 1e-9:
            svg += (f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="18" y2="{H - 28 - (18 if xlab else 0)}" stroke="var(--grid)" stroke-width="1"/>'
                    f'<text x="{X(t):.1f}" y="{H - 12 - (18 if xlab else 0)}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{sg(t) + fmt(t, tdec, unit)}</text>')
    svg += f'<line x1="{X(zero):.1f}" x2="{X(zero):.1f}" y1="12" y2="{H - 28 - (18 if xlab else 0)}" stroke="var(--ink2)" stroke-width="1.2"/>'
    for rv, rl in refs:
        svg += (f'<line x1="{X(rv):.1f}" x2="{X(rv):.1f}" y1="14" y2="{H - 28 - (18 if xlab else 0)}" stroke="var(--ink2)" stroke-width="1" opacity=".55"/>'
                f'<text x="{X(rv) + 4:.1f}" y="22" font-size="11" fill="var(--ink2)">{esc(rl)}</text>')
    for i, r in enumerate(rows):
        y = 28 + i * rh + rh / 2
        v = f'var({r.get("var", "--s1")})'
        est = sg(r["est"]) + fmt(r["est"], dec, unit)
        rng = f'{fmt(r["lo"], dec, unit)} to {fmt(r["hi"], dec, unit)}'
        svg += (f'<text x="{lab - 12}" y="{y + 4}" text-anchor="end" font-size="13" fill="var(--ink2)">{esc(r["label"])}</text>'
                f'<g><title>{esc(r["label"])}: {esc(est)} ({esc(rng)})</title>'
                f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y}" y2="{y}" stroke="{v}" stroke-width="2.5" stroke-linecap="round"/>'
                f'<circle cx="{X(r["est"]):.1f}" cy="{y}" r="5.5" fill="{v}" stroke="var(--surface)" stroke-width="2"/></g>'
                f'<text x="{W - right + 10:.1f}" y="{y + 4}" font-size="12.5" font-weight="600" fill="var(--ink)">{esc(est)}'
                + (f'<tspan font-weight="400" fill="var(--muted)"> {esc(r["note"])}</tspan>' if r.get("note") else "") + "</text>")
    if xlab:
        svg += f'<text x="{lab + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{esc(xlab)}</text>'
    leg = legend(series) if series else ""
    return figure(f'{leg}<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def line_chart(series, caption, xlab, ylab, xticks=None, yunit="", ydec=0, xdec=0, label="", logx=False, ymin=0, ymax=None, end_labels=True,
               hrefs=(), xfmt=None):
    """Lines over a continuous x. series: [(name, css var, [(x, y), ...])] or with a 4th item {"dash": bool, "end": label or None};
    one shared y axis; hrefs: [(y, label, css var)] reference lines."""
    xs = [x for s_ in series for x, _ in s_[2]]
    ys = [y for s_ in series for _, y in s_[2]]
    x0, x1 = min(xs), max(xs)
    y1 = ymax if ymax is not None else max(ys) * 1.1
    W, left, right, top, bottom = 640, 56, 130 if end_labels else 20, 30, 44
    pw, ph = W - left - right, 260
    H = top + ph + bottom

    def X(x):
        if logx:
            return left + (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0)) * pw
        return left + (x - x0) / (x1 - x0) * pw

    def Y(y):
        return top + ph * (1 - (y - ymin) / (y1 - ymin))
    svg = ""
    for t in nice_ticks(ymin, y1, 4):
        if t <= y1 + 1e-9:
            svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                    f'<text x="{left - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11.5" fill="var(--muted)">{fmt(t, ydec, yunit)}</text>')
    for t in (xticks or nice_ticks(x0, x1, 5)):
        svg += f'<text x="{X(t):.1f}" y="{top + ph + 18}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{xfmt(t) if xfmt else fmt(t, xdec)}</text>'
    for hy, hl, hv in hrefs:
        svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(hy):.1f}" y2="{Y(hy):.1f}" stroke="var({hv})" stroke-width="1" opacity=".7"/>'
                f'<text x="{left + 4}" y="{Y(hy) - 4:.1f}" font-size="10.5" fill="var(--ink2)">{esc(hl)}</text>')
    svg += f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="var(--base)" stroke-width="1"/>'
    svg += f'<text x="{left + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" font-size="12" fill="var(--ink2)">{esc(xlab)}</text>'
    svg += f'<text x="{left - 46}" y="{top - 16}" font-size="12" fill="var(--ink2)">{esc(ylab)}</text>'
    for s_ in series:
        name, var, pts = s_[:3]
        opt = s_[3] if len(s_) > 3 else {}
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if opt.get("dash") else ""
        svg += f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2"{dash} stroke-linejoin="round" stroke-linecap="round"/>'
        for x, y in pts:
            svg += (f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="var({var})" stroke="var(--surface)" stroke-width="2">'
                    f'<title>{esc(name)}: {fmt(x, xdec)} → {fmt(y, ydec, yunit)}</title></circle>')
        if end_labels and opt.get("end", name):
            xe, ye = pts[-1]
            svg += f'<text x="{X(xe) + 9:.1f}" y="{Y(ye) + 4 + opt.get("dy", 0):.1f}" font-size="12.5" fill="var(--ink)">{esc(opt.get("end", name))}</text>'
    def key(s_):
        o = s_[3] if len(s_) > 3 else {}
        line = f'<line x1="1" x2="21" y1="7" y2="7" stroke="var({s_[1]})" stroke-width="2.5"{" stroke-dasharray=\"5 3\"" if o.get("dash") else ""}/>'
        return f'<span><svg width="22" height="14" style="display:inline;width:22px;vertical-align:-2px;margin-right:.35em">{line}</svg>{esc(s_[0])}</span>'
    leg = f'<div class="leg">{"".join(key(s_) for s_ in series)}</div>' if len(series) > 1 else ""
    return figure(f'{leg}<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def slope(rows, caption, left_label, right_label, unit="", dec=2, top=1.0, bottom=0.0, label="", var="--s1", series=None):
    """Before → after per item, one shared scale. rows: [(item, before, after)] or [(item, before, after, css var)]."""
    W, H, x0, x1, top_m, bot_m = 560, 300, 170, 390, 34, 20
    ph = H - top_m - bot_m

    def Y(v):
        return top_m + ph * (1 - (v - bottom) / (top - bottom))
    svg = (f'<text x="{x0}" y="16" text-anchor="middle" font-size="12" fill="var(--ink2)">{esc(left_label)}</text>'
           f'<text x="{x1}" y="16" text-anchor="middle" font-size="12" fill="var(--ink2)">{esc(right_label)}</text>'
           f'<line x1="{x0}" x2="{x0}" y1="{top_m - 8}" y2="{H - bot_m}" stroke="var(--base)" stroke-width="1"/>'
           f'<line x1="{x1}" x2="{x1}" y1="{top_m - 8}" y2="{H - bot_m}" stroke="var(--base)" stroke-width="1"/>')
    def spread(vals, gap=15):
        order = sorted(range(len(vals)), key=lambda k: vals[k])
        out = list(vals)
        for j in range(1, len(order)):
            a_, b_ = order[j - 1], order[j]
            if out[b_] - out[a_] < gap:
                out[b_] = out[a_] + gap
        return out
    ly = spread([-Y(r[1]) for r in rows])
    ry = spread([-Y(r[2]) for r in rows])
    for k, row in enumerate(rows):
        item, a, b = row[:3]
        var = row[3] if len(row) > 3 else var
        svg += (f'<g><title>{esc(item)}: {fmt(a, dec, unit)} → {fmt(b, dec, unit)}</title>'
                f'<line x1="{x0}" x2="{x1}" y1="{Y(a):.1f}" y2="{Y(b):.1f}" stroke="var({var})" stroke-width="2"/>'
                f'<circle cx="{x0}" cy="{Y(a):.1f}" r="4.5" fill="var({var})" stroke="var(--surface)" stroke-width="2"/>'
                f'<circle cx="{x1}" cy="{Y(b):.1f}" r="4.5" fill="var({var})" stroke="var(--surface)" stroke-width="2"/></g>'
                f'<text x="{x0 - 10}" y="{-ly[k] + 4:.1f}" text-anchor="end" font-size="12.5" fill="var(--ink)">{esc(item)} {fmt(a, dec, unit)}</text>'
                f'<text x="{x1 + 10}" y="{-ry[k] + 4:.1f}" font-size="12.5" font-weight="600" fill="var(--ink)">{fmt(b, dec, unit)} {esc(item)}</text>')
    leg = legend(series) if series else ""
    return figure(f'{leg}<svg style="max-width:560px" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def columns(bands, caption, unit="%", dec=0, top=100, label="", highlight=None, sublabels=None, header=None):
    """Vertical columns for ordered bands. bands: [(label, value)]; highlight: indices drawn in the accent."""
    W, H, left, bottom, tp = 420, 262, 42, 48, 54 if header else 24
    pw, ph = W - left - 10, H - bottom - tp
    cw = pw / len(bands)
    bw = min(38, cw * .5)
    svg = ""
    for t in nice_ticks(0, top, 2):
        y = tp + ph * (1 - t / top)
        svg += (f'<line x1="{left}" x2="{W - 10}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="var(--muted)">{fmt(t, 0, unit)}</text>')
    for k, (lab, v) in enumerate(bands):
        x = left + k * cw + (cw - bw) / 2
        h = ph * v / top
        hot = highlight is None or k in highlight
        svg += (f'<path d="{vcol(x, tp + ph, bw, h)}" fill="var({"--s1" if hot else "--s1-mute"})"><title>{esc(lab)}: {fmt(v, dec, unit)}</title></path>'
                f'<text x="{x + bw / 2:.1f}" y="{tp + ph - h - 6:.1f}" text-anchor="middle" font-size="13" font-weight="{600 if hot else 400}" fill="var(--ink)">{fmt(v, dec, unit)}</text>'
                f'<text x="{x + bw / 2:.1f}" y="{tp + ph + 16}" text-anchor="middle" font-size="12" fill="var(--ink2)">{esc(lab)}</text>')
        if sublabels:
            svg += f'<text x="{x + bw / 2:.1f}" y="{tp + ph + 32}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{esc(sublabels[k])}</text>'
    svg += f'<line x1="{left}" x2="{W - 10}" y1="{tp + ph}" y2="{tp + ph}" stroke="var(--base)" stroke-width="1"/>'
    return figure(f'<svg style="max-width:560px" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def diverging(rows, caption, label="", left_var="--s2", right_var="--s1", top=None, header=""):
    """Counts that moved each way, around a centre line. rows: [(label, left n, left text, right n, right text)];
    keep the side texts short (a word or two): they sit at the bar ends."""
    top = top or max(max(r[1], r[3]) for r in rows) * 1.05
    W, lab, bh, gap = 660, 190, 16, 18
    room = 7.2 * max(len(f"{r[1]} {r[2]}") for r in rows) + 12
    room_r = 7.2 * max(len(f"{r[3]} {r[4]}") for r in rows) + 12
    half = (W - lab - room - room_r) / 2
    cx = lab + room + half
    svg, y = "", 8 + (20 if header else 0)
    if header:
        svg += f'<text x="{cx:.1f}" y="13" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{esc(header)}</text>'
    for name, ln, lt, rn, rt in rows:
        wl, wr = half * ln / top, half * rn / top
        svg += (f'<text x="{lab - 12}" y="{y + bh * .75 + 7}" text-anchor="end" font-size="13" fill="var(--ink2)">{esc(name)}</text>'
                f'<path d="{hbar(cx, y + 7, -wl, bh)}" fill="var({left_var})"><title>{esc(name)}: {ln} {esc(lt)}</title></path>'
                f'<path d="{hbar(cx, y + 7, wr, bh)}" fill="var({right_var})"><title>{esc(name)}: {rn} {esc(rt)}</title></path>'
                f'<text x="{cx - wl - 6:.1f}" y="{y + bh * .75 + 7}" text-anchor="end" font-size="12.5" fill="var(--ink)"><tspan font-weight="600">{ln}</tspan> <tspan fill="var(--muted)">{esc(lt)}</tspan></text>'
                f'<text x="{cx + wr + 6:.1f}" y="{y + bh * .75 + 7}" font-size="12.5" fill="var(--ink)"><tspan font-weight="600">{rn}</tspan> <tspan fill="var(--muted)">{esc(rt)}</tspan></text>')
        y += bh + gap
    H = y + 6
    svg = f'<line x1="{cx}" x2="{cx}" y1="{22 if header else 2}" y2="{H - 4}" stroke="var(--ink2)" stroke-width="1"/>' + svg
    return figure(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def scatter_panels(panels, caption, xlab, ylab, xmax, ymin, ymax, limit=None, label="", xdec=0, xunit="%", ydec=0, yunit="%"):
    """Operating points in side-by-side panels sharing scales; an optional shaded region beyond an x limit.
    panels: [{"title", "points": [{"label", "x", "y", "var", "dx", "dy"}]}]."""
    out = []
    for p in panels:
        W, H, left, right, top, bottom = 330, 270, 46, 16, 14, 40
        pw, ph = W - left - right, H - top - bottom

        def X(v):
            return left + v / xmax * pw

        def Y(v):
            return top + ph * (1 - (v - ymin) / (ymax - ymin))
        svg = ""
        if limit is not None:
            svg += (f'<rect x="{X(limit[0]):.1f}" y="{top}" width="{left + pw - X(limit[0]):.1f}" height="{ph}" fill="var(--same)" opacity=".55"/>'
                    f'<text x="{X(limit[0]) + 5:.1f}" y="{top + ph - 7}" font-size="11" fill="var(--ink2)">{esc(limit[1])}</text>')
        for t in nice_ticks(ymin, ymax, 4):
            if ymin - 1e-9 <= t <= ymax + 1e-9:
                svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                        f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{fmt(t, ydec, yunit)}</text>')
        for t in nice_ticks(0, xmax, 4):
            if t <= xmax + 1e-9:
                svg += f'<text x="{X(t):.1f}" y="{top + ph + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">{fmt(t, xdec, xunit)}</text>'
        svg += (f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="var(--base)" stroke-width="1"/>'
                f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + ph}" stroke="var(--base)" stroke-width="1"/>'
                f'<text x="{left + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{esc(xlab)}</text>')
        for q in p["points"]:
            svg += (f'<circle cx="{X(q["x"]):.1f}" cy="{Y(q["y"]):.1f}" r="5.5" fill="var({q.get("var", "--s1")})" stroke="var(--surface)" stroke-width="2">'
                    f'<title>{esc(q["label"])}: {fmt(q["y"], ydec, yunit)} correct, {fmt(q["x"], xdec, xunit)} wrong</title></circle>'
                    f'<text x="{X(q["x"]) + q.get("dx", 9):.1f}" y="{Y(q["y"]) + q.get("dy", 4):.1f}" text-anchor="{q.get("anchor", "start")}" font-size="11.5" fill="var(--ink)">{esc(q["label"])}</text>')
        out.append(f'<div><div class="ttl">{esc(p["title"])}</div><div class="note">{esc(p.get("note", ""))}</div>'
                   f'<svg style="max-width:420px" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(p["title"])}">{svg}</svg></div>')
    return figure(f'<div class="note" style="margin-bottom:.4em">{esc(ylab)}</div><div class="lfa-multi">{"".join(out)}</div>', caption, label)


def dot_rows(rows, series, caption, lo=0.0, hi=1.0, dec=3, unit="", xlab="", label="", ci=False, marks_legend=()):
    """One row per item, one marker per series on a shared axis; optional intervals and per-row reference marks.
    rows: [{"label", "values": [v or None per series], "lo": [...], "hi": [...], "marks": [(value, "pass"|"base", text)]}]
    series: [(name, css var, shape)] with shape "dot" | "diamond" | "ring" (a shape for a fourth series, never a 4th hue)."""
    W, lab, right, rh = 660, 190, 24, 30
    pw = W - lab - right

    def X(v):
        return lab + (v - lo) / (hi - lo) * pw
    H = 22 + len(rows) * rh + 28
    svg = ""
    for t in nice_ticks(lo, hi, 5):
        if lo - 1e-9 <= t <= hi + 1e-9:
            svg += (f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="14" y2="{H - 26}" stroke="var(--grid)" stroke-width="1"/>'
                    f'<text x="{X(t):.1f}" y="{H - 10}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{fmt(t, 1 if hi <= 1 else 0, unit)}</text>')
    if xlab:
        svg += f'<text x="{lab + pw / 2:.1f}" y="9" text-anchor="middle" font-size="11.5" fill="var(--muted)">{esc(xlab)}</text>'

    def marker(x, y, var, shape, tip):
        if shape == "diamond":
            s = 6.5
            return f'<path d="M{x:.1f},{y - s:.1f} L{x + s:.1f},{y:.1f} L{x:.1f},{y + s:.1f} L{x - s:.1f},{y:.1f} Z" fill="var({var})" stroke="var(--surface)" stroke-width="1.5"><title>{esc(tip)}</title></path>'
        if shape == "ring":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="none" stroke="var({var})" stroke-width="2"><title>{esc(tip)}</title></circle>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="var({var})" stroke="var(--surface)" stroke-width="2"><title>{esc(tip)}</title></circle>'
    for i, r in enumerate(rows):
        y = 22 + i * rh + rh / 2
        svg += f'<text x="{lab - 12}" y="{y + 4}" text-anchor="end" font-size="12.5" fill="var(--ink2)">{esc(r["label"])}</text>'
        svg += f'<line x1="{lab}" x2="{lab + pw}" y1="{y}" y2="{y}" stroke="var(--grid)" stroke-width="1"/>'
        for v, kind, text in r.get("marks", ()):
            if kind == "pass":
                svg += f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y - 10}" y2="{y + 10}" stroke="var(--ink)" stroke-width="2"><title>{esc(text)}</title></line>'
            else:
                svg += f'<circle cx="{X(v):.1f}" cy="{y}" r="4.5" fill="none" stroke="var(--ink2)" stroke-width="1.6"><title>{esc(text)}</title></circle>'
        for k, (name, var, shape) in enumerate(series):
            v = r["values"][k]
            if v is None:
                continue
            if ci and r.get("lo"):
                svg += f'<line x1="{X(r["lo"][k]):.1f}" x2="{X(r["hi"][k]):.1f}" y1="{y}" y2="{y}" stroke="var({var})" stroke-width="2.5" stroke-linecap="round"/>'
            svg += marker(X(v), y, var, shape, f'{r["label"]} · {name}: {fmt(v, dec, unit)}')
    leg_items = "".join(
        f'<span><svg width="14" height="14" style="display:inline;width:14px;vertical-align:-2px;margin-right:.35em">'
        + (f'<path d="M7,1 L13,7 L7,13 L1,7 Z" fill="var({v})"/>' if sh == "diamond" else
           f'<circle cx="7" cy="7" r="4.5" fill="var(--surface)" stroke="var({v})" stroke-width="2.2"/>' if sh == "ring" else
           f'<circle cx="7" cy="7" r="5" fill="var({v})"/>') + f'</svg>{esc(n)}</span>' for n, v, sh in series)
    leg_items += "".join(
        f'<span><svg width="14" height="14" style="display:inline;width:14px;vertical-align:-2px;margin-right:.35em">'
        + ('<line x1="7" x2="7" y1="1" y2="13" stroke="var(--ink)" stroke-width="2"/>' if kind == "pass" else
           '<circle cx="7" cy="7" r="4.5" fill="none" stroke="var(--ink2)" stroke-width="1.6"/>') + f'</svg>{esc(text)}</span>'
        for kind, text in marks_legend)
    return figure(f'<div class="leg">{leg_items}</div><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def strip(groups, caption, cut=None, cut_label="", xlab="", label="", lo=0.0, hi=1.0, cuts=(), tick_dec=1, tick_fmt=None):
    """Every item as a dot along x, one row per group, stacked where dots would overlap.
    groups: [{"label", "var", "points": [(x, highlight bool, tip)]}]."""
    W, lab, right, r = 660, 120, 20, 3.6
    pw = W - lab - right

    def X(v):
        return lab + (v - lo) / (hi - lo) * pw
    svg, y = "", 26
    rows_svg = []
    for g in groups:
        bins = {}
        pts = []
        for x, hot, tip in sorted(g["points"], key=lambda p: p[0]):
            b = round(X(x) / (2 * r + .6))
            k = bins.get(b, 0)
            bins[b] = k + 1
            pts.append((X(x), k, hot, tip))
        depth = max(bins.values()) if bins else 1
        band = max(28, depth * (2 * r + .8) + 10)
        base = y + band - 8
        s = f'<text x="{lab - 12}" y="{base - 2:.1f}" text-anchor="end" font-size="12.5" fill="var(--ink2)">{esc(g["label"])}</text>'
        s += f'<line x1="{lab}" x2="{lab + pw}" y1="{base + r + 2:.1f}" y2="{base + r + 2:.1f}" stroke="var(--grid)" stroke-width="1"/>'
        for x, k, hot, tip in pts:
            cy = base - k * (2 * r + .8)
            fill = "var(--ink)" if hot else f"var({g['var']})"
            s += f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r + (0.8 if hot else 0)}" fill="{fill}" stroke="var(--surface)" stroke-width="1"/>'   # no per-dot tooltip: Hugo counts it as reading words
        rows_svg.append(s)
        y += band + 6
    H = y + 24
    svg += "".join(rows_svg)
    for t in nice_ticks(lo, hi, 5):
        if t <= hi + 1e-9:
            svg += f'<text x="{X(t):.1f}" y="{H - 8}" text-anchor="middle" font-size="11.5" fill="var(--muted)">{tick_fmt(t) if tick_fmt else fmt(t, tick_dec)}</text>'
    for cv, cl in ([(cut, cut_label)] if cut is not None else []) + list(cuts):
        svg += (f'<line x1="{X(cv):.1f}" x2="{X(cv):.1f}" y1="12" y2="{H - 22}" stroke="var(--ink)" stroke-width="1.2" opacity=".8"/>'
                f'<text x="{X(cv) - 4:.1f}" y="10" text-anchor="end" font-size="11" fill="var(--ink)">{esc(cl)}</text>')
    if xlab:
        svg += f'<text x="{lab}" y="14" font-size="11.5" fill="var(--muted)">{esc(xlab)}</text>'
    return figure(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)



def log_dot_range(rows, caption, ref=None, ref_label="", xlab="", label=""):
    """Mean as a dot and a line to the maximum, on a log x axis. rows: [{"label", "mean", "max", "var", "note"}]."""
    import math as _m
    lo = 10 ** _m.floor(_m.log10(min(r["mean"] for r in rows)))
    hi = 10 ** _m.ceil(_m.log10(max(r["max"] for r in rows)))
    W, lab, right, rh = 660, 230, 118, 36
    pw = W - lab - right

    def X(v):
        return lab + (_m.log10(v) - _m.log10(lo)) / (_m.log10(hi) - _m.log10(lo)) * pw
    H = 24 + len(rows) * rh + 46
    svg = ""
    e = int(_m.log10(lo))
    while 10 ** e <= hi * 1.0001:
        x = X(10 ** e)
        svg += (f'<line x1="{x:.1f}" x2="{x:.1f}" y1="16" y2="{H - 44}" stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{x:.1f}" y="{H - 30}" text-anchor="middle" font-size="11" fill="var(--muted)">1e{e}</text>')
        e += 2 if (_m.log10(hi) - _m.log10(lo)) > 8 else 1
    if ref is not None:
        svg += (f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="10" y2="{H - 44}" stroke="var(--ink)" stroke-width="1.4"/>'
                f'<text x="{X(ref) + 5:.1f}" y="14" font-size="11.5" fill="var(--ink)">{esc(ref_label)}</text>')
    for i, r in enumerate(rows):
        y = 26 + i * rh + rh / 2
        v = f'var({r.get("var", "--s1")})'
        svg += (f'<text x="{lab - 12}" y="{y + 4}" text-anchor="end" font-size="12.5" fill="var(--ink2)">{esc(r["label"])}</text>'
                f'<g><title>{esc(r["label"])}: mean {r["mean"]:.3g}, max {r["max"]:.3g} nats</title>'
                f'<line x1="{X(r["mean"]):.1f}" x2="{X(r["max"]):.1f}" y1="{y}" y2="{y}" stroke="{v}" stroke-width="2.5" stroke-linecap="round"/>'
                f'<circle cx="{X(r["mean"]):.1f}" cy="{y}" r="5.5" fill="{v}" stroke="var(--surface)" stroke-width="2"/></g>')
        if r.get("note"):
            svg += f'<text x="{W - right + 10:.1f}" y="{y + 4}" font-size="11.5" fill="var(--muted)">{esc(r["note"])}</text>'
    if xlab:
        svg += f'<text x="{lab + pw / 2:.1f}" y="{H - 10}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{esc(xlab)}</text>'
    return figure(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def range_vs_point(items, caption, budget=None, budget_label="", ylab="", label="", unit=" h"):
    """Per category: a registered range (low to high, tick at mid) and a measured or projected dot, against a budget line.
    items: [(label, low, mid, high, point)]."""
    top = max(max(i[3], i[4]) for i in items) * 1.08
    W, H, left, right, tp, bottom = 660, 300, 48, 12, 24, 36
    pw, ph = W - left - right, H - tp - bottom
    cw = pw / len(items)

    def Y(v):
        return tp + ph * (1 - v / top)
    svg = ""
    for t in nice_ticks(0, top, 4):
        if t <= top:
            svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                    f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{fmt(t, 0)}{unit}</text>')
    if budget is not None:
        svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(budget):.1f}" y2="{Y(budget):.1f}" stroke="var(--ink)" stroke-width="1.4"/>'
                f'<text x="{left + pw - 4}" y="{Y(budget) - 6:.1f}" text-anchor="end" font-size="11.5" fill="var(--ink)">{esc(budget_label)}</text>')
    for k, (lab, lo_, mid, hi_, pt) in enumerate(items):
        x = left + k * cw + cw / 2
        svg += (f'<g><title>{esc(lab)}: registered {lo_}–{hi_}{unit} (pessimistic {mid}); projected {pt:.1f}{unit}</title>'
                f'<line x1="{x - 7:.1f}" x2="{x - 7:.1f}" y1="{Y(lo_):.1f}" y2="{Y(hi_):.1f}" stroke="var(--s1-mute)" stroke-width="6" stroke-linecap="round"/>'
                f'<line x1="{x - 12:.1f}" x2="{x - 2:.1f}" y1="{Y(mid):.1f}" y2="{Y(mid):.1f}" stroke="var(--s1)" stroke-width="2"/>'
                f'<circle cx="{x + 7:.1f}" cy="{Y(pt):.1f}" r="5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/></g>'
                f'<text x="{x:.1f}" y="{tp + ph + 16}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{esc(lab)}</text>')
    svg += f'<line x1="{left}" x2="{left + pw}" y1="{tp + ph}" y2="{tp + ph}" stroke="var(--base)" stroke-width="1"/>'
    if ylab:
        svg += f'<text x="{left}" y="12" font-size="11.5" fill="var(--ink2)">{esc(ylab)}</text>'
    leg = ('<div class="leg"><span><svg width="14" height="14" style="display:inline;width:14px;vertical-align:-2px;margin-right:.35em">'
           '<line x1="7" x2="7" y1="1" y2="13" stroke="var(--s1-mute)" stroke-width="6" stroke-linecap="round"/><line x1="2" x2="12" y1="7" y2="7" stroke="var(--s1)" stroke-width="2"/></svg>'
           'registered estimate: nominal to adverse, tick at pessimistic</span>'
           '<span><i style="background:var(--s2);border-radius:50%"></i>projected from the pilot</span></div>')
    return figure(f'{leg}<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)


def cat_dots(cats, caption, ylab="", ymin=0, ymax=8, label="", marks_legend=()):
    """Per category on x, each repetition as a dot on y, spread sideways so none hide; an optional tick for a summary.
    cats: [{"label", "sub", "groups": [(values, css var, filled bool)], "tick": value or None}]."""
    W, H, left, right, tp, bottom = 660, 310, 44, 10, 28, 52
    pw, ph = W - left - right, H - tp - bottom
    cw = pw / len(cats)

    def Y(v):
        return tp + ph * (1 - (v - ymin) / (ymax - ymin))
    svg = ""
    for t in range(int(ymin), int(ymax) + 1, 2 if ymax - ymin > 6 else 1):
        svg += (f'<line x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{t}</text>')
    for k, c in enumerate(cats):
        cx = left + k * cw + cw / 2
        off = -((len(c["groups"]) - 1) * 26) / 2
        for gi, (vals, var, filled) in enumerate(c["groups"]):
            gx = cx + off + gi * 26
            seen = {}
            for v in vals:
                n = seen.get(v, 0)
                seen[v] = n + 1
                x = gx + (n - (vals.count(v) - 1) / 2) * 9
                svg += (f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="4.2" fill="{f"var({var})" if filled else "var(--surface)"}" '
                        f'stroke="var({var})" stroke-width="{2 if not filled else 1}"><title>{esc(c["label"])}: {v}</title></circle>')
        if c.get("tick") is not None:
            svg += f'<line x1="{cx - 30:.1f}" x2="{cx + 30:.1f}" y1="{Y(c["tick"]):.1f}" y2="{Y(c["tick"]):.1f}" stroke="var(--ink)" stroke-width="2"/>'
        svg += (f'<text x="{cx:.1f}" y="{tp + ph + 18}" text-anchor="middle" font-size="12" fill="var(--ink2)">{esc(c["label"])}</text>'
                f'<text x="{cx:.1f}" y="{tp + ph + 34}" text-anchor="middle" font-size="11" fill="var(--muted)">{esc(c.get("sub", ""))}</text>')
    svg += f'<line x1="{left}" x2="{left + pw}" y1="{Y(ymin):.1f}" y2="{Y(ymin):.1f}" stroke="var(--base)" stroke-width="1"/>'
    if ylab:
        svg += f'<text x="{left}" y="11" font-size="11.5" fill="var(--ink2)">{esc(ylab)}</text>'
    leg = "".join(
        f'<span><svg width="14" height="14" style="display:inline;width:14px;vertical-align:-2px;margin-right:.35em">'
        + (f'<line x1="1" x2="13" y1="7" y2="7" stroke="var(--ink)" stroke-width="2"/>' if kind == "tick" else
           f'<circle cx="7" cy="7" r="4.2" fill="{"var(" + var + ")" if kind == "filled" else "var(--surface)"}" stroke="var({var})" stroke-width="2"/>')
        + f'</svg>{esc(text)}</span>' for kind, var, text in marks_legend)
    return figure(f'<div class="leg">{leg}</div><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label or caption)}">{svg}</svg>', caption, label)
