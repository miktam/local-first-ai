"""exp_036 key-numbers table and plain-answer sentences (Tier 2).

HYPOTHESIS "Publication angle" (one key-numbers table, ordered Q1 -> Q5: vendor,
ours, a V/R/untestable label, the truncation rate and the result excluding
truncated items) and "Plain answers" (the sentence chosen by the verdicts, with
the <...> placeholders filled). The sentence *selection* is Tier 1
(verdicts.plain_answer_states); this module only renders it.

render_markdown(verdicts) -> str
plain_answer_sentences(verdicts) -> {"Q1": [str, ...], ..., "Q5": [...]}
key_numbers_rows(verdicts) -> list of row dicts
"""

from __future__ import annotations

from analysis import verdicts as V


def pp(x, sign=True) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:+.1f}" if sign else f"{100.0 * x:.1f}"


def ci_pp(ci) -> str:
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{100.0 * ci[0]:+.1f}, {100.0 * ci[1]:+.1f}]"


def num(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci_num(ci, nd=2) -> str:
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def fmt_trunc(tr) -> str:
    """Truncation rates {arm: rate} or {arm: {row: rate}} as compact percentages."""
    if tr is None:
        return ""
    parts = []
    for arm, x in tr.items():
        if isinstance(x, dict):
            parts.append(f"{arm} " + "/".join(pp(y, False) for y in x.values()) + " %")
        else:
            parts.append(f"{arm} {pp(x, False)} %")
    return "; ".join(parts)


def _d(v, h):
    return v["verdicts"].get(f"{h}_detail") or {}


def _tps(v, arm):
    desc = ((_d(v, "H1").get("detail") or {}).get("descriptive") or {}).get("arms", {})
    b1 = (desc.get(arm) or {}).get("batch_1") or {}
    if b1.get("generation_tps") is not None:
        return b1["generation_tps"]
    tps = (_d(v, "H1").get("detail") or {}).get("tps") or {}
    vals = sorted(x[arm] for x in tps.values() if arm in x)
    return vals[len(vals) // 2] if vals else None


# --------------------------------------------------------------------------
# Key numbers
# --------------------------------------------------------------------------

def key_numbers_rows(v: dict) -> list[dict]:
    rows = []
    h1 = _d(v, "H1")
    rows.append({"q": "Q1", "what": "K4 / G4 batch-1 decode ratio (H1)", "vendor": "untestable",
                 "ours": num(h1.get("estimate")), "label": "M5 Max, MLX 0.31.2",
                 "ci": ci_num(h1.get("ci95")), "trunc": "", "excl": "", "verdict": v["verdicts"].get("H1")})
    d1 = _d(v, "D1")
    rows.append({"q": "Q1", "what": "K4 peak memory at 64k, GiB (D1)", "vendor": "untestable",
                 "ours": num(d1.get("median_peak_gib_64k")), "label": "M5 Max, MLX 0.31.2",
                 "ci": "", "trunc": "", "excl": "", "verdict": v["verdicts"].get("D1")})
    h2 = _d(v, "H2")
    for r, row in ((h2.get("detail") or {}).get("rows") or {}).items():
        rows.append({"q": "Q2", "what": f"{row.get('name', r)} (H2 row)",
                     "vendor": f"{pp(row['vendor'], False)} ({row.get('vendor_page')})",
                     "ours": pp(row["ours"], False), "label": row["label"], "ci": ci_pp(row["ci95"]),
                     "trunc": pp(row["truncation_rate"], False),
                     "excl": pp(row.get("D_excluding_truncated")), "verdict": ""})
    rows.append({"q": "Q2", "what": "mean D over rows (H2)", "vendor": "", "ours": pp(h2.get("estimate")),
                 "label": "", "ci": ci_pp(h2.get("ci95")), "trunc": "", "excl": "", "verdict": v["verdicts"].get("H2")})
    for h, what in (("H3", "MMLU-ProX-Lite K − peer mean, EN+DE (H3)"), ("H4", "IFBench K − Qwen3.6 (H4)"),
                    ("H6", "RGB closed-book K − peer mean (H6)"), ("H7", "4-bit task cost K4 − K8 (H7)")):
        d = _d(v, h)
        tr = (d.get("detail") or {}).get("truncation_rate")
        ts = d.get("truncation_sensitivity") or {}
        rows.append({"q": {"H3": "Q3", "H4": "Q3", "H6": "Q4", "H7": "Q5"}[h], "what": what, "vendor": "",
                     "ours": pp(d.get("estimate")), "label": "R", "ci": ci_pp(d.get("ci95")),
                     "trunc": fmt_trunc(tr), "excl": pp(ts.get("estimate")),
                     "verdict": v["verdicts"].get(h)})
    h5 = _d(v, "H5").get("detail") or {}
    rows.append({"q": "Q3", "what": "German bytes/token ratio K vs Gemma 4 / Qwen3.6 (H5)",
                 "vendor": "1.186 / 1.175 (p. 10–11/189)",
                 "ours": f"{num(h5.get('ratio_vs_gemma4'), 3)} / {num(h5.get('ratio_vs_qwen3_6'), 3)}",
                 "label": "", "ci": f"{ci_num(h5.get('ratio_vs_gemma4_ci95'), 3)} / {ci_num(h5.get('ratio_vs_qwen3_6_ci95'), 3)}",
                 "trunc": "", "excl": "", "verdict": v["verdicts"].get("H5")})
    h8 = _d(v, "H8")
    rows.append({"q": "Q5", "what": "KL(8‖4) per byte, K / peer median (H8)", "vendor": "untestable",
                 "ours": num(h8.get("estimate")), "label": "", "ci": ci_num(h8.get("ci95")),
                 "trunc": "", "excl": "", "verdict": v["verdicts"].get("H8")})
    rows.sort(key=lambda r: r["q"])     # Q1 -> Q5, stable within a question
    return rows


def render_key_numbers(v: dict) -> str:
    lines = ["## Key numbers (exploratory rendering; verdicts from the Tier-1 table)", "",
             "| Q | What | Vendor | Ours | V/R | 95 % CI | Truncation rate | Excluding truncated | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in key_numbers_rows(v):
        lines.append(f"| {r['q']} | {r['what']} | {r['vendor']} | {r['ours']} | {r['label']} | {r['ci']} | "
                     f"{r['trunc']} | {r['excl']} | {r['verdict'] or ''} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Plain answers
# --------------------------------------------------------------------------

def _fill(template: str, values: dict) -> str:
    out = template
    for k, val in values.items():
        out = out.replace(f"<{k}>", val)
    return out


def _not_run_sentence(state: dict) -> str:
    return _fill(V.PLAIN_ANSWERS["not_run"], {"H": state["H"], "reason": str(state.get("reason"))})


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _precision(v) -> str:
    """"8-bit" or "4-bit": the arm H2 ran on (K4 when K8 cannot run; HYPOTHESIS rule 6, "titled at MLX 4-bit")."""
    return "4-bit" if v["config"].get("kolibri_primary") == "K4" else "8-bit"


def _adapt(sentence: str, v: dict) -> str:
    """The fixed words of the pre-registered sentences that the pre-registration itself makes conditional: the
    precision (K4 primary), the number of H2 rows (4 under P10) and of reconstructed rows, and the H2 margin
    (Amendment 0 may set -3 or -5 pp). With the pre-registered defaults (K8, five rows, three reconstructed,
    -4 pp) the sentence is returned unchanged (review fix 2026-10-03)."""
    h2 = (v["verdicts"].get("H2_detail") or {}).get("detail") or {}
    n_rows = h2.get("n_rows")
    n_rec = len(h2.get("reconstructed_rows") or []) if h2.get("reconstructed_rows") is not None else None
    margin = (v["config"].get("margins") or {}).get("H2")
    out = sentence.replace("MLX 8-bit", f"MLX {_precision(v)}").replace("Kolibri 8-bit", f"Kolibri {_precision(v)}")
    if n_rows and n_rows in _NUMBER_WORDS and n_rows != 5:
        out = out.replace("the five public rows", f"the {_NUMBER_WORDS[n_rows]} public rows")
        out = out.replace("of the five rows", f"of the {_NUMBER_WORDS[n_rows]} rows")
    if n_rec is not None and n_rec in _NUMBER_WORDS and n_rec != 3:
        out = out.replace("three of the", f"{_NUMBER_WORDS[n_rec]} of the")
    if margin is not None and round(-100.0 * float(margin)) != 4:
        out = out.replace("no more than 4 pp below", f"no more than {round(-100.0 * float(margin)):d} pp below")
    return out


def plain_answer_sentences(v: dict) -> dict:
    pa = v["summary"]["plain_answers"]
    T = V.PLAIN_ANSWERS
    out: dict = {}

    q1 = []
    for st in pa["Q1"]:
        if st["id"] == "not_run":
            q1.append(_not_run_sentence(st))
        elif st["id"] == "Q1_always":
            q1.append(_fill(T["Q1_always"], {"K8 tok/s": num(_tps(v, "K8"), 1), "K4 tok/s": num(_tps(v, "K4"), 1)}))
        elif st["id"] == "Q1_H1":
            h1 = _d(v, "H1")
            q1.append(_fill(T["Q1_H1"], {"r": num(h1.get("estimate")), "H1 phrase": T["Q1_H1_phrases"][st["state"]]})
                      .replace("[CI]", ci_num(h1.get("ci95"))))
        elif st["id"] == "Q1_D1":
            q1.append(_fill(T["Q1_D1"], {"D1 phrase": T["Q1_D1_phrases"][st["state"]]}))
    out["Q1"] = q1

    q2 = pa["Q2"]
    h2 = _d(v, "H2")
    pc = (h2.get("detail") or {}).get("protocol_control") or {}
    if q2["id"] == "not_run":
        out["Q2"] = [_not_run_sentence(q2)]
    elif q2["id"] == "Q2_peers_outside":
        out["Q2"] = [_fill(T["Q2_peers_outside"], {
            "lower/higher": q2["direction"], "x": pp(pc.get("kolibri_mean_D_shared")),
            "y": pp(pc.get("peers_mean_D_shared")), "DiD": pp(pc.get("DiD"))}).replace("[CI]", ci_pp(pc.get("DiD_ci95")))]
    elif q2["id"] == "Q2_peer_control_not_run":
        # No pre-registered sentence applies (verdicts.plain_answer_states); a plain statement, never a headline.
        out["Q2"] = [f"[No pre-registered Q2 sentence applies: H2 is {q2.get('state')}, but the peer protocol control "
                     f"could not be computed ({q2.get('reason')}); D̄ = {pp(h2.get('estimate'))} {ci_pp(h2.get('ci95'))}.]"]
    elif q2["id"] == "Q2_confirmed":
        s = _fill(_adapt(T["Q2_confirmed"], v), {"x": pp(h2.get("estimate"))}).replace("[CI]", ci_pp(h2.get("ci95")))
        if q2.get("named_rows"):
            s += " Rows with a 95 % upper bound below −8 pp: " + ", ".join(q2["named_rows"]) + "."
        out["Q2"] = [s]
    elif q2["id"] == "Q2_refuted":
        out["Q2"] = [_fill(_adapt(T["Q2_refuted"], v), {"x": pp(-(h2.get("estimate") or 0.0), False),
                                             "y": pp(abs(pc.get("peers_mean_D_shared") or 0.0), False)})]
    else:
        out["Q2"] = [_fill(T["Q2_inconclusive"], {"x": pp(h2.get("estimate"))}).replace("[CI]", ci_pp(h2.get("ci95")))]

    q3 = pa["Q3"]
    h3, h4, h5 = _d(v, "H3"), _d(v, "H4"), _d(v, "H5")
    gaps = (h3.get("detail") or {}).get("per_language_gap") or {}
    vs_g = (h4.get("detail") or {}).get("kolibri_vs_gemma4") or {}
    h5d = h5.get("detail") or {}
    ratio = None
    if h5d.get("ratio_vs_gemma4") is not None:
        ratio = f"{h5d['ratio_vs_gemma4']:.3f} / {h5d['ratio_vs_qwen3_6']:.3f}"
    s = _fill(_adapt(T["Q3_always"], v), {
        "x": pp(h3.get("estimate")), "a": pp(gaps.get("en")), "b": pp(gaps.get("de")),
        "H3 verdict": q3["verdicts"]["H3"], "y": pp(h4.get("estimate")), "H4 verdict": q3["verdicts"]["H4"],
        "z": pp(vs_g.get("D")), "r": ratio or "n/a", "H5 verdict": q3["verdicts"]["H5"],
        "MMLU-ProX DE / plus GPQA-D DE and AIME DE": (
            q3["german_basis"][0] + (" plus " + " and ".join(q3["german_basis"][1:]) if len(q3["german_basis"]) > 1 else "")),
    }).replace("[CI]", ci_pp(h3.get("ci95")), 1)
    q3_out = [s]
    if q3["e1_label_applies"]:
        q3_out.append(f"The public-only composite {T['Q3_e1_label']}.")
    q3_out += [_not_run_sentence(st) for st in q3.get("not_run", [])]
    out["Q3"] = q3_out

    q4 = pa["Q4"]
    h6 = _d(v, "H6")
    e2 = v["verdicts"].get("E2") or {}
    forced = (h6.get("detail") or {}).get("forced") or {}
    A = e2.get("A")
    if q4["id"] == "not_run":
        out["Q4"] = [_not_run_sentence(q4)]
    elif q4["id"] == "Q4_knows_less":
        out["Q4"] = [_fill(T["Q4_knows_less"], {
            "x": pp(-(h6.get("estimate") or 0.0), False), "y": pp(-(forced.get("deficit") or 0.0), False),
            "A": pp(A, False) + " %" if A is not None else "n/a",
            "says so / guesses / unclear": e2.get("reading", "unclear")})]
    elif q4["id"] == "Q4_confirmed_rate":
        out["Q4"] = [_fill(T["Q4_confirmed_rate"], {
            "x": pp(-(h6.get("estimate") or 0.0), False), "A": pp(A, False) + " %" if A is not None else "n/a",
            "E2 reading": e2.get("reading", "unclear")})]
    elif q4["id"] == "Q4_refuted":
        out["Q4"] = [_fill(T["Q4_refuted"], {"x": pp(h6.get("estimate"))}).replace("[CI]", ci_pp(h6.get("ci95")))]
    else:
        out["Q4"] = [_fill(T["Q4_inconclusive"], {"x": pp(h6.get("estimate"))}).replace("[CI]", ci_pp(h6.get("ci95")))]

    q5 = pa["Q5"]
    if q5["id"] == "Q5_not_run":
        out["Q5"] = [_fill(T["Q5_not_run"], {"reason": q5.get("reason", "")})]
    else:
        h7, h8 = _d(v, "H7"), _d(v, "H8")
        s = _fill(T["Q5_both"], {"x": pp(h7.get("estimate")), "H7 verdict": q5["verdicts"]["H7"],
                                 "r": num(h8.get("estimate")), "H8 verdict": q5["verdicts"]["H8"]})
        out["Q5"] = [s] + ([q5["per_token_note"]] if q5.get("per_token_note") else [])
    return out


def render_markdown(v: dict) -> str:
    parts = [render_key_numbers(v), "## Plain answers (pre-registered wording, filled)", ""]
    for q, sents in plain_answer_sentences(v).items():
        parts.append(f"**{q}.** " + " ".join(sents))
        parts.append("")
    hl = [h for h in v["config"]["family"] if not (v["verdicts"].get(f"{h}_detail") or {}).get("headline_eligible")]
    if hl:
        parts.append("Not headline-eligible (E8 flag, truncation-sensitive or NOT RUN): " + ", ".join(hl) + ".")
    return "\n".join(parts) + "\n"
