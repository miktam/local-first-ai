"""exp_045: lead triage from voice notes.

Each enquiry of exp_044's sets (the public copy: contact numbers in Ofcom's drama range; Kev's masked input and the
code facts are identical to the sealed originals) is read aloud by a macOS voice in its language, encoded like a
WhatsApp voice note (Opus 16 kbit/s, mono, 16 kHz), transcribed on this machine by the bot's own path
(notetaker.to_wav + notetaker.transcribe: mlx-whisper large-v3-turbo, Whisper picks the language), and triaged by the
frozen rev 4 exactly as the bot's voice handler does: Whisper's language replaces the text detector's when it is one
of the seven, and phone numbers read out in groups are joined first (notetaker.join_spoken_numbers, as in the bot).
The "noisy" arm mixes seeded pink noise at 10 dB SNR into the speech before encoding (descriptive).

The "text" arm is rev 4 on the same public-copy text, so both arms read identical words. Kev is deterministic, so it
should repeat exp_044's recorded decisions except where tonight's public-copy edits changed Kev's input (14 messages
spelled their phone number out in words); score reports that check.

usage: voice45.py run dev|test [--arm text|whatsapp|noisy] [--max-seconds N]    (resumes; the test pass runs once per arm)
       voice45.py score dev|test
"""
import difflib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")             # the cached Whisper model only, as in the bot

from scripts import leads44 as L  # noqa: E402

D = ROOT / "data" / "leads45"
RUNS = ROOT / "runs"
ARMS = ("text", "whatsapp", "noisy")
# One voice per writer batch (X1 and X2 are both English; X1 is the deliberately messy one).
VOICES = {"X1": "Samantha", "X2": "Daniel", "X3": "Zosia", "X4": "Mónica", "X5": "Anna", "X6": "Alva",
          "X7": "Thomas", "X8": "Milena"}
OPUS = ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "16k", "-application", "voip"]   # WhatsApp-like
SNR_DB = 10.0
SETS = {"dev": "65ef5ac9da625b1eb0b16e83adb171b85102e3ff8c8c923b167625c10caf18b5",       # public copy, local-first-ai 9d87aa7
        "test": "514b6386dfba45464089a472d579097df58c3380964c1203aab9a39deddae808"}


def speakable(text: str) -> str:
    """What a person would say: emoji and other pictographs dropped (a voice note has none), whitespace kept."""
    s = "".join(c for c in text if unicodedata.category(c) != "So" and c not in "\u200d\ufe0f"
                and not 0x1F3FB <= ord(c) <= 0x1F3FF)                          # emoji skin tones are "Sk"
    return re.sub(r"[ \t]+", " ", s).strip()


def _ffmpeg(*args, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], input=stdin, capture_output=True,
                          check=True).stdout


def pink_noise(n: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    f[0] = 1
    noise = np.fft.irfft(spec / np.sqrt(f), n)
    return noise / (np.sqrt(np.mean(noise ** 2)) or 1.0)


def voice_note(row: dict, arm: str, out: Path) -> float:
    """Synthesise the enquiry and encode it like a WhatsApp voice note at `out` (.ogg). Returns the speech seconds."""
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        txt, aiff = Path(tmp) / "t.txt", Path(tmp) / "t.aiff"
        txt.write_text(speakable(row["text"]), encoding="utf-8")
        subprocess.run(["say", "-v", VOICES[row["id"].split("-")[0]], "-f", str(txt), "-o", str(aiff)], check=True)
        pcm = np.frombuffer(_ffmpeg("-i", str(aiff), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"), dtype=np.int16)
        x = pcm.astype(np.float64)
        if arm == "noisy":
            rms = np.sqrt(np.mean(x ** 2)) or 1.0
            x = x + pink_noise(len(x), zlib.crc32(row["id"].encode())) * rms / 10 ** (SNR_DB / 20)
            x = x * min(1.0, 32000 / (np.max(np.abs(x)) or 1.0))
        _ffmpeg("-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-", *OPUS, str(out),
                stdin=x.astype(np.int16).tobytes())
        return round(len(pcm) / 16000, 2)


EMPTY = {"intent": {"choice": None, "confidence": 0.0, "probabilities": {}},
         "property_type": {"choice": None, "confidence": 0.0, "probabilities": {}},
         "timeline": {"choice": None, "confidence": 0.0, "probabilities": {}},
         "financing": 0.0, "foreign_buyer": 0.0, "qualified": 0.0, "urgency": 0, "action": None}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cmd_run(set_: str, arm: str, max_s: float):
    from scripts import notetaker as nt
    from scripts.telegram_bot import _TRIAGE_LANG
    setp = D / f"{set_}.jsonl"
    if hashlib.sha256(setp.read_bytes()).hexdigest() != SETS[set_]:
        sys.exit(f"REFUSED: {setp.name} is not the registered public copy")
    path, frozen = L.VERSIONS["rev4"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
        sys.exit("REFUSED: kev_knife.py differs from the frozen rev 4")
    out = RUNS / f"exp045_{set_}_{arm}"
    if set_ == "test":
        if (out / "done.json").exists():
            sys.exit(f"REFUSED: the test set has had its one pass for the {arm} arm")
        if _git("status", "--porcelain", "scripts/voice45.py", "scripts/notetaker.py"):
            sys.exit("REFUSED: commit voice45.py / notetaker.py before the test pass")
    K = L.knife("rev4")
    rows = L.load(setp)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    pred = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    t_start = time.time()
    with pred.open("a", encoding="utf-8") as f:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            if max_s and time.time() - t_start > max_s:
                print(f"stopped after {max_s:.0f} s; run again to resume", flush=True)
                return
            if arm == "text":
                f.write(json.dumps({"id": r["id"], **K.triage(r["text"])}, ensure_ascii=False) + "\n")
                f.flush()
                continue
            ogg = out / "audio" / f"{r['id']}.ogg"
            t0 = time.time()
            speech_s = voice_note(r, arm, ogg)
            t1 = time.time()
            wav = nt.to_wav(ogg)
            try:
                heard_text, _readable, heard = nt.transcribe(wav)
            finally:
                wav.unlink(missing_ok=True)
            t2 = time.time()
            raw_text = heard_text.strip()
            heard_text = nt.join_spoken_numbers(raw_text)
            if heard_text:
                p = K.triage(heard_text)
                if heard in _TRIAGE_LANG:                  # the bot's voice handler does exactly this
                    p.setdefault("facts", {})["language"] = heard
            else:                                          # the bot answers "I couldn't hear any words": no triage
                p = {**json.loads(json.dumps(EMPTY)), "facts": {"language": None, "budget_eur": None,
                                                                  "has_phone": False, "has_email": False}}
            f.write(json.dumps({"id": r["id"], **p, "transcript": heard_text, "transcript_raw": raw_text, "heard": heard,
                                "speech_s": speech_s,
                                "tts_ms": round((t1 - t0) * 1000), "stt_ms": round((t2 - t1) * 1000),
                                "kev_ms": round((time.time() - t2) * 1000)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} {arm} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        (out / "done.json").write_text(json.dumps({
            "n": len(rows), "arm": arm, "set_sha256": SETS[set_], "kev_knife_sha256": frozen,
            "voice45_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "casasol_head": _git("rev-parse", "HEAD"),
            "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(), "whisper": nt.WHISPER_MODEL,
            "voices": VOICES if arm != "text" else None, "opus": OPUS if arm != "text" else None,
            "snr_db": SNR_DB if arm == "noisy" else None,
            "macos": platform.mac_ver()[0], "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                                                     text=True).stdout.split("\n")[0]}) + "\n")
    print("done")


def mcnemar_two_sided(b: int, c: int) -> float:
    return min(1.0, 2 * L.mcnemar_one_sided(max(b, c), min(b, c))) if b + c else 1.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def arm_report(rows: dict, text: dict, voice: dict) -> dict:
    ids = [i for i in rows if i in voice and i in text]
    out = {"n": len(ids), "empty_transcripts": sum(not voice[i]["transcript"] for i in ids)}
    for field in ("intent", "action"):
        rt = {i: L.correct(field, rows[i]["labels"][field], L.predicted(text[i])[field]) for i in ids}
        rv = {i: L.correct(field, rows[i]["labels"][field], L.predicted(voice[i])[field]) for i in ids}
        b, c = sum(rv[i] and not rt[i] for i in ids), sum(rt[i] and not rv[i] for i in ids)
        out[field] = {"text": L.acc(list(rt.values())), "voice": L.acc(list(rv.values())),
                      "voice_only_right": b, "text_only_right": c, "p_two_sided": round(mcnemar_two_sided(b, c), 4),
                      "same_decision": L.acc([L.predicted(text[i])[field] == L.predicted(voice[i])[field] for i in ids])}
        out[field]["diff"] = round(out[field]["voice"] - out[field]["text"], 3)
    st, sv = L.spam_stats(rows, text, ids), L.spam_stats(rows, voice, ids)
    out["spam"] = {k: {key: s[key] for key in ("caught", "spam_n", "recall", "false_spam", "genuine_n")} | {"false_spam_ids": s["false_spam_ids"]}
                   for k, s in (("text", st), ("voice", sv))}
    stated = [i for i in ids if rows[i]["labels"]["budget_eur"] not in (None, "disagree")]
    words = [i for i in stated if rows[i]["stratum"] == "hard_budget"]
    out["budget"] = {grp: {"n": len(g), "text": L.acc(L.scored(rows, text, "budget_eur", g)),
                           "voice": L.acc(L.scored(rows, voice, "budget_eur", g))}
                     for grp, g in (("stated", stated), ("in_words", words))}
    abbrev = re.compile(r"\d\s?(?:k|K|M€|M|Mio|mln|m€)(?![a-zà-ÿ])")
    lost = [i for i in stated if L.budget_agree(L.predicted(text[i])["budget_eur"], rows[i]["labels"]["budget_eur"])
            and not L.budget_agree(L.predicted(voice[i])["budget_eur"], rows[i]["labels"]["budget_eur"])]
    out["budget"]["lost_in_voice"] = {"n": len(lost), "abbreviation_in_text": sorted(i for i in lost if abbrev.search(rows[i]["text"])),
                                      "other": sorted(i for i in lost if not abbrev.search(rows[i]["text"]))}
    out["language"] = {"whisper_heard_right": L.acc([voice[i]["heard"] == rows[i]["lang"] for i in ids]),
                       "shown_right_voice": L.acc([(voice[i].get("facts") or {}).get("language") == rows[i]["lang"] for i in ids]),
                       "shown_right_text": L.acc([(text[i].get("facts") or {}).get("language") == rows[i]["lang"] for i in ids]),
                       "heard_wrong": dict(Counter(f"{rows[i]['lang']}→{voice[i]['heard']}" for i in ids
                                                   if voice[i]["heard"] != rows[i]["lang"]))}
    out["by_language_intent"] = {lang: {"n": len(g), "text": L.acc([rows[i]["labels"]["intent"] == text[i]["intent"]["choice"] for i in g]),
                                        "voice": L.acc([rows[i]["labels"]["intent"] == voice[i]["intent"]["choice"] for i in g]),
                                        "transcript_similarity_median": sorted(
                                            difflib.SequenceMatcher(None, _norm(rows[i]["text"]), _norm(voice[i]["transcript"])).ratio()
                                            for i in g)[len(g) // 2] if g else None}
                                 for lang in sorted({rows[i]["lang"] for i in ids})
                                 for g in [[i for i in ids if rows[i]["lang"] == lang]]}
    out["by_stratum_intent"] = {s: {"text": L.acc(L.scored(rows, text, "intent", g)), "voice": L.acc(L.scored(rows, voice, "intent", g))}
                                for s in L.STRATA for g in [[i for i in ids if rows[i]["stratum"] == s]]}
    out["phone_seen"] = {"text": sum((text[i].get("facts") or {}).get("has_phone", False) for i in ids),
                         "voice": sum((voice[i].get("facts") or {}).get("has_phone", False) for i in ids)}
    med = lambda k: sorted(voice[i][k] for i in ids)[len(ids) // 2] if ids else None
    out["median"] = {"speech_s": med("speech_s"), "stt_ms": med("stt_ms"), "kev_ms": med("kev_ms")}
    out["flips_intent"] = [f"{i}: {rows[i]['labels']['intent']} | text {text[i]['intent']['choice']} | voice {voice[i]['intent']['choice']}"
                           for i in ids if text[i]["intent"]["choice"] != voice[i]["intent"]["choice"]][:40]
    return out


def cmd_score(set_: str):
    rows = {r["id"]: r for r in L.load(D / f"{set_}.jsonl")}
    tp = RUNS / f"exp045_{set_}_text" / "pred.jsonl"
    if not tp.exists():
        sys.exit(f"run the text arm first: voice45.py run {set_} --arm text")
    text = {p["id"]: p for p in L.load(tp)}
    rec = {p["id"]: p for p in L.load(RUNS / f"exp044_{set_}_rev4" / "pred.jsonl")}
    differ = sorted(i for i in text if i in rec and (text[i]["intent"]["choice"], text[i]["action"])
                    != (rec[i]["intent"]["choice"], rec[i]["action"]))
    res = {"set": set_, "set_sha256": SETS[set_],
           "text_vs_exp044_recorded": {"n": len(text), "same_request_and_next_step": len(text) - len(differ), "differ": differ}}
    for arm in ARMS[1:]:
        pred = RUNS / f"exp045_{set_}_{arm}" / "pred.jsonl"
        if pred.exists():
            res[arm] = arm_report(rows, text, {p["id"]: p for p in L.load(pred)})
    w = res.get("whatsapp")
    if w and w["n"] == len(rows):
        res["hypotheses_whatsapp"] = {
            "H1_request": {"diff": w["intent"]["diff"], "pass": w["intent"]["diff"] >= -0.03},
            "H2_next_step": {"diff": w["action"]["diff"], "pass": w["action"]["diff"] >= -0.03},
            "H3_spam": {"false_spam": [w["spam"]["text"]["false_spam"], w["spam"]["voice"]["false_spam"]],
                        "recall_voice": w["spam"]["voice"]["recall"],
                        "pass": w["spam"]["voice"]["false_spam"] <= w["spam"]["text"]["false_spam"] + 1
                                and (w["spam"]["voice"]["recall"] or 0) >= 0.85},
            "H4_budget": {"stated": w["budget"]["stated"], "in_words": w["budget"]["in_words"],
                          "pass": (w["budget"]["stated"]["voice"] or 0) >= (w["budget"]["stated"]["text"] or 0) - 0.05
                                  and (w["budget"]["in_words"]["voice"] or 0) >= 0.90},
            "H5_language_shown": {"voice": w["language"]["shown_right_voice"], "pass": (w["language"]["shown_right_voice"] or 0) >= 0.95},
        }
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("run", "score") or sys.argv[2] not in SETS:
        sys.exit(__doc__)
    cmd, set_, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if cmd == "score":
        cmd_score(set_)
    else:
        arm = rest[rest.index("--arm") + 1] if "--arm" in rest else "whatsapp"
        if arm not in ARMS:
            sys.exit(f"arm must be one of {ARMS}")
        cmd_run(set_, arm, float(rest[rest.index("--max-seconds") + 1]) if "--max-seconds" in rest else 0.0)
