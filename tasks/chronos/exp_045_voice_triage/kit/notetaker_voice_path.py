"""The voice path of casasol `scripts/notetaker.py` at `ed02fbb`, verbatim: the functions exp_045 and the bot's
voice triage call. The rest of that file is the WhatsApp notetaker and is not needed here."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

WHISPER_MODEL = os.environ.get("NOTETAKER_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")


def to_wav(src: Path) -> Path:
    """Any container → 16 kHz mono PCM WAV, which is what Whisper expects.

    The output is a new file next to the source, `<stem>.16k.wav`, never the source itself: a .wav upload
    would otherwise be both ffmpeg's input and its output, which ffmpeg refuses. The caller deletes the
    output; on failure nothing is left behind.
    """
    src = Path(src)
    dst = src.with_name(src.stem + ".16k.wav")
    if dst.resolve() == src.resolve():      # cannot happen by construction; never let ffmpeg overwrite its input
        raise ValueError("to_wav: output would overwrite the input")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav",
             str(dst)],
            check=True,
        )
    except BaseException:
        dst.unlink(missing_ok=True)         # a half-written conversion is audio too
        raise
    return dst


def audio_seconds(wav_path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def format_transcript(segments: list[dict]) -> str:
    """Whisper segments → readable paragraphs, each opened by its [m:ss] start.

    A new paragraph starts after a pause of PARAGRAPH_GAP_S, or once a paragraph has
    run PARAGRAPH_MAX_S — deterministic, so the same audio always reads the same.
    """
    paras: list[tuple[float, list[str]]] = []
    last_end = 0.0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if not paras or start - last_end >= PARAGRAPH_GAP_S or start - paras[-1][0] >= PARAGRAPH_MAX_S:
            paras.append((start, [text]))
        else:
            paras[-1][1].append(text)
        last_end = end
    return "\n\n".join(f"[{_stamp(start)}] {' '.join(parts)}" for start, parts in paras)


def transcribe(wav_path: Path, model: str = WHISPER_MODEL) -> tuple[str, str, str]:
    """Returns (plain transcript for the model, [m:ss] paragraphs for the email, language)."""
    import mlx_whisper  # heavy import; only when actually transcribing
    result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=model)
    plain = result["text"].strip()
    return plain, format_transcript(result.get("segments", [])) or plain, result.get("language", "")


_SPOKEN_PHONE = re.compile(r"(?<![\w+])(?P<plus>\+|plus|плюс|más|mas)?[\s,]*(?P<groups>\d{1,6}(?:[\s,.-]+\d{1,6}){2,})(?![\d])",
                           re.I)


def join_spoken_numbers(text: str) -> str:
    """A phone number read out in groups comes back from Whisper as "plus 44, 77, 00, 90, 01, 28": the commas hide it
    from every phone detector downstream. Join such runs into "+44 77 00 90 01 28" when they start with a spoken plus
    or a 0 and hold 9–15 digits; anything else (prices, dates, "12, 00, 15, 00 par mois") is left alone (exp_045 dev)."""
    def join(m: re.Match) -> str:
        groups = re.findall(r"\d+", m.group("groups"))
        digits = sum(len(g) for g in groups)
        if not 9 <= digits <= 15 or not (m.group("plus") or groups[0].startswith("0")):
            return m.group(0)
        lead = " " if m.group(0)[:1].isspace() else ""
        return lead + ("+" if m.group("plus") else "") + " ".join(groups)
    return _SPOKEN_PHONE.sub(join, text)
