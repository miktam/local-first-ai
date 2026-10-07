"""Hashed shingles of withheld text, for the leak check (BUILD_SPEC §5.5, §5.9).

The file `tools/withheld_shingles.sha256` is built on the mbp in S1 (RUNBOOK
step 7, by `tasks/build_manifests.py`, which should call `write_shingle_file`
here so that writer and reader share one normalisation). It holds no text:

    # comment lines (format, counts)
    <12 hex>     sha256 prefix of one normalised 8-word shingle
    <64 hex>     full sha256 of one normalised withheld option string (>= 4 words)

Normalisation: Unicode NFKC, lower-case, every character that is neither a
letter, a digit nor whitespace removed, whitespace collapsed; words are the
whitespace-separated tokens. A shingle is 8 consecutive words joined by one
space. An option of 4–8 words is found by hashing every 4–8-word window of
the scanned text (an 8-word option is a full-hash finding, not one shingle
warning; review fix 2026-10-03); a longer option is covered by its shingles.
Short withheld questions (RGB queries, GPQA EN/DE questions, AIME-DE problems
of 4–8 words) are hashed as options too, so quoting one alone is a finding.
Strings of 1–3 words are not hashed (Amendment 1, 2026-10-04): on the mbp, six
3-word GPQA options matched ordinary English in the kit's own code and docs
(30 places, one inside GENERIC_OPTIONS itself). A phrase that short, without
its question, reveals nothing withheld, and it would block ordinary commits.

`collect_from_sources()` builds the same sets on the fly from the withheld
sources when the file does not exist yet (before RUNBOOK step 7):
GPQA (all files of `gpqa` and `gpqa-multilingual`), the German files of
`aime26-multilingual`, RGB queries and answers (`RGB-src/data/en*.json`),
RGB's `config/instruction.yaml` strings, and `$EXP036_PRIVATE/manifests`.
Generic multiple-choice phrases are never hashed as options
(GENERIC_OPTIONS), because they are not withheld text and would block every
push.

Local corpora (`local_corpus_texts()`, review fix 2026-10-03): the RGB
documents (positive, negative and positive_wrong passages of en, en_fact,
en_int, en_refine) and the FineWeb-2 rows the kit reads (0–4999 for H5, plus
the T5, T6 and T9 rows recorded in gate/texts/MANIFEST.json) are far too
large to commit as hashes (≈ 3.6 M distinct 8-word shingles, ≈ 47 MB). The
leak check streams them from `$EXP036_DATA` instead, on every run where the
data is present (both hosts), and matches them against the shingles of the
scanned text only. `public_shingle_strings()` lists the kit's own public text
(the Apache licence and the gate texts T1–T4 with their sources, which quote
the Grundgesetz that FineWeb also quotes); those shingles and all-digit ones
are never evidence of a leak.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

SHINGLE_WORDS = 8
PREFIX_HEX = 12
MIN_OPTION_WORDS = 4                     # Amendment 1 (2026-10-04); was 3
MAX_OPTION_WINDOW = SHINGLE_WORDS        # 4–8-word windows (review fix 2026-10-03 made it 3–8; was 3–7)
FORMAT_LINE = (
    "# exp036 withheld shingles v1: 12-hex sha256 prefixes of normalised 8-word shingles; "
    "64-hex sha256 of normalised option strings of >= 4 words"
)

# Normalised phrases that are never treated as withheld options.
GENERIC_OPTIONS = frozenset({
    "none of the above", "all of the above", "none of these", "all of these",
    "both a and b", "neither a nor b", "cannot be determined", "not enough information",
    "none of the above answers", "all of the above answers", "keine der genannten",
    "alle genannten", "keine der obigen", "alle der obigen", "keine der antworten",
    "insufficient information", "i dont know",
})

_OPTION_COLUMN = re.compile(r"answer|option|choice|antwort", re.IGNORECASE)


def normalise_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return text.split()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def shingle_prefixes(text_or_words) -> set[str]:
    """Prefixes of every 8-word shingle except the all-digit ones (digit runs are not withheld text; review
    fix 2026-10-03)."""
    words = normalise_words(text_or_words) if isinstance(text_or_words, str) else list(text_or_words)
    out = set()
    for i in range(len(words) - SHINGLE_WORDS + 1):
        w = words[i:i + SHINGLE_WORDS]
        if not all(x.isdigit() for x in w):
            out.add(_h(" ".join(w))[:PREFIX_HEX])
    return out


def option_hash(text: str) -> str | None:
    words = normalise_words(text)
    if len(words) < MIN_OPTION_WORDS:
        return None
    joined = " ".join(words)
    if joined in GENERIC_OPTIONS:
        return None
    return _h(joined)


class ShingleSet:
    """The loaded sets plus where they came from."""

    def __init__(self, prefixes=None, options=None, sources=None):
        self.prefixes: set[str] = set(prefixes or ())
        self.options: set[str] = set(options or ())
        self.sources: list[str] = list(sources or [])

    def __bool__(self) -> bool:
        return bool(self.prefixes or self.options)

    def add_text(self, text: str) -> None:
        self.prefixes |= shingle_prefixes(text)

    def add_option(self, text: str) -> None:
        h = option_hash(text)
        if h:
            self.options.add(h)

    def matches(self, text_or_words) -> list[dict]:
        """Every hit as {"kind": "shingle"|"option", "word_index": i, "hash": h}
        (h is a 12-hex prefix; no text is returned)."""
        words = normalise_words(text_or_words) if isinstance(text_or_words, str) else list(text_or_words)
        hits = []
        if self.prefixes:
            for i in range(len(words) - SHINGLE_WORDS + 1):
                h = _h(" ".join(words[i:i + SHINGLE_WORDS]))[:PREFIX_HEX]
                if h in self.prefixes:
                    hits.append({"kind": "shingle", "word_index": i, "hash": h})
        if self.options:
            for n in range(MIN_OPTION_WORDS, MAX_OPTION_WINDOW + 1):
                for i in range(len(words) - n + 1):
                    joined = " ".join(words[i:i + n])
                    if joined in GENERIC_OPTIONS:
                        continue
                    h = _h(joined)
                    if h in self.options:
                        hits.append({"kind": "option", "word_index": i, "hash": h[:PREFIX_HEX], "words": n})
        return hits


def write_shingle_file(path, texts: Iterable[str], options: Iterable[str], note: str = "",
                       public_texts: Iterable[str] = ()) -> dict:
    """Write the shingle file; byte-identical output for the same input.

    public_texts (optional, recommended): text of the public sets (MMLU-ProX
    questions and options, IFBench prompts, AIME EN problems). A shingle or an
    option string that also occurs in public text is not evidence of a leak,
    so it is left out; the header records how many were removed."""
    s = ShingleSet()
    n_texts = n_opts = 0
    for t in texts:
        if t:
            s.add_text(t)
            n_texts += 1
    for o in options:
        if o:
            s.add_option(o)
            s.add_text(o)
            n_opts += 1
    removed = subtract_public(s, public_texts)
    lines = sorted(s.prefixes) + sorted(s.options)
    header = [FORMAT_LINE, f"# {len(s.prefixes)} shingle prefixes, {len(s.options)} option hashes"
              + (f"; removed as also public: {removed[0]} shingles, {removed[1]} options" if any(removed) else "")]
    if note:
        header.append("# " + note.replace("\n", " "))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    return {"prefixes": len(s.prefixes), "options": len(s.options), "texts": n_texts, "option_strings": n_opts}


def read_shingle_file(path) -> ShingleSet:
    s = ShingleSet(sources=[f"shingle file {Path(path).name}"])
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"[0-9a-f]{12}", line):
            s.prefixes.add(line)
        elif re.fullmatch(r"[0-9a-f]{64}", line):
            s.options.add(line)
        else:
            raise ValueError(f"{path}: unexpected line {line[:20]!r}")
    return s


# --------------------------------------------------------------------------
# On-the-fly sources (before the shingle file exists)
# --------------------------------------------------------------------------

def _rows_from_file(path: Path, warnings: list) -> list[dict]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                return list(csv.DictReader(f))
        if suffix == ".jsonl":
            rows = []
            for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        obj = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    rows.append(obj if isinstance(obj, dict) else {"value": obj})
            return rows
        if suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                # RGB's en*.json files are JSON lines despite the suffix.
                rows = []
                for ln in text.splitlines():
                    if ln.strip().startswith("{"):
                        try:
                            rows.append(json.loads(ln))
                        except json.JSONDecodeError:
                            continue
                return rows
            if isinstance(obj, list):
                return [r if isinstance(r, dict) else {"value": r} for r in obj]
            if isinstance(obj, dict):
                for key in ("items", "rows", "data", "records"):
                    if isinstance(obj.get(key), list):
                        return [r if isinstance(r, dict) else {"value": r} for r in obj[key]]
                return [obj]
            return []
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # lazy; only for parquet sources
            except ImportError:
                warnings.append(f"pyarrow missing: cannot read {path.name}")
                return []
            return pq.read_table(path).to_pylist()
    except (OSError, UnicodeError, csv.Error) as e:
        warnings.append(f"cannot read {path.name}: {e}")
    return []


def _strings(value, key: str = ""):
    """(key, string) pairs of every string inside a JSON-like value."""
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v, key)


_QUESTION_KEYS = frozenset({"question", "problem", "query"})


def _add_rows(s: ShingleSet, rows: list[dict], option_keys=None) -> int:
    n = 0
    for row in rows:
        for key, text in _strings(row):
            n_words = len(normalise_words(text))
            if n_words >= SHINGLE_WORDS:
                s.add_text(text)
            is_option = key in option_keys if option_keys is not None else bool(_OPTION_COLUMN.search(key))
            if key.lower() in _QUESTION_KEYS and n_words <= SHINGLE_WORDS:
                is_option = True                     # a short withheld question is hashed whole
            if is_option and MIN_OPTION_WORDS <= n_words:
                s.add_option(text)
        n += 1
    return n


_DATA_SUFFIXES = (".csv", ".json", ".jsonl", ".parquet")
_GERMAN_PATH = re.compile(r"(^|[/_.\-=])(deu|de|german|deutsch)([/_.\-=]|$)", re.IGNORECASE)


def _data_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES and ".cache" not in p.parts
    )


def public_set(public_texts: Iterable[str]) -> ShingleSet:
    """Shingle prefixes and every 4-8-word window hash of public text."""
    pub = ShingleSet()
    for t in public_texts:
        if t:
            pub.add_text(t)
            words = normalise_words(t)
            for n in range(MIN_OPTION_WORDS, MAX_OPTION_WINDOW + 1):
                for i in range(len(words) - n + 1):
                    pub.options.add(_h(" ".join(words[i:i + n])))
    return pub


def subtract_public(s: ShingleSet, public_texts: Iterable[str]) -> tuple[int, int]:
    """Remove from s what also occurs in public text; returns (shingles, options) removed."""
    pub = public_set(public_texts)
    removed = (len(s.prefixes & pub.prefixes), len(s.options & pub.options))
    s.prefixes -= pub.prefixes
    s.options -= pub.options
    return removed


def public_file_texts(exp_dir) -> list[str]:
    """The kit's own public text (PUBLIC_FILES, PUBLIC_GLOBS), as step 7 passes it to write_shingle_file."""
    root = Path(exp_dir)
    files = [root / f for f in PUBLIC_FILES] + [p for g in PUBLIC_GLOBS for p in sorted(root.glob(g))]
    return [f.read_text(encoding="utf-8", errors="replace") for f in files if f.is_file()]


# The GPQA EN files the kit uses; step 7 (tasks/build_manifests.py) hashes exactly these.
GPQA_EN_FILES = ("gpqa_diamond.csv", "gpqa_main.csv")


def collect_from_sources(data_dir, private_dir=None, exp_dir=None) -> tuple[ShingleSet, list[str]]:
    """Shingles and option hashes from the withheld sources found locally.
    Returns (set, warnings); set.sources names what was read.

    Mirrors step 7 (Amendment 2, 2026-10-04): GPQA EN only from GPQA_EN_FILES, and the
    kit's public text (licence, gate texts) subtracted, so the check gives the same
    answer before and after tools/withheld_shingles.sha256 exists."""
    data_dir = Path(data_dir)
    s = ShingleSet()
    warnings: list[str] = []

    for name in ("gpqa", "gpqa-multilingual"):
        files = _data_files(data_dir / name)
        if name == "gpqa":
            used = [f for f in files if f.name in GPQA_EN_FILES]
            if files and not used:
                warnings.append(f"gpqa: none of {GPQA_EN_FILES} found; reading every data file")
            files = used or files
        if name == "gpqa-multilingual":
            german = [f for f in files if _GERMAN_PATH.search(f.relative_to(data_dir / name).as_posix())]
            files = german or files
        n = sum(_add_rows(s, _rows_from_file(f, warnings)) for f in files)
        if n:
            s.sources.append(f"{name} ({len(files)} files, {n} rows)")

    files = _data_files(data_dir / "aime26-multilingual")
    german = [f for f in files if _GERMAN_PATH.search(f.relative_to(data_dir / "aime26-multilingual").as_posix())]
    rows = []
    for f in german:
        rows.extend(_rows_from_file(f, warnings))
    if not german:
        # Fall back to rows labelled German; never the English rows (AIME EN is public).
        for f in files:
            for r in _rows_from_file(f, warnings):
                lang = str(r.get("language") or r.get("lang") or r.get("config") or "").lower()
                if lang in {"de", "deu", "german", "deutsch"}:
                    rows.append(r)
        if files and not rows:
            warnings.append("aime26-multilingual present but no German rows identified; not used")
    if rows:
        n = _add_rows(s, rows, option_keys=set())
        s.sources.append(f"aime26-multilingual German ({n} rows)")

    rgb = data_dir / "RGB-src" / "data"
    n_rgb = 0
    for fname in ("en.json", "en_fact.json", "en_int.json"):
        p = rgb / fname
        if p.is_file():
            for row in _rows_from_file(p, warnings):
                q = row.get("query")
                if isinstance(q, str):
                    s.add_text(q)
                    s.add_option(q)
                for key in ("answer", "fakeanswer"):
                    for _, ans in _strings(row.get(key)):
                        s.add_option(ans)
                        s.add_text(ans)
                n_rgb += 1
    if n_rgb:
        s.sources.append(f"RGB queries and answers ({n_rgb} rows)")
    instr = rgb_instruction_strings(data_dir, warnings)
    for t in instr:
        s.add_text(t)
        s.add_option(t)
    if instr:
        s.sources.append(f"RGB config/instruction.yaml ({len(instr)} strings)")

    if private_dir:
        pm = Path(private_dir) / "manifests"
        files = _data_files(pm)
        n = sum(_add_rows(s, _rows_from_file(f, warnings)) for f in files)
        if n:
            s.sources.append(f"$EXP036_PRIVATE/manifests ({len(files)} files, {n} rows)")
    if s:
        removed = subtract_public(s, public_file_texts(exp_dir or Path(__file__).resolve().parents[1]))
        if any(removed):
            s.sources.append(f"minus the kit's public text ({removed[0]} shingles, {removed[1]} options)")
    return s, warnings


# --------------------------------------------------------------------------
# Local corpora: withheld text too large to commit as hashes (review fix 2026-10-03)
# --------------------------------------------------------------------------

RGB_DATA_FILES = ("en.json", "en_fact.json", "en_int.json", "en_refine.json")
RGB_DOC_KEYS = ("positive", "negative", "positive_wrong")
FINEWEB_REL = Path("fineweb-2") / "data" / "deu_Latn" / "test" / "000_00000.parquet"
FINEWEB_H5_ROWS = 5000          # HYPOTHESIS H5: the first 5,000 documents in file order
PUBLIC_FILES = ("LICENSE-APACHE-2.0",)
PUBLIC_GLOBS = ("gate/texts/src/*.txt", "gate/texts/T*.txt")


def rgb_instruction_strings(data_dir, warnings: list) -> list[str]:
    """Every string of RGB's config/instruction.yaml (system and instruction prompts)."""
    p = Path(data_dir) / "RGB-src" / "config" / "instruction.yaml"
    if not p.is_file():
        return []
    try:
        import yaml  # lazy: pinned in env/requirements-mbp.txt
    except ImportError:
        warnings.append("PyYAML missing: RGB config/instruction.yaml not read")
        return []
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as e:  # a broken file is reported, never fatal
        warnings.append(f"cannot read RGB config/instruction.yaml: {type(e).__name__}")
        return []
    return [t for _, t in _strings(doc) if len(normalise_words(t)) >= MIN_OPTION_WORDS]


def fineweb_rows(exp_dir) -> list[int]:
    """The FineWeb-2 rows the kit reads: 0–4999 (H5) and the T5, T6 and T9 rows of gate/texts/MANIFEST.json."""
    rows = set(range(FINEWEB_H5_ROWS))
    try:
        web = json.loads((Path(exp_dir) / "gate" / "texts" / "MANIFEST.json").read_text(encoding="utf-8"))["web"]
    except (OSError, ValueError, KeyError, TypeError):
        return sorted(rows)
    for k in ("T5", "T6"):
        r = (web.get(k) or {}).get("row")
        if isinstance(r, int):
            rows.add(r)
    for c in (web.get("T9") or {}).get("components") or []:
        if isinstance(c, dict) and isinstance(c.get("row"), int):
            rows.add(c["row"])
    return sorted(rows)


def local_corpus_available(data_dir) -> bool:
    d = Path(data_dir)
    return (d / "RGB-src" / "data").is_dir() or (d / FINEWEB_REL).is_file()


def local_corpus_texts(data_dir, exp_dir, warnings: list, sources: list):
    """Yield (source name, text) for every local-corpus document; `sources` gets one line per corpus read."""
    d = Path(data_dir)
    for fname in RGB_DATA_FILES:
        p = d / "RGB-src" / "data" / fname
        if not p.is_file():
            continue
        n = 0
        for row in _rows_from_file(p, warnings):
            for key in RGB_DOC_KEYS:
                for _, t in _strings(row.get(key)):
                    n += 1
                    yield f"RGB {fname} documents", t
        sources.append(f"RGB {fname} documents ({n} passages)")
    pq_path = d / FINEWEB_REL
    if pq_path.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            warnings.append("pyarrow missing: FineWeb-2 rows not read")
            return
        col = pq.read_table(str(pq_path), columns=["text"]).column("text")
        rows = [r for r in fineweb_rows(exp_dir) if r < len(col)]
        for r in rows:
            t = col[r].as_py()
            if isinstance(t, str):
                yield "FineWeb-2 deu_Latn rows", t
        sources.append(f"FineWeb-2 deu_Latn rows ({len(rows)} documents)")


def window_strings(words: list[str], n: int = SHINGLE_WORDS):
    for i in range(len(words) - n + 1):
        yield i, " ".join(words[i:i + n])


def public_shingle_strings(exp_dir) -> set[str]:
    """8-word shingles of the kit's own public text (never evidence of a leak)."""
    root = Path(exp_dir)
    files = [root / f for f in PUBLIC_FILES] + [p for g in PUBLIC_GLOBS for p in sorted(root.glob(g))]
    out: set[str] = set()
    for f in files:
        if f.is_file():
            out |= {w for _, w in window_strings(normalise_words(f.read_text(encoding="utf-8", errors="replace")))}
    return out


def all_digits(shingle: str) -> bool:
    return all(w.isdigit() for w in shingle.split())
