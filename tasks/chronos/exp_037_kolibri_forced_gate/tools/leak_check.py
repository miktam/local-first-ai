"""Leak check before every commit and push (BUILD_SPEC §5.9; HYPOTHESIS
"Sources & Rights": withheld text is never published anywhere).

Modes (combinable):
    leak_check.py --range @{u}..HEAD --staged     every commit not yet pushed, plus the index
    leak_check.py --new-tip SHA                    SHA --not --remotes (a new branch)
    leak_check.py --all                            every file of the experiment directory
    leak_check.py PATH [PATH ...]                  files or directories (e.g. a blog post)
In range and staged mode, text rules run on the lines each commit adds (lines
already upstream are already public); the file rules (type, size, withheld
JSON fields) run on each committed file version. Every commit in the range is
checked, so a leak added and removed again before the push is still caught.

Rules. "Kit files" are everything outside the raw and pilot outputs (below):
- tailnet addresses 100.64.0.0/10 and the Tailscale IPv6 prefix fd7a:115c:a1e0::/48;
  home-directory paths /Users/<name>, also JSON-escaped (\\/Users\\/<name>); tokens
  (hf_…, GitHub, Anthropic, OpenAI, AWS, Slack, Telegram bot); private keys;
  e-mail addresses other than hello@localfirstai.eu and noreply@anthropic.com
  (git@host SSH remotes are not e-mail; reserved documentation domains such as
  example.com are warnings); the benchmark canary; withheld shingles and option
  hashes; `*.local` and `*.ts.net` host names; and runtime names read here and
  never committed: `whoami`, `scutil --get LocalHostName`, $EXP036_PRIVATE_NAMES.
  The bare LocalHostName is a finding in machine-written records (results/,
  evidence/, aborted/, host/, tasks/manifests/) and a warning in hand-written
  files, which name the hosts by role.
- Raw and pilot outputs (results/raw/**, results/pilot/**, and the copies an abort
  moves to aborted/<tag>/S*-*.jsonl and aborted/<UTC>-pilot/pilot/**): withheld shingles,
  the canary and withheld fields are findings; every other match is a warning
  (e.g. an address a model wrote), printed for the commit note.
- Withheld-text matches, everywhere: two or more 8-word shingle hits in one
  text unit are findings (a quote of 9+ words always gives two); one isolated
  hit is a warning. Option-hash hits are findings, except 3–4-word ones inside
  raw / pilot outputs, which are warnings (generic short phrases in public
  model outputs must not block a push; scan_withheld()).
- Withheld fields: in JSON/JSONL under results/, evidence/, aborted/, host/ and
  tasks/manifests/, a record of a withheld set (GPQA, RGB, AIME-DE) must not
  carry text, gold, prompt or completion-id fields.
- Local corpora (review fix 2026-10-03): the RGB documents and the FineWeb-2
  rows the kit reads are streamed from $EXP036_DATA when present
  (tools/shingles.py local_corpus_texts) and matched against the scanned
  text's own 8-word shingles, minus the kit's public text and all-digit
  shingles; two or more hits in one text unit are a finding, one a warning.
  Public raw / pilot outputs are not matched (a model may echo a common web
  phrase, and they are the bulk of a session push); withheld-set raw files are,
  since they carry no text by construction.
- Files: *.pdf, *.parquet, *.npy, *.npz, *.safetensors are refused; archives and
  binary formats (*.gz, *.zip, *.tar, *.tgz, *.bz2, *.xz, *.7z, *.zst, *.arrow,
  *.bin, *.pt, *.pth, *.gguf, *.h5) and any file with a NUL byte or that is not
  strict UTF-8 are findings inside the experiment directory and warnings outside
  it (the text rules cannot read them); any file > 5 MB outside results/raw/**,
  results/pilot/** and aborted/**, > 50 MB inside them (pre-freeze review
  2026-10-06: aborted/ holds moved raw outputs and pilot files, and a moved K8
  completions file can exceed 5 MB; a committed pilot step log is about 1 MB a
  cell at B = 8, and one over 50 MB stays uncommitted, RUNBOOK step 13).
- tests/fixtures/leak/ is exempt only from the personal-pattern rules (addresses,
  paths, e-mail, host and runtime names), for planted test fixtures; every other
  rule applies there. No file is exempt as a whole (review fix 2026-10-03: this
  file passes its own rules). tools/withheld_shingles.sha256 is checked for its
  format (hex and comment lines only) instead of its content.

Withheld text source: tools/withheld_shingles.sha256, else the GPQA / RGB /
AIME-DE sources under $EXP036_DATA or $EXP036_PRIVATE/manifests. If none is
present the check exits 3, unless --no-gpqa-source is passed (only for the
pre-registration push from the mini, which carries no withheld text).

Exit codes: 0 clean (warnings allowed), 1 finding, 2 usage, 3 no withheld source.
No network; stdlib only (pyarrow is used lazily to read parquet sources).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common, shingles

SHINGLE_FILE_REL = "tools/withheld_shingles.sha256"
ALLOWLIST_FILES: set[str] = set()           # no whole-file exemption (review fix 2026-10-03)
ALLOWLIST_DIRS = ("tests/fixtures/leak/",)   # personal-pattern rules only
BINARY_ALLOWLIST: set[str] = set()         # kit paths that may hold binary or non-UTF-8 bytes (none)
RAW_DIRS = ("results/raw/", "results/pilot/")
RECORD_DIRS = ("results/", "evidence/", "aborted/", "host/", "tasks/manifests/")
REFUSED_SUFFIXES = (".pdf", ".parquet", ".npy", ".npz", ".safetensors")
ARCHIVE_SUFFIXES = (".gz", ".zip", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".zst", ".arrow", ".bin", ".pt", ".pth",
                    ".gguf", ".h5")
MAX_BYTES = 5 * 1000 * 1000
MAX_BYTES_RAW = 50 * 1000 * 1000
LARGE_DIRS = RAW_DIRS + ("aborted/",)        # MAX_BYTES_RAW applies here (pre-freeze review 2026-10-06)
# Raw and pilot outputs moved by an abort keep the raw rules (pre-freeze review 2026-10-06): runner/run.py abort_cell
# moves a cell's files to aborted/<tag>/<session>-<name>, cmd_abort_pilot a pilot to aborted/<UTC>-pilot/pilot/.
RAW_ABORTED_RE = re.compile(r"^aborted/[^/]+/S\d+b?-[^/]+\.jsonl$|^aborted/[^/]+-pilot/pilot/")
ALLOWED_EMAILS = {"hello@localfirstai.eu", "noreply@anthropic.com"}

TAILNET_IP = re.compile(r"(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?!\.?\d)")
TAILNET_NETWORK_LITERAL = "100.64.0.0"   # the documented CIDR 100.64.0.0/10 itself
TAILNET_IP6 = re.compile(r"(?i)(?<![0-9a-f:])fd7a:115c:a1e0:[0-9a-f:]*[0-9a-f]")
HOME_PATH = re.compile(r"\\?/Users\\?/(?!Shared(?:\\?/|\b))[A-Za-z0-9_.-]+")
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
RESERVED_DOMAIN = re.compile(r"(?:^|\.)(?:example\.(?:com|net|org)|example|test|invalid|localhost)$", re.I)
MDNS_HOST = re.compile(r"(?<![\w.-])[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.local(?![\w.(-])")
TAILNET_HOST = re.compile(r"(?<![\w-])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.ts\.net(?![\w-])", re.I)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
TOKENS = (
    ("hf_token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("anthropic_key", re.compile(r"(?<![A-Za-z0-9_])sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"(?<![A-Za-z0-9_])sk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("aws_key", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("telegram_bot_token", re.compile(r"(?<!\d)\d{8,10}:AA[A-Za-z0-9_-]{33}(?![A-Za-z0-9_-])")),
)
CANARY = (
    re.compile(r"canary\s+GUID\s+[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", re.I),
    re.compile(r"BENCHMARK\s+DATA\s+SHOULD\s+NEVER\s+APPEAR", re.I),
)
WITHHELD_TASK = re.compile(
    r"(?i)(?<![a-z])(?:gpqa|rgb)|aime[_-]?(?:26[_-]?)?(?:de|deu|multilingual)(?![a-z])"
)
WITHHELD_FIELDS = {
    "text", "gold", "prompt", "prompt_text", "question", "options", "choices", "messages",
    "content", "completion", "completion_ids", "prompt_ids", "rendered_prompt_ids",
    "reasoning", "answer_text", "documents", "passages",
    # review fix 2026-10-03: RGB-shaped and rendered-prompt fields
    "rendered_prompt", "docs", "docs_text", "document", "passage", "positive", "negative", "positive_wrong",
    "query", "fakeanswer",
}
JSON_SUFFIXES = (".json", ".jsonl")
MIN_SHINGLE_HITS = 2        # one isolated 8-word overlap is a warning
MIN_OPTION_WORDS_RAW = 5    # 4-word option matches in raw / pilot outputs are warnings (options start at 4 words, Amendment 1)


@dataclass
class Finding:
    kind: str
    path: str
    line: int | None
    detail: str
    severity: str = "finding"         # "finding" | "warning"
    commit: str | None = None

    def fmt(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        at = f" @{self.commit[:10]}" if self.commit else ""
        tag = "FINDING" if self.severity == "finding" else "warning"
        return f"{tag} [{self.kind}] {where}{at} — {self.detail}"


@dataclass
class Result:
    findings: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    source_warnings: list = field(default_factory=list)
    withheld_check: str = "skipped"
    n_files: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, f: Finding) -> None:
        (self.findings if f.severity == "finding" else self.warnings).append(f)


class NoWithheldSource(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Context: runtime names, shingles, classification
# --------------------------------------------------------------------------

def _runtime_names() -> dict:
    names = {"whoami": None, "localhostname": None, "private": []}
    try:
        user = getpass.getuser()
        names["whoami"] = user if user and len(user) >= 4 else None
    except Exception:
        pass
    rc, out, _ = common.run(["scutil", "--get", "LocalHostName"], timeout=5)
    if rc == 0 and len(out.strip()) >= 4:
        names["localhostname"] = out.strip()
    raw = os.environ.get("EXP036_PRIVATE_NAMES", "")
    names["private"] = [n.strip() for n in re.split(r"[,;:\n]", raw) if len(n.strip()) >= 3]
    return names


class Context:
    def __init__(self, exp_dir=None, shingle_set=None, names=None):
        self.exp_dir = Path(exp_dir or common.EXP_DIR).resolve()
        self.repo = common.repo_root(self.exp_dir)
        self.exp_prefix = None
        if self.repo:
            try:
                self.exp_prefix = self.exp_dir.relative_to(self.repo.resolve()).as_posix() + "/"
            except ValueError:
                self.exp_prefix = None
        self.shingles = shingle_set or shingles.ShingleSet()
        self.names = names if names is not None else _runtime_names()
        # Local-corpus pass (review fix 2026-10-03): off unless enable_local() found the data.
        self.local_data: Path | None = None
        self.pending: list = []
        self._name_res = []
        if self.names.get("whoami"):
            self._name_res.append(("username", re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(self.names["whoami"]) + r"(?![A-Za-z0-9_])"), "finding"))
        if self.names.get("localhostname"):
            self._name_res.append(("hostname", re.compile(
                r"(?<![A-Za-z0-9_-])" + re.escape(self.names["localhostname"]) + r"(?![A-Za-z0-9_-])",
                re.IGNORECASE), "hostname"))
        for n in self.names.get("private") or []:
            self._name_res.append(("private_name", re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", re.IGNORECASE), "finding"))

    # path helpers ---------------------------------------------------------
    def exp_rel(self, repo_rel: str | None, abs_path: Path | None = None) -> str | None:
        """Path relative to the experiment directory, or None if outside it."""
        if abs_path is not None:
            try:
                return Path(abs_path).resolve().relative_to(self.exp_dir).as_posix()
            except ValueError:
                return None
        if repo_rel is not None and self.exp_prefix and repo_rel.startswith(self.exp_prefix):
            return repo_rel[len(self.exp_prefix):]
        return None

    @staticmethod
    def is_raw(erel: str | None) -> bool:
        return bool(erel) and (erel.startswith(RAW_DIRS) or RAW_ABORTED_RE.match(erel) is not None)

    @staticmethod
    def is_record(erel: str | None) -> bool:
        return bool(erel) and erel.startswith(RECORD_DIRS)

    @staticmethod
    def allowlisted(erel: str | None) -> bool:
        return bool(erel) and erel in ALLOWLIST_FILES

    @staticmethod
    def fixture(erel: str | None) -> bool:
        """tests/fixtures/leak/: only the personal-pattern rules are off there."""
        return bool(erel) and erel.startswith(ALLOWLIST_DIRS)

    def enable_local(self, data_dir) -> bool:
        if data_dir is not None and shingles.local_corpus_available(data_dir):
            self.local_data = Path(data_dir)
        return self.local_data is not None


def load_withheld(exp_dir=None, allow_missing: bool = False) -> tuple[shingles.ShingleSet, list[str]]:
    exp_dir = Path(exp_dir or common.EXP_DIR)
    f = exp_dir / SHINGLE_FILE_REL
    if f.is_file():
        return shingles.read_shingle_file(f), []
    s, warns = shingles.collect_from_sources(common.data_dir(), common.private_dir())
    if not s and not allow_missing:
        raise NoWithheldSource(
            f"no withheld-text source: {SHINGLE_FILE_REL} is absent and no GPQA / RGB / AIME-DE "
            f"source was found under $EXP036_DATA or $EXP036_PRIVATE/manifests"
        )
    return s, warns


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def _mask(s: str, keep: int = 6) -> str:
    return s if len(s) <= keep * 2 else f"{s[:keep]}…{s[-3:]}"


def scan_lines(lines, display: str, erel: str | None, ctx: Context, commit=None) -> list[Finding]:
    """Text rules on (line_no, text) pairs."""
    raw = ctx.is_raw(erel)
    record = ctx.is_record(erel)
    personal = not ctx.fixture(erel)
    out: list[Finding] = []

    def add(kind, ln, detail, strict=True):
        sev = "finding" if (strict and not raw) else "warning"
        out.append(Finding(kind, display, ln, detail, sev, commit))

    for ln, text in lines:
        for m in TAILNET_IP.finditer(text) if personal else ():
            if m.group(0) != TAILNET_NETWORK_LITERAL:
                add("tailnet_ip", ln, m.group(0))
        for m in TAILNET_IP6.finditer(text) if personal else ():
            add("tailnet_ip", ln, _mask(m.group(0)))
        for m in HOME_PATH.finditer(text) if personal else ():
            add("home_path", ln, m.group(0))
        for kind, rx in TOKENS:
            for m in rx.finditer(text):
                add(kind, ln, _mask(m.group(0)))
        if PRIVATE_KEY.search(text):
            add("private_key", ln, "PRIVATE KEY block")
        for m in EMAIL.finditer(text) if personal else ():
            local, domain = m.group(1), m.group(2)
            addr = f"{local}@{domain}"
            if addr.lower() in ALLOWED_EMAILS or local == "git":
                continue
            add("email", ln, addr, strict=not RESERVED_DOMAIN.search(domain))
        for m in MDNS_HOST.finditer(text) if personal else ():
            add("mdns_host", ln, m.group(0))
        for m in TAILNET_HOST.finditer(text) if personal else ():
            add("tailnet_host", ln, m.group(0))
        for rx in CANARY:
            if rx.search(text):
                out.append(Finding("canary", display, ln, "benchmark canary string", "finding", commit))
        for kind, rx, mode in ctx._name_res if personal else ():
            for m in rx.finditer(text):
                strict = record if mode == "hostname" else True
                add(kind, ln, _mask(m.group(0), 2), strict=strict)
    return out


def _units_for_shingles(lines, is_json: bool):
    """(line_no, text) units: each JSON string value separately for JSON(L),
    otherwise one unit per contiguous block of lines (line numbers kept)."""
    if is_json:
        for ln, text in lines:
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                yield ln, text
                continue
            for _, s in shingles._strings(obj):
                yield ln, s
        return
    block: list = []
    prev = None
    for ln, text in lines:
        if block and prev is not None and ln is not None and ln != prev + 1:
            yield block
            block = []
        block.append((ln, text))
        prev = ln
    if block:
        yield block


def scan_withheld(lines, display: str, ctx: Context, is_json: bool, commit=None, erel: str | None = None) -> list[Finding]:
    """Withheld shingles and option hashes.

    Severity (to keep generic phrasing in public model outputs from blocking a
    push, while any real quote still blocks it): within one text unit, two or
    more 8-word shingle hits are findings (any quote of 9+ words gives two);
    a single isolated hit is a warning. An option-hash hit is a finding,
    except a 3–4-word one inside raw or pilot outputs, which is a warning."""
    if not ctx.shingles and ctx.local_data is None:
        return []
    raw = ctx.is_raw(erel)
    out = []
    for unit in _units_for_shingles(lines, is_json):
        if isinstance(unit, tuple):
            ln, text = unit
            words = shingles.normalise_words(text)
            line_of = lambda i, ln=ln: ln  # noqa: E731
        else:
            words, starts = [], []
            for ln, text in unit:
                starts.append((len(words), ln))
                words.extend(shingles.normalise_words(text))

            def line_of(i, starts=starts):
                cur = starts[0][1]
                for w0, ln in starts:
                    if w0 > i:
                        break
                    cur = ln
                return cur
        if ctx.local_data is not None and len(words) >= shingles.SHINGLE_WORDS and (
                not raw or WITHHELD_TASK.search(Path(erel or display).name)):
            # Public raw / pilot outputs are not indexed: a hit there could only be a warning, and they are the
            # bulk of a session push. Withheld-set raw files carry no text by construction, so they are.
            ctx.pending.append((display, erel, commit, words, line_of))
        hits = ctx.shingles.matches(words) if ctx.shingles else []
        n_shingles = sum(1 for h in hits if h["kind"] == "shingle")
        for hit in hits:
            if hit["kind"] == "shingle":
                sev = "finding" if n_shingles >= MIN_SHINGLE_HITS else "warning"
                detail = (f"matches withheld shingle hash {hit['hash']} ({n_shingles} in this text; text not shown)")
            else:
                sev = "warning" if (raw and hit["words"] < MIN_OPTION_WORDS_RAW) else "finding"
                detail = f"matches a withheld {hit['words']}-word option, hash {hit['hash']} (text not shown)"
            out.append(Finding("withheld_" + hit["kind"], display, line_of(hit["word_index"]), detail, sev, commit))
    # One line per location and severity is enough.
    seen, uniq = set(), []
    for f in out:
        key = (f.kind, f.line, f.severity)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def local_corpus_pass(ctx: Context, res: "Result") -> None:
    """Match the pending text units against the local corpora (RGB documents, FineWeb-2 rows) in one stream.

    The scanned text's own 8-word shingles are indexed (small); the corpora are streamed once and every
    window is looked up. Public kit text and all-digit shingles are never hits. Severity as for withheld
    shingles (>= 2 hits in one unit is a finding, 1 a warning), also in withheld-set raw / pilot files; public raw
    and pilot outputs are not indexed (scan_withheld)."""
    if ctx.local_data is None or not ctx.pending:
        return
    public = shingles.public_shingle_strings(ctx.exp_dir)
    index: dict[str, list[tuple[int, int]]] = {}
    for u, (_d, _e, _c, words, _l) in enumerate(ctx.pending):
        for i, sh in shingles.window_strings(words):
            if sh not in public and not shingles.all_digits(sh):
                index.setdefault(sh, []).append((u, i))
    if not index:
        return
    matched: dict[str, str] = {}
    sources: list[str] = []
    for name, text in shingles.local_corpus_texts(ctx.local_data, ctx.exp_dir, res.source_warnings, sources):
        for _, sh in shingles.window_strings(shingles.normalise_words(text)):
            if sh in index and sh not in matched:
                matched[sh] = name
    res.sources += [f"local corpus: {s}" for s in sources]
    per_unit: dict[int, list[tuple[int, str, str]]] = {}
    for sh, src in matched.items():
        for u, i in index[sh]:
            per_unit.setdefault(u, []).append((i, src, shingles._h(sh)[:shingles.PREFIX_HEX]))
    for u, hits in sorted(per_unit.items()):
        display, erel, commit, _w, line_of = ctx.pending[u]
        hits.sort()
        n = len(hits)
        sev = "finding" if n >= MIN_SHINGLE_HITS else "warning"
        i, src, h = hits[0]
        res.add(Finding("local_corpus", display, line_of(i),
                        f"{n} 8-word shingle(s) of {src} in this text, first hash {h} (text not shown)", sev, commit))


def _walk_withheld(obj, withheld: bool, path: str, out: list) -> None:
    if isinstance(obj, dict):
        task = obj.get("task")
        if task is None and isinstance(obj.get("key"), dict):
            task = obj["key"].get("task")
        for k in ("set", "dataset"):
            if task is None and isinstance(obj.get(k), str):
                task = obj[k]
        here = withheld or (isinstance(task, str) and bool(WITHHELD_TASK.search(task)))
        for k, v in obj.items():
            sub = here or bool(WITHHELD_TASK.search(str(k)))
            if here and k in WITHHELD_FIELDS and v not in (None, "", [], {}):
                out.append(f"{path}.{k}" if path else str(k))
            _walk_withheld(v, sub, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_withheld(v, withheld, f"{path}[{i}]", out)


def scan_structure(data: bytes, display: str, erel: str | None, ctx: Context, commit=None) -> list[Finding]:
    """Withheld fields in machine-written JSON / JSONL records."""
    if not (erel and ctx.is_record(erel) and erel.endswith(JSON_SUFFIXES)):
        return []
    file_withheld = bool(WITHHELD_TASK.search(Path(erel).name))
    text = data.decode("utf-8", errors="replace")
    docs: list[tuple[int | None, object]] = []
    if erel.endswith(".jsonl"):
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    docs.append((i, json.loads(line)))
                except json.JSONDecodeError:
                    pass
    else:
        try:
            docs.append((None, json.loads(text)))
        except json.JSONDecodeError:
            return []
    out = []
    for ln, doc in docs:
        hits: list[str] = []
        _walk_withheld(doc, file_withheld, "", hits)
        for h in hits[:5]:
            out.append(Finding("withheld_field", display, ln, f"withheld-set record carries `{h}`", "finding", commit))
    return out


def scan_file_meta(size: int, display: str, erel: str | None, ctx: Context, commit=None) -> list[Finding]:
    out = []
    name = (erel or display).lower()
    if name.endswith(REFUSED_SUFFIXES):
        out.append(Finding("refused_type", display, None, f"{Path(name).suffix} files are never committed", "finding", commit))
    elif name.endswith(ARCHIVE_SUFFIXES):
        out.append(Finding("refused_type", display, None, f"{Path(name).suffix} archive or binary format: its content "
                           "cannot be checked", "finding" if erel is not None else "warning", commit))
    limit = MAX_BYTES_RAW if (erel or "").startswith(LARGE_DIRS) else MAX_BYTES
    if size > limit:
        out.append(Finding("too_large", display, None, f"{size:,} bytes > {limit:,}", "finding", commit))
    return out


def check_shingle_file(data: bytes, display: str, commit=None) -> list[Finding]:
    bad = []
    for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        s = line.strip()
        if s and not s.startswith("#") and not re.fullmatch(r"[0-9a-f]{12}|[0-9a-f]{64}", s):
            bad.append(i)
    if bad:
        return [Finding("shingle_file_format", display, bad[0],
                        f"{len(bad)} lines are neither hex hashes nor comments", "finding", commit)]
    return []


def scan_blob(data: bytes, display: str, erel: str | None, ctx: Context,
              added: list | None = None, commit=None) -> list[Finding]:
    """All rules for one file version. `added` = [(line_no, text)] restricts the
    text rules to those lines (range / staged mode)."""
    if ctx.allowlisted(erel):
        return []
    out = scan_file_meta(len(data), display, erel, ctx, commit)
    if erel == SHINGLE_FILE_REL:
        return out + check_shingle_file(data, display, commit)
    try:
        strict_text = None if b"\0" in data else data.decode("utf-8")
    except UnicodeDecodeError:
        strict_text = None
    if strict_text is None:
        # NUL bytes or not strict UTF-8 (UTF-16, compressed, binary): the text rules cannot read it (review fix
        # 2026-10-03). Inside the experiment a finding; outside it a warning.
        if erel is not None and erel not in BINARY_ALLOWLIST:
            out.append(Finding("binary_file", display, None, "binary or non-UTF-8 file; never committed", "finding", commit))
        elif erel is None and b"\0" in data[:8192]:
            out.append(Finding("binary_file", display, None, "binary file outside the experiment; not text-checked",
                               "warning", commit))
            return out
        if erel is not None:
            return out
    if added is None:
        text = data.decode("utf-8", errors="replace")
        lines = list(enumerate(text.splitlines(), 1))
    else:
        lines = added
    is_json = (erel or display).endswith(JSON_SUFFIXES)
    out += scan_lines(lines, display, erel, ctx, commit)
    out += scan_withheld(lines, display, ctx, is_json, commit, erel)
    out += scan_structure(data, display, erel, ctx, commit)
    return out


# --------------------------------------------------------------------------
# File sources: paths, --all, staged, ranges
# --------------------------------------------------------------------------

def _git_bytes(args, cwd) -> bytes | None:
    import subprocess

    try:
        p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


_DIFF_GIT = re.compile(r"^diff --git a/(.*) b/(.*)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_patch(text: str):
    """Yield (commit, path, [(line_no, added_text)]) from `git log -p -U0`
    or `git diff -p -U0` output (commit is None for a plain diff)."""
    commit = None
    path = None
    added: list = []
    new_line = 0
    in_hunk = False

    def flush():
        if path is not None:
            yield commit, path, added

    for line in text.split("\n"):
        if line.startswith("\x00commit "):
            yield from flush()
            path, added, in_hunk = None, [], False
            commit = line.split()[1]
            continue
        m = _DIFF_GIT.match(line)
        if m:
            yield from flush()
            path, added, in_hunk = m.group(2), [], False
            continue
        if path is None:
            continue
        if not in_hunk:
            # File header lines; an added line "++ x" inside a hunk looks like
            # "+++ x", so headers are only read before the first hunk.
            if line.startswith("+++ "):
                target = line[4:]
                if target != "/dev/null" and target.startswith("b/"):
                    path = target[2:]
                continue
            if line.startswith("--- "):
                continue
        h = _HUNK.match(line)
        if h:
            new_line = int(h.group(1))
            in_hunk = True
            continue
        if in_hunk and line.startswith("+"):
            added.append((new_line, line[1:]))
            new_line += 1
    yield from flush()


def _rev_args_from(ranges, new_tips) -> list[list[str]]:
    out = [[r] for r in ranges]
    out += [[t, "--not", "--remotes"] for t in new_tips]
    return out


def iter_range(repo: Path, rev_args: list[str]):
    """(commit, repo_rel_path, added_lines, blob_bytes) for every file a commit
    in the range adds or modifies."""
    patch = _git_bytes(
        ["-c", "core.quotepath=off", "log", "--no-color", "--no-ext-diff", "--format=%x00commit %H",
         "-p", "-U0", "-M", "--diff-filter=ACMR", *rev_args],
        repo,
    )
    if patch is None:
        raise RuntimeError(f"git log failed for {' '.join(rev_args)}")
    for commit, path, added in parse_patch(patch.decode("utf-8", errors="replace")):
        blob = _git_bytes(["cat-file", "blob", f"{commit}:{path}"], repo)
        if blob is None:
            continue
        yield commit, path, added, blob


def iter_staged(repo: Path):
    patch = _git_bytes(
        ["-c", "core.quotepath=off", "diff", "--cached", "--no-color", "--no-ext-diff",
         "-p", "-U0", "-M", "--diff-filter=ACMR"],
        repo,
    )
    if patch is None:
        raise RuntimeError("git diff --cached failed")
    for _, path, added in parse_patch(patch.decode("utf-8", errors="replace")):
        blob = _git_bytes(["cat-file", "blob", f":{path}"], repo)
        if blob is not None:
            yield path, added, blob


def _list_dir(d: Path) -> list[Path]:
    from tools import hash_tree

    return [d / r for r in hash_tree.list_files(d)]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def check(paths=(), staged: bool = False, ranges=(), new_tips=(), all_files: bool = False,
          no_gpqa_source: bool = False, exp_dir=None, ctx: Context | None = None) -> Result:
    """Run the leak check; raises NoWithheldSource (exit 3) when required."""
    res = Result()
    if ctx is None:
        sset, swarn = load_withheld(exp_dir, allow_missing=no_gpqa_source)
        ctx = Context(exp_dir, sset)
        res.source_warnings = swarn
        if not ctx.enable_local(common.data_dir()):
            res.source_warnings.append("local corpora (RGB documents, FineWeb-2 rows) not found under $EXP036_DATA: "
                                       "their check did not run")
    res.sources = list(ctx.shingles.sources)
    res.withheld_check = "on" if (ctx.shingles or ctx.local_data is not None) else "skipped (--no-gpqa-source)"

    def handle(display, erel, data, added=None, commit=None):
        res.n_files += 1
        for f in scan_blob(data, display, erel, ctx, added=added, commit=commit):
            res.add(f)

    for p in paths:
        p = Path(p)
        files = _list_dir(p) if p.is_dir() else [p]
        for f in files:
            erel = ctx.exp_rel(None, abs_path=f)
            handle(erel or str(f), erel, f.read_bytes())
    if all_files:
        for f in _list_dir(ctx.exp_dir):
            erel = ctx.exp_rel(None, abs_path=f)
            handle(erel, erel, f.read_bytes())
    if (staged or ranges or new_tips) and ctx.repo is None:
        raise RuntimeError("range / staged mode needs a git repository")
    if staged:
        for path, added, blob in iter_staged(ctx.repo):
            handle(path, ctx.exp_rel(path), blob, added=added, commit="index")
    for rev_args in _rev_args_from(ranges, new_tips):
        for commit, path, added, blob in iter_range(ctx.repo, rev_args):
            handle(path, ctx.exp_rel(path), blob, added=added, commit=commit)
    local_corpus_pass(ctx, res)
    return res


def _resolve_ranges(ranges: list[str], repo: Path | None) -> tuple[list[str], list[str], list[str]]:
    """Replace a range whose base does not resolve (no upstream) by
    HEAD --not --remotes. Returns (ranges, new_tips, notes)."""
    out, tips, notes = [], [], []
    for r in ranges:
        base = r.split("..", 1)[0] if ".." in r else None
        if base and repo is not None and common.git_out(["rev-parse", "--verify", "-q", base], cwd=repo) is None:
            tip = r.split("..", 1)[1] or "HEAD"
            notes.append(f"{base} does not resolve; checking {tip} --not --remotes instead")
            tips.append(tip)
        else:
            out.append(r)
    return out, tips, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_037 leak check (exp_036 BUILD_SPEC §5.9)")
    ap.add_argument("paths", nargs="*", help="files or directories to check")
    ap.add_argument("--range", dest="ranges", action="append", default=[], metavar="REV..REV")
    ap.add_argument("--new-tip", dest="new_tips", action="append", default=[], metavar="SHA")
    ap.add_argument("--staged", action="store_true", help="also check the index")
    ap.add_argument("--all", dest="all_files", action="store_true", help="every file of the experiment dir")
    ap.add_argument("--no-gpqa-source", action="store_true",
                    help="allow running without any withheld-text source (pre-registration push only)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    if not (args.paths or args.ranges or args.new_tips or args.staged or args.all_files):
        ap.print_usage(sys.stderr)
        return 2

    repo = common.repo_root(common.EXP_DIR)
    ranges, tips, notes = _resolve_ranges(args.ranges, repo)
    for n in notes:
        print(f"leak_check: {n}", file=sys.stderr)
    try:
        res = check(args.paths, args.staged, ranges, tips + args.new_tips, args.all_files, args.no_gpqa_source)
    except NoWithheldSource as e:
        print(f"leak_check: {e}. Exit 3 (pass --no-gpqa-source only for the pre-registration push).",
              file=sys.stderr)
        return 3
    except RuntimeError as e:
        print(f"leak_check: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "ok": res.ok, "n_files": res.n_files, "withheld_check": res.withheld_check,
            "sources": res.sources, "source_warnings": res.source_warnings,
            "findings": [asdict(f) for f in res.findings], "warnings": [asdict(f) for f in res.warnings],
        }, indent=2, sort_keys=True))
    else:
        for f in res.findings:
            print(f.fmt())
        for f in res.warnings:
            print(f.fmt())
        for w in res.source_warnings:
            print(f"note: {w}")
        src = "; ".join(res.sources) if res.sources else "none"
        print(f"leak_check: {res.n_files} file versions, {len(res.findings)} findings, "
              f"{len(res.warnings)} warnings; withheld-text check {res.withheld_check} (sources: {src})")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
