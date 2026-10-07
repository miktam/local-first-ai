"""CPU + GPU + ANE power from powermetrics, joined to the run's windows (descriptive only).

Andrei's request, 2026-10-06 ("also add powermetrics"): exp_037's machine energy is
measured on the mbp, where exp_036's was only estimated (time x an assumed wall draw,
exp_036 compute/COMPUTE.md). HYPOTHESIS "Power and energy (descriptive)" registers what
this measures and that it decides nothing: no hypothesis, verdict, gate rule, threshold
or plan rule reads a power record, and a missing or partial log blocks nothing.

**What it measures** (LABEL): CPU + GPU + ANE power, powermetrics' `combined_power`, an
estimate. The powermetrics man page lists CPU, GPU and ANE among the "SoC subsystems" whose
power it estimates; their sum is not the whole SoC (the memory controllers and fabric, the
system-level cache, the media and display engines and I/O are not in it, nor is DRAM) and
not wall power. The record's key for it is `combined`. It is a lower bound of the
machine's energy, and a loose one (CAVEATS).

The logger (RUNBOOK, "Power log"; Andrei's second terminal, before the step starts):

    sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist \
        > "$EXP036_WORK/exp037/power/power_<UTC>.plist"

The mbp's sudoers rule pins exactly that command line (COMMAND below); any other
argument, -o and -a included, is refused by sudo. Andrei's shell does the redirect, and
Ctrl-C in the logger's own terminal stops it after the step.

CLI:
    power_log.py [RAW ...] [--window NAME START END]... [--results DIR] [--out DIR]
                 [--no-auto] [--again]
        writes results/power/power_<UTC>.json from the raw log(s) (default: the newest
        $EXP036_WORK/exp037/power/power_<UTC>.plist; a power_smoke_* file is never the
        default); requires the git identity
    power_log.py --check [RAW] [--max-age SECONDS]
        is the raw log readable and current? One JSON line; writes nothing
    power_log.py --layout [RAW]
        the key names and plist types of the documents' top level, `processor` and `gpu`
        dicts, grouped by layout, with each layout's document and sample counts and its
        elapsed times (median, max); no other value. One JSON line; writes nothing. It is
        the pre-freeze smoke's dump (RUNBOOK "Power log")

START / END: ISO UTC (2026-10-07T05:01:12Z), the kit's stamp (20261007T050112Z) or
epoch seconds. Exit codes: 0 ok, 1 refused or check failed, 2 usage.

**Raw log.** `-f plist` writes one XML plist document per sample, NUL-separated (man
page). Text before a document's `<?xml` (or `<!DOCTYPE plist` / `<plist`), and a complete
chunk with no plist in it, is skipped and counted (`text_bytes_skipped`). A document cut
short by Ctrl-C (the last one, with no NUL after it) is reported as `truncated_tail` and
left out; a complete document that does not parse is counted as `unreadable`. Read per
sample, and only these:
- `is_delta` (a sample with is_delta false is skipped);
- `elapsed_ns`, an integer or a real;
- `timestamp`;
- per component c in cpu, gpu, ane: `processor.<c>_power` in mW, else
  `processor.<c>_energy` in mJ / elapsed (counted as `from_energy`); the GPU's also from
  the `gpu` dict when the processor dict lacks it;
- `combined` = `processor.combined_power` in mW, or cpu + gpu + ane when it is absent
  (counted as `combined_from_components`).
Every other field is dropped at parse time. Until the pre-freeze smoke has shown the mbp's
real layout, these names are assumptions; a log in which documents parse but give no
sample makes --check say "key layout differs" and --layout shows what the log holds.

**Never written:** the raw log itself (never committed or copied into the repository;
inside the experiment directory the leak check refuses a file with NUL bytes, and this
tool refuses a raw path inside the repository) and its host fields: `kern_bootargs`,
`kern_boottime`, `kern_osversion` and `hw_model` (HOST_KEYS). The one derived host fact
is `model_family`, the part of `hw_model` before the comma (`Mac17` from `Mac17,7`). It
names a product line shared by every unit of it, not a machine, and is less than what
the preflight record already carries (its whitelisted `machine.hw_model`); it tells an
mbp log from a mini one. The raw file is recorded by name, redacted directory,
sha256 and byte size. --layout prints key names, never their values.

**Method.** A sample covers [timestamp - elapsed, timestamp] (its timestamp is taken as
the end of its interval; the plist date has whole-second resolution, so a placement error
is under 1 s and matters only at a window's edges). Per window:
- energy_wh: sum over samples of power x the overlap of the sample's interval with the
  window (`combined` and each component), in Wh;
- mean_w: that energy / the sampled time inside the window (no extrapolation over gaps);
- peak_w: the largest per-sample power among the samples that overlap the window;
- sampled_s: the summed overlaps; sampled_s <= duration_s is expected;
- coverage: the union of the sample intervals inside the window / the window's length;
- overlap_s: sampled_s - covered_s, the time that more than one sample counts. Whole-second
  timestamps make it a few ms per sample (elapsed minus the 1 s step between two
  timestamps; the energy of each sample is still counted once);
- gaps_over_5s: every uncovered stretch inside the window longer than 5 s (edges too).
Windows come from the run records the kit already writes (discover_windows), plus
--window and the whole log ("log"). A discovered window that does not overlap the log
is left out (counted). A run that crashed before writing its closing time has no
window: give it with --window.

**Double counting** (per raw file). excess_s = the sum of the samples' elapsed times - the
span from the first sample's start to the last one's end. A clean stream gives about 0
(within the timestamps' 1 s placement at each end) whatever its per-sample jitter; a
document that covers time other documents already sampled adds its elapsed again: the
"poweravg" summaries (`-a`, "Display poweravg every N samples", default 10, which the
pinned command cannot turn off; how they appear in -f plist is unknown until the smoke)
or duplicates. When excess_s > 2 % of the span + 2 s the tool writes no record and --check
fails. A document whose elapsed is over 3 x the file's median is "long": reported, and
--check fails on it (a poweravg summary, or a sleep); a record keeps it, because a sample
that spans a sleep is real time, counted once.

stdlib only; no network.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import hashlib
import json
import math
import plistlib
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common
from tools.redact import redact_path

SCHEMA = "exp037 power v1"
LABEL = "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power"
# The mbp's sudoers rule (Andrei, 2026-10-06T14:43:27Z) pins exactly this command line; sudo refuses any other.
COMMAND = "/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist"
SAMPLERS = ("cpu_power", "gpu_power", "ane_power")
INTERVAL_MS = 1000
GAP_S = 5.0                  # a gap is an uncovered stretch longer than this
MAX_GAPS_LISTED = 20         # per window; the count and the total cover all of them
HOST_KEYS = ("hw_model", "kern_osversion", "kern_bootargs", "kern_boottime")  # never written
RAW_SUBDIR = ("exp037", "power")   # under $EXP036_WORK
RAW_GLOB = "power_*.plist"
OUT_SUBDIR = ("results", "power")
CHECK_MAX_AGE_S = 120.0      # stdout to a file is block-buffered; samples reach the file in bursts
CHECK_MIN_SAMPLES = 2
RECENT_S = 30.0              # record mode: a newer last sample suggests the logger still runs (a warning only)
INTERVAL_OK_MS = (500.0, 2000.0)
LONG_FACTOR = 3.0            # a document longer than this x the median interval is "long"
DOUBLE_COUNT_FRAC = 0.02     # excess_s above this share of the span ...
DOUBLE_COUNT_SLACK_S = 2.0   # ... plus this (the first and last timestamps' whole-second placement) is refused
CHUNK = 1 << 20
COMPONENTS = ("combined", "cpu", "gpu", "ane")
_PLIST_MARKS = (b"<?xml", b"<!DOCTYPE plist", b"<plist")

NOT_MEASURED = [
    "the rest of the SoC: the memory controllers and fabric, the system-level cache (SLC), the media and display "
    "engines, and I/O",
    "DRAM (the unified memory itself), which bandwidth-bound decode loads heavily",
    "wall (AC) power and the power adapter's conversion losses",
    "battery charging",
    "the display",
    "the SSD",
    "fans and the rest of the board (I/O controllers, USB, Wi-Fi)",
]
DESCRIPTIVE = "Descriptive only: no hypothesis, verdict, gate rule, threshold or plan rule reads this record."
CAVEATS = [
    "combined is CPU + GPU + ANE only, the 'SoC subsystems' whose power the powermetrics man page says it estimates; "
    "it is not the whole SoC and not wall power.",
    "It is a lower bound of the machine's energy, and a loose one that may be about half of it. For scale only, from "
    "a different machine: on the mini (Apple M4 Pro; macmon spot reading, 2026-10-06, MLX GPU load) CPU + GPU power "
    "read 28.6 W against a system total of 64.6 W, about 44 %. No factor converts it to the whole SoC or to wall "
    "energy here.",
    "The powermetrics man page: average power values 'are estimated and may be inaccurate' and 'should not be used "
    "for any comparison between devices'; this record compares nothing across machines.",
    "The logger adds a small background load (one root sampler at 1 Hz) to every logged step alike; it is not "
    "measured or corrected for.",
    DESCRIPTIVE,
]
METHOD = {
    "combined": "processor.combined_power (mW): CPU + GPU + ANE, not the whole SoC (memory controllers and fabric, "
                "SLC, media and display engines, I/O and DRAM are not in it) and not wall power; cpu + gpu + ane "
                "when the key is absent (counted as combined_from_components)",
    "components": "processor.<c>_power (mW), else processor.<c>_energy (mJ) / elapsed (counted as from_energy); "
                  "the GPU's also from the gpu dict",
    "sample_interval": "[timestamp - elapsed_ns, timestamp]; elapsed_ns an integer or a real; the timestamp is taken "
                       "as the end of the sample's interval (plist dates have whole-second resolution)",
    "energy_wh": "sum over samples of power x overlap(sample interval, window), in Wh",
    "mean_w": "energy / sampled time inside the window; nothing is extrapolated over gaps",
    "peak_w": "largest per-sample power among the samples that overlap the window",
    "sampled_s": "sum of the samples' overlaps with the window; sampled_s <= duration_s is expected",
    "coverage": "union of the sample intervals inside the window / window length",
    "overlap_s": "sampled_s - covered_s, time counted by more than one sample: a few ms per sample from whole-second "
                 "timestamps (each sample's energy is still counted once); a raw log whose documents overlap is "
                 "refused (double_count)",
    "double_count": "per raw file, excess_s = sum of elapsed - span (first start to last end); about 0 for a clean "
                    "stream; above 2 % of the span + 2 s (poweravg summaries or duplicates) no record is written",
    "long_samples": "documents whose elapsed is over 3 x the file's median (a poweravg summary or a sleep): counted; "
                    "--check fails on them, a record keeps them",
    "gaps_over_5s": "uncovered stretches inside the window longer than 5 s, window edges included",
    "windows": "t_start / t_end of the run records under results/ (gate and its phases, peer check, preflight "
               "--deep, clone-and-refresh, bench cells and their cool-down idle, pilot and its arms, S2/S3/S3b "
               "session start/stop pairs), --window, and the whole log; a discovered window that does not "
               "overlap the log is left out",
}

_STAMP = re.compile(r"^\d{8}T\d{6}Z$")
_RAW_NAME = re.compile(r"^power_\d{8}T\d{6}Z\.plist$")    # the RUNBOOK's logger file; never power_smoke_*
_GATE_RECORD = re.compile(r"^gate_(\d{8}T\d{6}Z)\.json$")
_FAMILY = re.compile(r"^([A-Za-z]+\d+),\d+$")


class Refused(RuntimeError):
    """A refusal with a message for the operator (exit 1)."""


# ------------------------------------------------------------------------------------------------ time


def parse_time(v) -> float:
    """Epoch seconds (UTC) from an ISO UTC string, the kit's stamp, epoch seconds or a datetime."""
    if isinstance(v, bool):
        raise ValueError(f"not a time: {v!r}")
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            raise ValueError(f"not a time: {v!r}")
        return float(v)
    if isinstance(v, _dt.datetime):
        t = v if v.tzinfo else v.replace(tzinfo=_dt.timezone.utc)
        return t.timestamp()
    if not isinstance(v, str):
        raise ValueError(f"not a time: {v!r}")
    s = v.strip()
    if _STAMP.match(s):
        return _dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    t = _dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.timestamp()


def iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------- raw log


@dataclass
class Sample:
    start: float
    end: float
    combined: float       # W
    cpu: float | None     # W
    gpu: float | None
    ane: float | None


@dataclass
class RawLog:
    path: Path
    sha256: str = ""
    bytes: int = 0
    documents: int = 0
    unreadable: int = 0
    truncated_tail: bool = False
    text_bytes_skipped: int = 0
    skipped: dict = field(default_factory=lambda: {"not_delta": 0, "no_time": 0, "no_power": 0})
    combined_from_components: int = 0
    from_energy: int = 0
    seen: set = field(default_factory=set)
    model_family: str | None = None
    samples: list = field(default_factory=list)
    layouts: dict = field(default_factory=dict)   # key-name/type signature -> counts and elapsed times (--layout)

    @property
    def parsed(self) -> int:
        """Complete documents that parse (whether or not they give a sample)."""
        return self.documents - self.unreadable - int(self.truncated_tail)


def _mw(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v >= 0 else None


def _elapsed_s(v) -> float | None:
    """elapsed_ns (an integer or a real) in seconds; None unless finite and > 0."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v / 1e9 if math.isfinite(v) and v > 0 else None


def _component(proc: dict, gpu_d: dict, name: str, el_s: float) -> tuple[float | None, bool]:
    """(W, from_energy) of one component: <name>_power (mW), else <name>_energy (mJ) / elapsed; the GPU's also from
    the gpu dict when the processor dict lacks it."""
    dicts = (proc, gpu_d) if name == "gpu" else (proc,)
    for d in dicts:
        v = _mw(d.get(f"{name}_power"))
        if v is not None:
            return v / 1000.0, False
    for d in dicts:
        v = _mw(d.get(f"{name}_energy"))
        if v is not None:
            return v / el_s / 1000.0, True
    return None, False


def _take(raw: RawLog, doc: dict) -> tuple[float | None, bool]:
    """One sample from one parsed document; only the whitelisted fields are read. Returns (elapsed s, sampled)."""
    if raw.model_family is None:
        m = _FAMILY.match(str(doc.get("hw_model") or ""))
        if m:
            raw.model_family = m.group(1)
    el = _elapsed_s(doc.get("elapsed_ns"))
    if doc.get("is_delta") is False:
        raw.skipped["not_delta"] += 1
        return el, False
    try:
        end = parse_time(doc.get("timestamp"))
    except (ValueError, TypeError, OverflowError):
        end = None
    if end is None or el is None:
        raw.skipped["no_time"] += 1
        return el, False
    proc = doc.get("processor") if isinstance(doc.get("processor"), dict) else {}
    gpu_d = doc.get("gpu") if isinstance(doc.get("gpu"), dict) else {}
    vals, n_energy = {}, 0
    for c in ("cpu", "gpu", "ane"):
        vals[c], fe = _component(proc, gpu_d, c, el)
        n_energy += fe
    combined = _mw(proc.get("combined_power"))
    derived = combined is None and None not in vals.values()
    if combined is not None:
        combined /= 1000.0
    elif derived:
        combined = vals["cpu"] + vals["gpu"] + vals["ane"]
    if combined is None:
        raw.skipped["no_power"] += 1
        return el, False
    raw.combined_from_components += int(derived)
    raw.from_energy += n_energy
    for c, v in vals.items():
        if v is not None:
            raw.seen.add(f"{c}_power")
    raw.samples.append(Sample(end - el, end, combined, vals["cpu"], vals["gpu"], vals["ane"]))
    return el, True


def _ptype(v) -> str:
    """The plist type name of a parsed value (never the value)."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "real"
    if isinstance(v, str):
        return "string"
    if isinstance(v, _dt.datetime):
        return "date"
    if isinstance(v, (bytes, bytearray)):
        return "data"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        return "array"
    return "other"


def _signature(doc: dict) -> tuple:
    """Key names and types of the top level and the processor and gpu dicts: (where, name, type), sorted."""
    out = [("top", str(k), _ptype(v)) for k, v in doc.items()]
    for sub in ("processor", "gpu"):
        d = doc.get(sub)
        if isinstance(d, dict):
            out += [(sub, str(k), _ptype(v)) for k, v in d.items()]
    return tuple(sorted(out))


def _plist_start(s: bytes) -> int:
    hits = [i for i in (s.find(m) for m in _PLIST_MARKS) if i >= 0]
    return min(hits) if hits else -1


def _document(raw: RawLog, chunk: bytes, tail: bool) -> None:
    s = chunk.strip()
    if not s:
        return
    pos = _plist_start(s)
    if pos < 0 and not tail:
        raw.text_bytes_skipped += len(s)    # a complete chunk with no plist in it
        return
    if pos > 0:
        raw.text_bytes_skipped += pos       # text before the document
        s = s[pos:]
    raw.documents += 1
    try:
        doc = plistlib.loads(s, fmt=plistlib.FMT_XML)
        if not isinstance(doc, dict):
            raise ValueError("not a dict")
    except Exception:
        if tail:
            raw.truncated_tail = True   # Ctrl-C mid-write: the last document is cut short
        else:
            raw.unreadable += 1
        return
    el, sampled = _take(raw, doc)
    lay = raw.layouts.setdefault(_signature(doc), {"documents": 0, "samples": 0, "elapsed": []})
    lay["documents"] += 1
    lay["samples"] += int(sampled)
    if el is not None:
        lay["elapsed"].append(el)


def read_raw(path) -> RawLog:
    """Stream a NUL-separated powermetrics plist log: sha256, size and the samples."""
    raw = RawLog(Path(path))
    h = hashlib.sha256()
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            raw.bytes += len(block)
            buf += block
            parts = buf.split(b"\0")
            buf = parts.pop()
            for p in parts:
                _document(raw, p, tail=False)
    _document(raw, buf, tail=True)
    raw.sha256 = h.hexdigest()
    raw.samples.sort(key=lambda s: (s.end, s.start))
    return raw


def time_stats(raw: RawLog) -> dict:
    """Span, summed elapsed, union, overlap, excess and long documents of one raw file (seconds)."""
    ss = raw.samples
    if not ss:
        return {"first": None, "last": None, "span_s": 0.0, "elapsed_sum_s": 0.0, "covered_s": 0.0,
                "overlap_s": 0.0, "excess_s": 0.0, "median_ms": None, "min_ms": None, "max_ms": None,
                "long_n": 0, "long_max_ms": None}
    el = [s.end - s.start for s in ss]
    lo, hi = min(s.start for s in ss), max(s.end for s in ss)
    covered, cur = 0.0, -math.inf
    for a, b in sorted((s.start, s.end) for s in ss):
        if b > cur:
            covered += b - max(a, cur)
            cur = b
    total = math.fsum(el)
    med = statistics.median(el)
    long_ = [x for x in el if x > LONG_FACTOR * med]
    return {"first": lo, "last": hi, "span_s": hi - lo, "elapsed_sum_s": total, "covered_s": covered,
            "overlap_s": max(0.0, total - covered), "excess_s": total - (hi - lo),
            "median_ms": med * 1000.0, "min_ms": min(el) * 1000.0, "max_ms": max(el) * 1000.0,
            "long_n": len(long_), "long_max_ms": max(long_) * 1000.0 if long_ else None}


def double_counted(st: dict) -> bool:
    """Do the documents of one raw file count the same time twice (beyond whole-second placement)?"""
    return st["excess_s"] > DOUBLE_COUNT_FRAC * st["span_s"] + DOUBLE_COUNT_SLACK_S


def _double_count_text(name: str, st: dict) -> str:
    return (f"{name}: its documents overlap: their elapsed times sum to {st['elapsed_sum_s']:.0f} s over a "
            f"{st['span_s']:.0f} s span, so {st['excess_s']:.0f} s would be counted twice (powermetrics' poweravg "
            f"summaries, -a, default every 10 samples, or duplicate documents)")


def _r(x: float | None, nd: int) -> float | None:
    return None if x is None else round(x, nd)


def raw_summary(raw: RawLog) -> dict:
    st = time_stats(raw)
    return {
        "name": raw.path.name,
        "dir": redact_path(raw.path.resolve().parent),
        "sha256": raw.sha256,
        "bytes": raw.bytes,
        "documents": raw.documents,
        "samples": len(raw.samples),
        "unreadable": raw.unreadable,
        "truncated_tail": raw.truncated_tail,
        "text_bytes_skipped": raw.text_bytes_skipped,
        "skipped": dict(raw.skipped),
        "combined_from_components": raw.combined_from_components,
        "from_energy": raw.from_energy,
        "first_sample_utc": iso(st["first"]) if st["first"] is not None else None,
        "last_sample_utc": iso(st["last"]) if st["last"] is not None else None,
        "span_s": round(st["span_s"], 3),
        "elapsed_sum_s": round(st["elapsed_sum_s"], 3),
        "covered_s": round(st["covered_s"], 3),
        "overlap_s": round(st["overlap_s"], 3),
        "excess_s": round(st["excess_s"], 3),
        "long_samples": {"n": st["long_n"], "max_ms": _r(st["long_max_ms"], 3)},
        "samplers_seen": sorted(raw.seen),
        "interval_ms": {"declared": INTERVAL_MS, "median": _r(st["median_ms"], 3), "min": _r(st["min_ms"], 3),
                        "max": _r(st["max_ms"], 3)},
        "model_family": raw.model_family,
    }


def layout_report(raw: RawLog) -> dict:
    """--layout: key names and plist types per layout, with counts and elapsed times; never a field's value."""
    lays = []
    for sig, v in sorted(raw.layouts.items(), key=lambda kv: (-kv[1]["documents"], kv[0])):
        keys: dict[str, dict[str, str]] = {"top": {}, "processor": {}, "gpu": {}}
        for where, name, typ in sig:
            keys[where][name] = typ
        el = v["elapsed"]
        lays.append({"documents": v["documents"], "samples": v["samples"],
                     "elapsed_ms": ({"median": round(statistics.median(el) * 1000.0, 3),
                                     "max": round(max(el) * 1000.0, 3)} if el else None),
                     "keys": keys})
    return {"file": raw.path.name, "dir": redact_path(raw.path.resolve().parent), "bytes": raw.bytes,
            "documents": raw.documents, "parsed": raw.parsed, "samples": len(raw.samples),
            "unreadable": raw.unreadable, "truncated_tail": raw.truncated_tail,
            "text_bytes_skipped": raw.text_bytes_skipped, "skipped": dict(raw.skipped),
            "from_energy": raw.from_energy, "combined_from_components": raw.combined_from_components,
            "layouts": lays,
            "note": "key names and plist types only; the one value shown is each layout's elapsed time"}


# ------------------------------------------------------------------------------------------- windows


@dataclass
class Window:
    name: str
    kind: str
    start: float
    end: float
    source: str | None = None


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _win(name: str, kind: str, t0, t1, source: str | None) -> Window | None:
    """A window, or None when a time is missing or unreadable or the window is empty."""
    try:
        a, b = parse_time(t0), parse_time(t1)
    except (ValueError, TypeError, OverflowError):
        return None
    return Window(name, kind, a, b, source) if b > a else None


def _rel(p: Path, results: Path) -> str:
    try:
        return "results/" + p.relative_to(results).as_posix()
    except ValueError:
        return p.name


def _session_windows(sdir: Path, results: Path) -> list[Window]:
    """Start/stop pairs of results/raw/<S>/session.jsonl. Session-level events (no "cell") pair with each other;
    `run.py cell` events pair by cell. A start without a stop (a crash) ends at the session's heartbeat when that
    lies after it and before the next start; otherwise it has no window."""
    p = sdir / "session.jsonl"
    src = _rel(p, results)
    hb = _load_json(sdir / ".heartbeat")
    hb_t = None
    if isinstance(hb, dict) and hb.get("utc"):
        try:
            hb_t = parse_time(hb["utc"])
        except (ValueError, TypeError):
            hb_t = None
    out: list[Window] = []
    open_: dict[str | None, float] = {}
    n_runs = 0

    def close(key, t0, t1):
        nonlocal n_runs
        if key is None:
            n_runs += 1
            w = _win(f"{sdir.name} run {n_runs}", "session", t0, t1, src)
        else:
            w = _win(f"{sdir.name} cell {key}", "session_cell", t0, t1, src)
        if w:
            out.append(w)

    for r in _jsonl(p):
        if r.get("type") not in ("start", "stop") or not r.get("utc"):
            continue
        try:
            t = parse_time(r["utc"])
        except (ValueError, TypeError):
            continue
        key = r.get("cell")
        if r["type"] == "start":
            if key in open_:   # the previous start never stopped: a crash
                t0 = open_.pop(key)
                if hb_t is not None and t0 < hb_t < t:
                    close(key, t0, hb_t)
                elif key is None:
                    n_runs += 1  # numbered, but without a window
            open_[key] = t
        elif key in open_:
            close(key, open_.pop(key), t)
    for key, t0 in open_.items():
        if hb_t is not None and hb_t > t0:
            close(key, t0, hb_t)
    return out


def discover_windows(results) -> list[Window]:
    """The windows the kit's run records carry (their t_start / t_end), oldest first."""
    results = Path(results)
    out: list[Window] = []

    def add(w):
        if w is not None:
            out.append(w)

    # Gate: the record and its phases (an exit-3 run keeps its record here; its phase files move to aborted/).
    gdir = results / "gate"
    recorded = set()
    for p in sorted(gdir.glob("gate_*Z.json")):
        m = _GATE_RECORD.match(p.name)
        r = _load_json(p)
        if not m or not isinstance(r, dict):
            continue
        utc = m.group(1)
        recorded.add(utc)
        add(_win(f"gate {utc}", "gate", r.get("t_start"), r.get("t_end"), _rel(p, results)))
        for ph in r.get("phases") or []:
            if isinstance(ph, dict):
                add(_win(f"gate {utc} phase {ph.get('phase')}", "gate_phase", ph.get("t_start"), ph.get("t_end"),
                         _rel(p, results)))
    # A gate run without a record yet (running, or killed outright): its phase files.
    for d in sorted(x for x in gdir.glob("*") if x.is_dir() and _STAMP.match(x.name) and x.name not in recorded):
        for p in sorted(d.glob("phase*.json")):
            r = _load_json(p)
            if isinstance(r, dict):
                add(_win(f"gate {d.name} phase {r.get('phase', p.stem[5:])}", "gate_phase", r.get("t_start"),
                         r.get("t_end"), _rel(p, results)))
    # Peer check, preflight --deep, clone-and-refresh.
    for p in sorted(results.glob("peers_*.json")):
        r = _load_json(p)
        if isinstance(r, dict):
            add(_win(f"peer check {p.stem[6:]}", "peers", r.get("t_start"), r.get("t_end"), _rel(p, results)))
    for p in sorted(results.glob("preflight_*.json")):
        r = _load_json(p)
        if isinstance(r, dict) and r.get("mode") in ("deep", "--deep"):
            add(_win(f"preflight deep {p.stem[10:]}", "preflight", r.get("t_start"), r.get("t_end"),
                     _rel(p, results)))
    for p in sorted((results / "convert").glob("refresh_*.json")):
        r = _load_json(p)
        if isinstance(r, dict):
            add(_win(f"refresh {p.stem[len('refresh_'):]}", "refresh", r.get("t_start"), r.get("t_end"),
                     _rel(p, results)))
    # Bench: each cell file's closing line, and the speed cells' cool-down idle (from the header).
    for p in sorted((results / "bench").glob("*.jsonl")):
        for r in _jsonl(p):
            if r.get("type") == "end":
                add(_win(f"bench {p.stem}", "bench", r.get("t_start"), r.get("t_end"), _rel(p, results)))
            elif r.get("type") == "header" and isinstance(r.get("cool_down"), dict):
                cd = r["cool_down"]
                b, a = cd.get("before") or {}, cd.get("after") or {}
                if isinstance(b, dict) and isinstance(a, dict) and b.get("utc") and a.get("utc"):
                    w = _win("bench cool-down idle", "idle", b["utc"], a["utc"], _rel(p, results))
                    if w is not None:
                        t0 = _dt.datetime.fromtimestamp(w.start, tz=_dt.timezone.utc)
                        w.name = f"bench cool-down idle {common.utc_stamp(t0)}"
                        add(w)
    for pat in ("bench/*.json", "tokenizer_*.json", "kl_8v4_*.json"):
        for p in sorted(results.glob(pat)):
            r = _load_json(p)
            if isinstance(r, dict):
                add(_win(f"bench {p.stem}", "bench", r.get("t_start"), r.get("t_end"), _rel(p, results)))
    # Pilot and its arms (the arms' times are epoch seconds).
    for p in sorted(results.glob("pilot_summary_*.json")):
        r = _load_json(p)
        if not isinstance(r, dict):
            continue
        stamp = p.stem[len("pilot_summary_"):]
        add(_win(f"pilot {stamp}", "pilot", r.get("t_start"), r.get("t_end"), _rel(p, results)))
        for arm, a in sorted((r.get("arms") or {}).items()):
            if isinstance(a, dict):
                add(_win(f"pilot {stamp} {arm}", "pilot_arm", a.get("t_start"), a.get("t_end"), _rel(p, results)))
    # Sessions S2, S3, S3b.
    for sdir in sorted(x for x in (results / "raw").glob("*") if x.is_dir()):
        out.extend(_session_windows(sdir, results))
    seen, uniq = set(), []
    for w in sorted(out, key=lambda w: (w.start, -w.end, w.kind, w.name)):   # a run before its parts
        key = (w.kind, w.start, w.end) if w.kind == "idle" else (w.kind, w.name, w.start, w.end)
        if key not in seen:   # the speed cells share one cool-down
            seen.add(key)
            uniq.append(w)
    return uniq


# ------------------------------------------------------------------------------------------- measure


def measure(samples: list[Sample], ends: list[float], max_len: float, w: Window) -> dict:
    """Energy, split, mean and peak W, samples, coverage, overlap and gaps of one window."""
    a, b = w.start, w.end
    joules = {c: 0.0 for c in COMPONENTS}
    have = {c: False for c in COMPONENTS}
    peak = {c: None for c in COMPONENTS}
    sampled, n, pieces = 0.0, 0, []
    i = bisect.bisect_right(ends, a)
    while i < len(samples) and samples[i].end < b + max_len:
        s = samples[i]
        i += 1
        lo, hi = max(s.start, a), min(s.end, b)
        if hi <= lo:
            continue
        ov = hi - lo
        n += 1
        sampled += ov
        pieces.append((lo, hi))
        for c in COMPONENTS:
            v = getattr(s, c)
            if v is None:
                continue
            have[c] = True
            joules[c] += v * ov
            peak[c] = v if peak[c] is None else max(peak[c], v)
    pieces.sort()
    covered, gaps, cur = 0.0, [], a
    for lo, hi in pieces:
        if lo > cur:
            gaps.append((cur, lo))
        if hi > cur:
            covered += hi - max(lo, cur)
            cur = hi
    if cur < b:
        gaps.append((cur, b))
    big = [(g0, g1) for g0, g1 in gaps if g1 - g0 > GAP_S]
    dur = b - a
    return {
        "name": w.name,
        "kind": w.kind,
        "source": w.source,
        "t_start": iso(a),
        "t_end": iso(b),
        "duration_s": round(dur, 3),
        "n_samples": n,
        "sampled_s": round(sampled, 3),
        "covered_s": round(covered, 3),
        "overlap_s": round(max(0.0, sampled - covered), 3),
        "coverage": round(covered / dur, 4) if dur > 0 else None,
        "energy_wh": {c: (round(joules[c] / 3600.0, 6) if have[c] else None) for c in COMPONENTS},
        "mean_w": {c: (round(joules[c] / sampled, 3) if have[c] and sampled > 0 else None) for c in COMPONENTS},
        "peak_w": {c: _r(peak[c], 3) for c in COMPONENTS},
        "gaps_over_5s": {"n": len(big), "total_s": round(sum(g1 - g0 for g0, g1 in big), 3),
                         "listed": [{"start": iso(g0), "end": iso(g1), "seconds": round(g1 - g0, 3)}
                                    for g0, g1 in big[:MAX_GAPS_LISTED]]},
    }


def build_record(raws: list[RawLog], windows: list[Window], *, auto: list[Window] | None = None,
                 utc: str | None = None, again_after: list[str] | None = None) -> dict:
    """The power record. `windows` are explicit ones (always kept); `auto` are discovered ones (kept when they
    overlap the log). Refuses a raw log whose documents overlap, and two raw logs that overlap in time."""
    stats = {id(r): time_stats(r) for r in raws}
    for r in raws:
        if double_counted(stats[id(r)]):
            raise Refused(_double_count_text(r.path.name, stats[id(r)]) + "; nothing written. Keep the raw log, "
                          "run --layout on it, and see RUNBOOK 'Power log'")
    samples = sorted((s for r in raws for s in r.samples), key=lambda s: (s.end, s.start))
    spans = sorted((stats[id(r)]["first"], stats[id(r)]["last"], r.path.name) for r in raws if r.samples)
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1 - 1.0:   # two loggers at once would count the same energy twice
            raise Refused(f"raw logs {an} and {bn} overlap in time; give one of them per record")
    ends = [s.end for s in samples]
    max_len = max((s.end - s.start for s in samples), default=0.0)
    chosen: list[Window] = []
    lo = min((sp[0] for sp in spans), default=None)
    hi = max((sp[1] for sp in spans), default=None)
    if samples:
        chosen.append(Window("log", "log", lo, hi, None))
    chosen += list(windows)
    left_out = 0
    if samples:
        for w in auto or []:
            if w.start < hi and w.end > lo:
                chosen.append(w)
            else:
                left_out += 1
    else:
        left_out = len(auto or [])
    rec = {
        "schema": SCHEMA,
        "label": LABEL,
        "descriptive": DESCRIPTIVE,
        "utc": utc or common.utc_iso(),
        "command": COMMAND,
        "samplers_declared": list(SAMPLERS),
        "interval_ms_declared": INTERVAL_MS,
        "raw_files": [raw_summary(r) for r in raws],
        "raw_policy": "the raw log is never committed or copied into the repository (it carries boot and host "
                      "fields); only these derived numbers and its sha256 and size",
        "not_measured": list(NOT_MEASURED),
        "caveats": list(CAVEATS),
        "method": dict(METHOD),
        "windows": [measure(samples, ends, max_len, w) for w in chosen],
        "windows_left_out_no_overlap": left_out,
        "again_after": list(again_after or []),
        "tool_sha256": common.sha256_file(Path(__file__)),
    }
    return rec


# ---------------------------------------------------------------------------------------------- check


def check_raw(raw: RawLog, *, max_age_s: float = CHECK_MAX_AGE_S, now: float | None = None) -> dict:
    """Readable and current? `max_age_s` <= 0 skips the currency test (a log that has ended)."""
    s = raw_summary(raw)
    st = time_stats(raw)
    problems = []
    if raw.bytes == 0:
        problems.append("empty: no sample written yet (output reaches the file in bursts; wait 30 s and check "
                        "again), or sudo or powermetrics refused the command (see the logger's terminal)")
    elif raw.documents == 0:
        problems.append(f"no plist document: {raw.text_bytes_skipped} byte(s) of text only (see the logger's "
                        f"terminal)")
    elif not raw.samples and raw.parsed > 0:
        problems.append(f"key layout differs: {raw.documents} document(s), {raw.parsed} parse, 0 samples, skipped "
                        f"{json.dumps(raw.skipped, sort_keys=True)}; keep the log and run --layout on it")
    elif len(raw.samples) < CHECK_MIN_SAMPLES:
        problems.append(f"{len(raw.samples)} readable sample(s); at least {CHECK_MIN_SAMPLES} expected "
                        f"(wait 30 s and check again)")
    if raw.unreadable:
        problems.append(f"{raw.unreadable} complete document(s) do not parse")
    missing = [x for x in SAMPLERS if x not in raw.seen]
    if raw.samples and missing:
        problems.append(f"samplers missing: {', '.join(missing)}")
    med = s["interval_ms"]["median"]
    if med is not None and not (INTERVAL_OK_MS[0] <= med <= INTERVAL_OK_MS[1]):
        problems.append(f"median interval {med} ms, expected about {INTERVAL_MS} ms")
    if double_counted(st):
        problems.append(_double_count_text(s["name"], st) + "; no record would be written: run --layout")
    if st["long_n"]:
        problems.append(f"{st['long_n']} document(s) cover more than {LONG_FACTOR:g} x the median interval "
                        f"(longest {st['long_max_ms']:.0f} ms): poweravg summaries, or a sleep; run --layout")
    age = None
    if raw.samples:
        age = (time.time() if now is None else now) - st["last"]
        if max_age_s > 0 and age > max_age_s:
            problems.append(f"stale: the last sample is {age:.0f} s old (> {max_age_s:.0f} s); is the logger "
                            f"running and writing to this file?")
    keys = ("name", "dir", "bytes", "documents", "samples", "unreadable", "truncated_tail", "text_bytes_skipped",
            "skipped", "combined_from_components", "from_energy", "first_sample_utc", "last_sample_utc", "span_s",
            "elapsed_sum_s", "overlap_s", "excess_s", "long_samples", "samplers_seen", "model_family")
    out = {("file" if k == "name" else k): s[k] for k in keys}
    out.update({"ok": not problems, "parsed": raw.parsed, "age_s": None if age is None else round(age, 1),
                "interval_ms_median": med, "problems": problems})
    return out


# ------------------------------------------------------------------------------------------------ cli


def raw_dir() -> Path:
    return common.work_dir().joinpath(*RAW_SUBDIR)


def newest_raw() -> Path:
    """The newest power_<UTC>.plist in the work dir, by its stamp (a power_smoke_* file never counts)."""
    d = raw_dir()
    hits = sorted(p for p in d.glob(RAW_GLOB) if _RAW_NAME.match(p.name))
    if not hits:
        raise Refused(f"no power_<UTC>.plist in {redact_path(d)} (a power_smoke_* file is never taken by default): "
                      f"give the raw log's path")
    return hits[-1]


def _refuse_inside_repo(p: Path) -> None:
    rp = p.expanduser().resolve()
    for root in (common.EXP_DIR, common.EXP_DIR.parents[2]):
        try:
            rp.relative_to(root.resolve())
        except ValueError:
            continue
        raise Refused(f"{redact_path(rp)} is inside the repository: a raw powermetrics log carries host fields and "
                      f"is never kept there; move it to {redact_path(raw_dir())}/")


def _recorded_raw(out_dir: Path) -> dict[str, list[str]]:
    """raw sha256, and "name:" + raw file name -> the power records in out_dir that already measure it. The name
    catches a log recorded while its logger was still running: it has grown since, so its sha256 differs, and a
    second record would count the same energy twice."""
    seen: dict[str, list[str]] = {}
    for p in sorted(out_dir.glob("power_*.json")):
        r = _load_json(p)
        for f in (r or {}).get("raw_files") or []:
            if not isinstance(f, dict):
                continue
            for k in (f.get("sha256"), f"name:{f['name']}" if f.get("name") else None):
                if k:
                    seen.setdefault(k, []).append(p.name)
    return seen


def write(raws: list[RawLog], windows: list[Window], *, results, out_dir, auto: bool = True,
          again: bool = False) -> Path:
    common.require_identity()
    out_dir = Path(out_dir)
    prior = _recorded_raw(out_dir)
    dup = sorted({n for r in raws for k in (r.sha256, f"name:{r.path.name}") for n in prior.get(k, [])})
    if dup and not again:
        raise Refused(f"{', '.join(dup)} already record(s) this raw log (same sha256, or same name: a log recorded "
                      f"while its logger still ran has grown since); nothing written (--again writes a new record "
                      f"that names the earlier one)")
    if not any(r.samples for r in raws):
        r = raws[0]
        raise Refused(f"the raw log holds no readable sample ({r.documents} document(s), {r.parsed} parse, skipped "
                      f"{json.dumps(r.skipped, sort_keys=True)}); nothing written. Keep it and run --layout on it "
                      f"(RUNBOOK 'Power log')")
    found = discover_windows(results) if auto else []
    for _ in range(5):
        now = common.utc_now()
        path = out_dir / f"power_{common.utc_stamp(now)}.json"
        if not path.exists():
            break
        time.sleep(1.0)
    rec = build_record(raws, windows, auto=found, utc=common.utc_iso(now), again_after=dup)
    return common.write_new_json(path, rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="exp_037 power log: CPU + GPU + ANE power from powermetrics (combined_power, an estimate; not "
                    "the whole SoC, not wall power) joined to the run's windows; descriptive only (HYPOTHESIS "
                    "'Power and energy (descriptive)')")
    ap.add_argument("raw", nargs="*", help="raw powermetrics plist log(s) (default: the newest "
                                           "$EXP036_WORK/exp037/power/power_<UTC>.plist)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="is the raw log readable and current? writes nothing")
    mode.add_argument("--layout", action="store_true",
                      help="key names and plist types of the log's documents, grouped by layout; no values; "
                           "writes nothing")
    ap.add_argument("--max-age", type=float, default=CHECK_MAX_AGE_S,
                    help=f"--check: the newest sample's largest age in s (default {CHECK_MAX_AGE_S:.0f}; 0 skips it)")
    ap.add_argument("--window", nargs=3, action="append", default=[], metavar=("NAME", "START", "END"),
                    help="an explicit window (ISO UTC, 20261007T050112Z or epoch seconds); repeatable")
    ap.add_argument("--results", default=None, help="where the run records are (default: this kit's results/)")
    ap.add_argument("--out", default=None, help="output directory (default: this kit's results/power/)")
    ap.add_argument("--no-auto", action="store_true", help="only the --window windows and the whole log")
    ap.add_argument("--again", action="store_true", help="write a record for a raw log that already has one")
    args = ap.parse_args(argv)
    try:
        paths = [Path(p) for p in args.raw] or [newest_raw()]
        for p in paths:
            _refuse_inside_repo(p)
        if args.check or args.layout:
            if len(paths) != 1:
                print(f"power_log: {'--check' if args.check else '--layout'} takes one raw log", file=sys.stderr)
                return 2
            if not paths[0].is_file():
                raise Refused(f"{redact_path(paths[0])} does not exist")
            raw = read_raw(paths[0])
            if args.layout:
                print(json.dumps(layout_report(raw), sort_keys=True, ensure_ascii=False))
                return 0 if raw.layouts else 1
            res = check_raw(raw, max_age_s=args.max_age)
            print(json.dumps(res, sort_keys=True, ensure_ascii=False))
            return 0 if res["ok"] else 1
        wins = []
        for name, a, b in args.window:
            try:
                t0, t1 = parse_time(a), parse_time(b)
            except (ValueError, TypeError) as e:
                print(f"power_log: --window {name}: {e}", file=sys.stderr)
                return 2
            if t1 <= t0:
                print(f"power_log: --window {name}: END is not after START", file=sys.stderr)
                return 2
            wins.append(Window(name, "explicit", t0, t1, None))
        if len({w.name for w in wins}) != len(wins) or any(w.name == "log" for w in wins):
            print("power_log: --window names must be distinct and not 'log'", file=sys.stderr)
            return 2
        for p in paths:
            if not p.is_file():
                raise Refused(f"{redact_path(p)} does not exist")
        raws = [read_raw(p) for p in paths]
        newest = max((r.samples[-1].end for r in raws if r.samples), default=None)
        if newest is not None and time.time() - newest < RECENT_S:
            print(f"power_log: the newest sample is {time.time() - newest:.0f} s old: is the logger still running? "
                  f"Stop it with Ctrl-C in its own terminal first, or this record misses the rest of the log",
                  file=sys.stderr)
        results = Path(args.results) if args.results else common.EXP_DIR / "results"
        out_dir = Path(args.out) if args.out else common.EXP_DIR.joinpath(*OUT_SUBDIR)
        path = write(raws, wins, results=results, out_dir=out_dir, auto=not args.no_auto, again=args.again)
    except (Refused, common.IdentityError) as e:
        print(f"power_log: refused: {e}", file=sys.stderr)
        return 1
    rec = json.loads(path.read_text(encoding="utf-8"))
    log = rec["windows"][0] if rec["windows"] and rec["windows"][0]["kind"] == "log" else {}
    print(json.dumps({"ok": True, "record": redact_path(path), "label": LABEL, "windows": len(rec["windows"]),
                      "log_combined_wh": (log.get("energy_wh") or {}).get("combined"),
                      "log_coverage": log.get("coverage"),
                      "truncated_tail": any(f["truncated_tail"] for f in rec["raw_files"])},
                     sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
