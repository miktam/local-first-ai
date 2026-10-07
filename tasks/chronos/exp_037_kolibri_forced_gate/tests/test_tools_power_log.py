"""tools/power_log.py: CPU + GPU + ANE power (powermetrics combined_power) joined to the run's windows (descriptive
only; Andrei's request 2026-10-06; HYPOTHESIS "Power and energy (descriptive)"; RUNBOOK "Power log").

On synthetic NUL-separated plist streams (powermetrics never runs here; the key names follow the tool's assumptions
until the pre-freeze smoke on the mbp has shown the real layout): both separator placements parse alike; text before a
document, or between two, is skipped and counted; a document cut short by Ctrl-C is reported as truncated_tail and left
out, a broken complete one as unreadable; the sample fields (is_delta, an integer or real elapsed_ns, components,
gpu dict fallback, <c>_energy in mJ when <c>_power is absent, combined from components); energy, split, mean and peak
W; coverage, overlap and gaps over 5 s, window edges included; documents that count the same time twice (poweravg-like
summaries, duplicates) refused, while whole-second timestamps are not mistaken for them; the windows found in the kit's
run records (gate record and phases, an orphan gate run's phase files, peer check, preflight --deep, refresh, bench
cells and their shared cool-down idle, pilot and its arms, session start/stop pairs with a crash); no host field
(hw_model, kern_*) is ever written, and --layout prints key names and types, never values; the record's schema and
label; --check (readable and current, and "key layout differs"); the CLI (identity, duplicates, --again, a raw log
inside the repository refused, the newest power_<UTC>.plist by default, never a smoke file); the RUNBOOK's logger
lines are exactly the pinned sudoers command; the label and what is not measured agree with HYPOTHESIS and RUNBOOK;
and no deciding code reads a power record.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import plistlib
import re
import time
from pathlib import Path

import pytest

from tools import common, power_log as pl
from tools.redact import forbidden_keys

EXP = Path(__file__).resolve().parents[1]
T0 = dt.datetime(2026, 10, 7, 5, 0, 0)          # naive UTC, as plistlib writes and reads plist dates
T0_EPOCH = T0.replace(tzinfo=dt.timezone.utc).timestamp()
HOST = {"hw_model": "Mac17,7", "kern_osversion": "26A428Q", "kern_bootargs": "fixture-bootarg-7f3a",
        "kern_boottime": 1790012345}
LABEL = "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power"


def doc(i, *, cpu=5000.0, gpu=24000.0, ane=1000.0, combined="sum", elapsed_ns=1_000_000_000, is_delta=True,
        base=T0, gpu_in_gpu_dict=False, host=True, energy=False) -> bytes:
    """One powermetrics-shaped plist sample ending at base + i seconds (powers in mW; with energy=True the components
    are written as <c>_energy, in mJ over the sample's interval, instead of <c>_power). plistlib writes the date to
    the whole second, as a plist <date> holds it."""
    proc = {"clusters": [{"name": "P0-Cluster", "freq_hz": 3.2e9}]}
    gpu_d = {"freq_hz": 1.4e9, "idle_ratio": 0.05}
    el_s = elapsed_ns / 1e9 if isinstance(elapsed_ns, (int, float)) and elapsed_ns > 0 else 1.0
    for name, v in (("cpu", cpu), ("gpu", gpu), ("ane", ane)):
        if v is None:
            continue
        key, val = (f"{name}_energy", v * el_s) if energy else (f"{name}_power", v)
        (gpu_d if name == "gpu" and gpu_in_gpu_dict else proc)[key] = val
    if combined == "sum":
        proc["combined_power"] = (cpu or 0.0) + (gpu or 0.0) + (ane or 0.0)
    elif combined is not None:
        proc["combined_power"] = combined
    d = {"is_delta": is_delta, "elapsed_ns": elapsed_ns, "timestamp": base + dt.timedelta(seconds=i),
         "processor": proc, "gpu": gpu_d}
    if host:
        d.update(HOST)
    return plistlib.dumps(d)


def stream(docs: list[bytes], *, nul_before=False, tail: bytes = b"") -> bytes:
    if nul_before:
        return b"".join(b"\0" + d for d in docs) + (b"\0" + tail if tail else b"")
    return b"".join(d + b"\0" for d in docs) + tail


def write_raw(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def steady(n=120, **kw) -> list[bytes]:
    """n contiguous 1 s samples ending at T0+1 .. T0+n: 30 W combined (5 CPU, 24 GPU, 1 ANE)."""
    return [doc(i, **kw) for i in range(1, n + 1)]


def poweravg_like(n=120, every=10) -> list[bytes]:
    """n 1 s samples at 30 W and, after every `every`-th, one document covering the previous `every` seconds: how
    powermetrics' poweravg summaries (-a, "Display poweravg every N samples", default 10; the pinned command cannot
    pass -a 0) might appear in -f plist. Their real form is unknown until the mbp smoke."""
    out = []
    for i in range(1, n + 1):
        out.append(doc(i))
        if i % every == 0:
            out.append(doc(i, elapsed_ns=every * 1_000_000_000))
    return out


def window(name, a, b, kind="explicit"):
    return pl.Window(name, kind, T0_EPOCH + a, T0_EPOCH + b)


def record(tmp_path, docs, windows=(), auto=None, name="power_20261007T045959Z.plist"):
    raw = pl.read_raw(write_raw(tmp_path / "work" / name, stream(docs)))
    return pl.build_record([raw], list(windows), auto=auto, utc="2026-10-07T07:00:00Z")


def by_name(rec):
    return {w["name"]: w for w in rec["windows"]}


@pytest.fixture
def identity(monkeypatch):
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "Miktam", "email": "hello@localfirstai.eu"})


# ------------------------------------------------------------------------------------------------- parse


def test_both_separator_placements_parse_alike(tmp_path):
    a = pl.read_raw(write_raw(tmp_path / "a.plist", stream(steady(10))))
    b = pl.read_raw(write_raw(tmp_path / "b.plist", stream(steady(10), nul_before=True)))
    for r in (a, b):
        assert (r.documents, len(r.samples), r.unreadable, r.truncated_tail) == (10, 10, 0, False)
        assert r.text_bytes_skipped == 0 and r.from_energy == 0
    assert [(s.start, s.end, s.combined) for s in a.samples] == [(s.start, s.end, s.combined) for s in b.samples]
    s = a.samples[0]
    assert (s.start, s.end) == (T0_EPOCH, T0_EPOCH + 1)
    assert (s.combined, s.cpu, s.gpu, s.ane) == (30.0, 5.0, 24.0, 1.0)
    assert sorted(a.seen) == ["ane_power", "cpu_power", "gpu_power"] and a.model_family == "Mac17"


@pytest.mark.parametrize("nul_before", [False, True])
def test_truncated_tail_after_ctrl_c_is_reported_and_left_out(tmp_path, nul_before):
    cut = doc(11)[:157]
    r = pl.read_raw(write_raw(tmp_path / "t.plist", stream(steady(10), nul_before=nul_before, tail=cut)))
    assert r.truncated_tail is True and r.unreadable == 0
    assert len(r.samples) == 10 and r.documents == 11 and r.parsed == 10
    assert r.bytes == (tmp_path / "t.plist").stat().st_size
    assert r.sha256 == common.sha256_file(tmp_path / "t.plist")
    c = pl.check_raw(r, now=T0_EPOCH + 15)
    assert c["ok"] and c["truncated_tail"] and c["samples"] == 10          # normal while the logger runs


def test_a_broken_complete_document_is_unreadable_and_fails_the_check(tmp_path):
    docs = steady(10)
    docs[4] = docs[4][:300]                                              # broken, but followed by a NUL
    r = pl.read_raw(write_raw(tmp_path / "u.plist", stream(docs)))
    assert (r.unreadable, r.truncated_tail, len(r.samples)) == (1, False, 9)
    c = pl.check_raw(r, now=T0_EPOCH + 12)
    assert not c["ok"] and any("do not parse" in p for p in c["problems"])


def test_text_before_or_between_documents_is_skipped_and_counted(tmp_path):
    """No text is expected in -f plist output; if some precedes a document (a banner) or stands alone between two,
    it is skipped and counted, and no document is lost to it."""
    docs = steady(20)
    data = (b"banner text\n" + b"".join(d + b"\0" for d in docs[:10]) + b"note between documents\0"
            + b"".join(d + b"\0" for d in docs[10:]))
    r = pl.read_raw(write_raw(tmp_path / "x.plist", data))
    assert (r.documents, r.unreadable, len(r.samples)) == (20, 0, 20)
    assert r.text_bytes_skipped == len(b"banner text\n") + len(b"note between documents")
    c = pl.check_raw(r, now=T0_EPOCH + 21)
    assert c["ok"] and c["text_bytes_skipped"] == r.text_bytes_skipped
    only_text = pl.check_raw(pl.read_raw(write_raw(tmp_path / "t.plist", b"usage: powermetrics ...\0")))
    assert not only_text["ok"] and only_text["problems"][0].startswith("no plist document")


def test_sample_fields(tmp_path):
    docs = [
        doc(1),
        doc(2, is_delta=False),                                           # skipped
        doc(3, combined=None),                                            # combined = cpu + gpu + ane
        doc(4, gpu_in_gpu_dict=True),                                     # gpu.gpu_power
        doc(5, combined=None, ane=None),                                  # no combined, a component missing
        doc(6, elapsed_ns=0),                                             # no interval
        doc(7, cpu=-5.0, combined=40000.0),                               # a negative component is not a value
        plistlib.dumps({"is_delta": True, "elapsed_ns": 1_000_000_000, "processor": {"combined_power": 1.0}}),
        plistlib.dumps(["not", "a", "dict"]),
    ]
    r = pl.read_raw(write_raw(tmp_path / "f.plist", stream(docs)))
    assert r.skipped == {"not_delta": 1, "no_time": 2, "no_power": 1}
    assert r.combined_from_components == 1 and r.unreadable == 1 and r.from_energy == 0
    by_end = {round(s.end - T0_EPOCH): s for s in r.samples}
    assert sorted(by_end) == [1, 3, 4, 7]
    assert by_end[3].combined == 30.0 and by_end[4].gpu == 24.0 and by_end[4].combined == 30.0
    assert by_end[7].cpu is None and by_end[7].combined == 40.0


def test_elapsed_ns_as_a_real_is_read(tmp_path):
    docs = [doc(i, elapsed_ns=1.0e9) for i in range(1, 31)]
    docs += [doc(31, elapsed_ns=float("nan")), doc(32, elapsed_ns=-1.0e9), doc(33, elapsed_ns=True)]
    r = pl.read_raw(write_raw(tmp_path / "r.plist", stream(docs)))
    assert len(r.samples) == 30 and r.skipped["no_time"] == 3
    assert r.samples[0].start == T0_EPOCH and r.samples[0].end == T0_EPOCH + 1
    assert pl.check_raw(r, now=T0_EPOCH + 31)["ok"]


def test_energy_keys_stand_in_for_missing_power_keys(tmp_path):
    """If the mbp's documents carry <c>_energy (mJ) but no <c>_power, power is energy / elapsed."""
    docs = [doc(2 * i, elapsed_ns=2_000_000_000, energy=True) for i in range(1, 11)]           # 10 x 2 s
    docs += [doc(30 + i, energy=True, combined=None) for i in range(1, 6)]                    # no combined_power
    docs += [doc(40, energy=True, gpu_in_gpu_dict=True)]                                      # gpu.gpu_energy
    r = pl.read_raw(write_raw(tmp_path / "e.plist", stream(docs)))
    assert len(r.samples) == 16 and r.from_energy == 16 * 3 and r.combined_from_components == 5
    for s in r.samples:
        assert (s.combined, s.cpu, s.gpu, s.ane) == pytest.approx((30.0, 5.0, 24.0, 1.0))
    c = pl.check_raw(r, now=T0_EPOCH + 41)
    assert c["ok"] and c["from_energy"] == 48 and c["samplers_seen"] == ["ane_power", "cpu_power", "gpu_power"]
    rec = pl.build_record([r], [window("first", 0, 20)])
    assert by_name(rec)["first"]["energy_wh"]["combined"] == pytest.approx(20 * 30 / 3600, abs=1e-6)


def test_a_timestamp_sample_covers_the_interval_before_it(tmp_path):
    r = pl.read_raw(write_raw(tmp_path / "e.plist", stream([doc(10, elapsed_ns=1_250_000_000)])))
    s = r.samples[0]
    assert s.end == T0_EPOCH + 10 and s.start == pytest.approx(T0_EPOCH + 8.75)


# --------------------------------------------------------------------------------------- energy and windows


def test_energy_split_mean_and_peak(tmp_path):
    docs = steady(120)
    docs[59] = doc(60, cpu=5000.0, gpu=74000.0, ane=1000.0)               # one 80 W sample
    rec = record(tmp_path, docs, [window("first half", 0, 60)])
    log, half = by_name(rec)["log"], by_name(rec)["first half"]
    assert log["kind"] == "log" and rec["windows"][0] is log
    assert log["duration_s"] == 120 and log["coverage"] == 1.0 and log["n_samples"] == 120
    assert log["sampled_s"] == 120.0 and log["overlap_s"] == 0.0
    assert log["energy_wh"]["combined"] == pytest.approx((119 * 30 + 80) / 3600, abs=1e-6)
    assert log["energy_wh"]["cpu"] == pytest.approx(120 * 5 / 3600, abs=1e-6)
    assert log["energy_wh"]["gpu"] == pytest.approx((119 * 24 + 74) / 3600, abs=1e-6)
    assert log["energy_wh"]["ane"] == pytest.approx(120 / 3600, abs=1e-6)
    assert log["peak_w"] == {"combined": 80.0, "cpu": 5.0, "gpu": 74.0, "ane": 1.0}
    assert log["mean_w"]["combined"] == pytest.approx((119 * 30 + 80) / 120, abs=1e-3)
    assert half["energy_wh"]["combined"] == pytest.approx((59 * 30 + 80) / 3600, abs=1e-6)
    assert half["n_samples"] == 60 and half["gaps_over_5s"] == {"n": 0, "total_s": 0.0, "listed": []}


def test_one_watt_hour(tmp_path):
    rec = record(tmp_path, steady(120), [window("all", 0, 120)])
    assert by_name(rec)["all"]["energy_wh"]["combined"] == pytest.approx(1.0, abs=1e-9)   # 30 W x 120 s = 3,600 J


def test_partial_samples_at_the_window_edges(tmp_path):
    rec = record(tmp_path, steady(120), [window("edge", 30.5, 31.5)])
    w = by_name(rec)["edge"]
    assert w["n_samples"] == 2 and w["sampled_s"] == 1.0 and w["coverage"] == 1.0 and w["overlap_s"] == 0.0
    assert w["energy_wh"]["combined"] == pytest.approx(30 / 3600, abs=1e-6)              # Wh to 6 decimals


def test_coverage_and_gaps(tmp_path):
    hole = set(range(41, 56)) | {70}                                     # 15 s missing, then 1 s
    docs = [doc(i) for i in range(1, 121) if i not in hole]
    rec = record(tmp_path, docs, [window("mid", 30, 90), window("tail", 100, 140), window("before", -20, 10)])
    mid, tail, before = (by_name(rec)[k] for k in ("mid", "tail", "before"))
    assert mid["covered_s"] == 44.0 and mid["coverage"] == pytest.approx(44 / 60, abs=1e-4)
    assert mid["gaps_over_5s"]["n"] == 1 and mid["gaps_over_5s"]["total_s"] == 15.0
    assert mid["gaps_over_5s"]["listed"] == [{"start": "2026-10-07T05:00:40Z", "end": "2026-10-07T05:00:55Z",
                                              "seconds": 15.0}]                # the 1 s hole is not listed
    assert mid["energy_wh"]["combined"] == pytest.approx(44 * 30 / 3600, abs=1e-6)
    assert mid["mean_w"]["combined"] == 30.0                             # no extrapolation over the gaps
    assert tail["coverage"] == 0.5 and tail["gaps_over_5s"]["listed"][0]["seconds"] == 20.0   # the log ends
    assert before["coverage"] == pytest.approx(10 / 30, abs=1e-4) and before["gaps_over_5s"]["total_s"] == 20.0
    assert by_name(rec)["log"]["gaps_over_5s"]["n"] == 1


def test_a_window_outside_the_log_has_no_energy(tmp_path):
    rec = record(tmp_path, steady(30), [window("later", 600, 660)],
                 auto=[window("auto later", 600, 660, "bench"), window("auto in", 5, 25, "bench")])
    w = by_name(rec)["later"]
    assert w["n_samples"] == 0 and w["coverage"] == 0.0 and w["energy_wh"]["combined"] is None
    assert w["mean_w"]["combined"] is None and w["peak_w"]["combined"] is None and w["overlap_s"] == 0.0
    assert "auto later" not in by_name(rec) and "auto in" in by_name(rec)       # discovered: only if it overlaps
    assert rec["windows_left_out_no_overlap"] == 1


def test_overlapping_raw_logs_are_refused(tmp_path):
    a = pl.read_raw(write_raw(tmp_path / "a.plist", stream(steady(60))))
    b = pl.read_raw(write_raw(tmp_path / "b.plist", stream([doc(i) for i in range(30, 90)])))
    with pytest.raises(pl.Refused, match="overlap in time"):
        pl.build_record([a, b], [])
    c = pl.read_raw(write_raw(tmp_path / "c.plist", stream([doc(i) for i in range(61, 90)])))
    rec = pl.build_record([a, c], [])                                    # back to back: one record
    assert by_name(rec)["log"]["covered_s"] == 89.0 and len(rec["raw_files"]) == 2


# ----------------------------------------------------------------------------------- double counting (P1 review)


def test_poweravg_like_documents_are_refused_not_counted_twice(tmp_path, identity, capsys):
    """The review's probe: 120 s at 30 W (1 Wh) plus a 10 s document every 10 samples. Summed blindly, the log gives
    2 Wh at coverage 1.0 with a median interval of 1,000 ms; the excess of summed elapsed over the span shows it."""
    p = write_raw(tmp_path / "work" / "power_20261007T045959Z.plist", stream(poweravg_like()))
    raw = pl.read_raw(p)
    s = pl.raw_summary(raw)
    assert (s["samples"], s["span_s"], s["elapsed_sum_s"], s["covered_s"]) == (132, 120.0, 240.0, 120.0)
    assert s["overlap_s"] == 120.0 and s["excess_s"] == 120.0
    assert s["long_samples"] == {"n": 12, "max_ms": 10000.0}
    assert s["interval_ms"]["median"] == 1000.0                          # the median alone does not show it
    c = pl.check_raw(raw, now=T0_EPOCH + 125)
    assert not c["ok"] and c["excess_s"] == 120.0 and c["long_samples"]["n"] == 12
    assert any("counted twice" in x for x in c["problems"])
    assert any("3 x the median interval" in x for x in c["problems"])
    with pytest.raises(pl.Refused, match="counted twice"):
        pl.build_record([raw], [window("all", 0, 120)])
    out = tmp_path / "out"
    assert pl.main([str(p), "--no-auto", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "counted twice" in err and "--layout" in err and not out.exists()


def test_duplicate_documents_are_refused_too(tmp_path):
    """A repeated 1 s document is not long, so only the excess shows it."""
    docs = []
    for i in range(1, 121):
        docs.append(doc(i))
        if i % 10 == 0:
            docs.append(doc(i))
    raw = pl.read_raw(write_raw(tmp_path / "d.plist", stream(docs)))
    s = pl.raw_summary(raw)
    assert s["excess_s"] == 12.0 and s["overlap_s"] == 12.0 and s["long_samples"]["n"] == 0
    c = pl.check_raw(raw, now=T0_EPOCH + 121)
    assert not c["ok"] and [x for x in c["problems"] if "counted twice" in x]
    with pytest.raises(pl.Refused, match="counted twice"):
        pl.build_record([raw], [])


@pytest.mark.parametrize("elapsed_ms", [1000.4, 1004.0, 1020.0, 1050.0])
def test_whole_second_timestamps_are_not_taken_for_double_counting(tmp_path, elapsed_ms):
    """The real sample ends drift by elapsed - 1 s per sample, and the plist date keeps whole seconds only: neighbours
    overlap by that drift, and about a second is left uncovered each time a second is skipped. That is placement,
    not double counting (each sample's energy is counted once), and the excess stays within the 1 s placement of
    the first and last timestamps, whatever the drift; overlap_s grows with it (5 % at 1,050 ms)."""
    el = elapsed_ms / 1000.0
    n = 600
    docs = [doc(0.5 + el * k, elapsed_ns=int(round(el * 1e9))) for k in range(n)]    # plistlib truncates the date
    raw = pl.read_raw(write_raw(tmp_path / "w.plist", stream(docs)))
    s = pl.raw_summary(raw)
    assert abs(s["excess_s"]) <= 2.0 and s["long_samples"]["n"] == 0
    assert s["overlap_s"] == pytest.approx((el - 1.0) * s["covered_s"], rel=0.1, abs=0.5)
    c = pl.check_raw(raw, now=raw.samples[-1].end + 1)
    assert c["ok"], c["problems"]
    rec = pl.build_record([raw], [])
    log = by_name(rec)["log"]
    assert log["energy_wh"]["combined"] == pytest.approx(30.0 * n * el / 3600, abs=1e-6)   # exact: once each
    assert log["mean_w"]["combined"] == pytest.approx(30.0, abs=1e-3)
    assert log["overlap_s"] == pytest.approx(s["overlap_s"], abs=1e-3)
    assert log["sampled_s"] == pytest.approx(n * el, abs=1e-3) and log["sampled_s"] <= log["duration_s"] + 1.0


def test_a_long_sample_alone_is_kept_in_the_record(tmp_path):
    """A sample that spans a sleep covers real time once: --check flags it (it could be a poweravg summary), the
    record keeps it."""
    docs = steady(10) + [doc(3610, elapsed_ns=3_600_000_000_000, combined=2000.0)] + \
        [doc(i) for i in range(3611, 3621)]
    raw = pl.read_raw(write_raw(tmp_path / "s.plist", stream(docs)))
    s = pl.raw_summary(raw)
    assert s["long_samples"] == {"n": 1, "max_ms": 3_600_000.0} and abs(s["excess_s"]) < 1e-6
    c = pl.check_raw(raw, max_age_s=0)
    assert not c["ok"] and c["problems"] == [c["problems"][0]] and "3 x the median interval" in c["problems"][0]
    log = by_name(pl.build_record([raw], []))["log"]
    assert log["coverage"] == 1.0 and log["energy_wh"]["combined"] == pytest.approx((20 * 30 + 3600 * 2) / 3600,
                                                                                  abs=1e-6)


# ------------------------------------------------------------------------------------- windows from records


def _j(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _jl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{torn", encoding="utf-8")


def results_tree(root: Path) -> Path:
    r = root / "results"
    _j(r / "gate" / "gate_20261007T050000Z.json", {
        "t_start": "2026-10-07T05:00:00Z", "t_end": "2026-10-07T08:00:00Z",
        "phases": [{"phase": "1", "title": "G0 static, G1", "t_start": "2026-10-07T05:00:00Z",
                    "t_end": "2026-10-07T05:10:00Z"},
                   {"phase": "2", "title": "reference", "t_start": "2026-10-07T05:10:00Z",
                    "t_end": "2026-10-07T06:00:00Z"}]})
    _j(r / "gate" / "20261007T050000Z" / "phase1.json", {"phase": "1", "t_start": "2026-10-07T05:00:00Z",
                                                         "t_end": "2026-10-07T05:10:00Z"})   # has a record: not again
    _j(r / "gate" / "20261007T090000Z" / "phase1.json", {"phase": "1", "t_start": "2026-10-07T09:00:00Z",
                                                         "t_end": "2026-10-07T09:05:00Z"})   # orphan run
    _j(r / "gate" / "20261007T090000Z" / "run.json", {"t_start": "2026-10-07T09:00:00Z"})
    _j(r / "peers_20261007T044000Z.json", {"t_start": "2026-10-07T04:10:00Z", "t_end": "2026-10-07T04:40:00Z"})
    (r / "peers_20261007T044500Z.json").write_text("{", encoding="utf-8")                  # unreadable: skipped
    _j(r / "preflight_20261007T040000Z.json", {"mode": "quick", "t_start": "2026-10-07T04:00:00Z",
                                               "t_end": "2026-10-07T04:00:01Z"})
    _j(r / "preflight_20261007T040500Z.json", {"mode": "deep", "t_start": "2026-10-07T04:05:00Z",
                                               "t_end": "2026-10-07T04:09:00Z"})
    _j(r / "convert" / "refresh_8bit_20261007T041000Z.json", {"t_start": "2026-10-07T04:09:30Z",
                                                              "t_end": "2026-10-07T04:10:00Z"})
    cool = {"idle_s": 600, "before": {"utc": "2026-10-07T10:00:00Z", "thermal_state": 0},
            "after": {"utc": "2026-10-07T10:10:00Z", "thermal_state": 0}}
    _jl(r / "bench" / "speed_20261007T101000Z.jsonl", [
        {"type": "header", "t_start": "2026-10-07T10:10:00Z", "cool_down": cool},
        {"type": "record", "kind": "run", "t_start": "2026-10-07T10:11:00Z", "t_end": "2026-10-07T10:12:00Z"},
        {"type": "end", "complete": True, "t_start": "2026-10-07T10:10:00Z", "t_end": "2026-10-07T10:40:00Z"}])
    _jl(r / "bench" / "speed_desc_20261007T104000Z.jsonl", [
        {"type": "header", "t_start": "2026-10-07T10:40:00Z", "cool_down": cool},           # the same cool-down
        {"type": "end", "complete": True, "t_start": "2026-10-07T10:40:00Z", "t_end": "2026-10-07T11:00:00Z"}])
    _j(r / "tokenizer_20261007T110500Z.json", {"t_start": "2026-10-07T11:00:00Z", "t_end": "2026-10-07T11:05:00Z"})
    _j(r / "kl_8v4_20261007T113000Z.json", {"t_start": "2026-10-07T11:05:00Z", "t_end": "2026-10-07T11:30:00Z"})
    a0 = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc).timestamp()
    _j(r / "pilot_summary_20261007T120000Z.json", {
        "t_start": "2026-10-07T12:00:00Z", "t_end": "2026-10-07T13:30:00Z",
        "arms": {"K8": {"t_start": a0, "t_end": a0 + 1800.5}, "G8": {"t_start": a0 + 1800.5, "t_end": a0 + 3600}}})
    _jl(r / "raw" / "S2" / "session.jsonl", [
        {"type": "start", "utc": "2026-10-07T20:00:00Z", "pid": 1},
        {"type": "stop", "utc": "2026-10-07T22:00:00Z", "reason": "SIGINT"},
        {"type": "start", "utc": "2026-10-07T22:05:00Z"},                                    # crashed
        {"type": "start", "utc": "2026-10-08T01:00:00Z"},
        {"type": "stop", "utc": "2026-10-08T09:00:00Z"},
        {"type": "private_file", "utc": "2026-10-08T09:00:01Z"}])
    _j(r / "raw" / "S2" / ".heartbeat", {"utc": "2026-10-08T00:30:00Z", "state": "crashed"})
    _jl(r / "raw" / "S3" / "session.jsonl", [
        {"type": "start", "utc": "2026-10-08T20:00:00Z", "cell": "K8:gpqa_en:high:0"},
        {"type": "stop", "utc": "2026-10-08T21:00:00Z", "cell": "K8:gpqa_en:high:0"},
        {"type": "start", "utc": "2026-10-08T21:10:00Z"}])                                  # still running
    _j(r / "raw" / "S3" / ".heartbeat", {"utc": "2026-10-08T23:00:00Z", "state": "running"})
    return r


def _utc(w) -> tuple[str, str]:
    return pl.iso(w.start), pl.iso(w.end)


def test_windows_from_the_run_records(tmp_path):
    r = results_tree(tmp_path)
    ws = pl.discover_windows(r)
    got = {w.name: (w.kind, *_utc(w)) for w in ws}
    assert got == {
        "gate 20261007T050000Z": ("gate", "2026-10-07T05:00:00Z", "2026-10-07T08:00:00Z"),
        "gate 20261007T050000Z phase 1": ("gate_phase", "2026-10-07T05:00:00Z", "2026-10-07T05:10:00Z"),
        "gate 20261007T050000Z phase 2": ("gate_phase", "2026-10-07T05:10:00Z", "2026-10-07T06:00:00Z"),
        "gate 20261007T090000Z phase 1": ("gate_phase", "2026-10-07T09:00:00Z", "2026-10-07T09:05:00Z"),
        "peer check 20261007T044000Z": ("peers", "2026-10-07T04:10:00Z", "2026-10-07T04:40:00Z"),
        "preflight deep 20261007T040500Z": ("preflight", "2026-10-07T04:05:00Z", "2026-10-07T04:09:00Z"),
        "refresh 8bit_20261007T041000Z": ("refresh", "2026-10-07T04:09:30Z", "2026-10-07T04:10:00Z"),
        "bench cool-down idle 20261007T100000Z": ("idle", "2026-10-07T10:00:00Z", "2026-10-07T10:10:00Z"),
        "bench speed_20261007T101000Z": ("bench", "2026-10-07T10:10:00Z", "2026-10-07T10:40:00Z"),
        "bench speed_desc_20261007T104000Z": ("bench", "2026-10-07T10:40:00Z", "2026-10-07T11:00:00Z"),
        "bench tokenizer_20261007T110500Z": ("bench", "2026-10-07T11:00:00Z", "2026-10-07T11:05:00Z"),
        "bench kl_8v4_20261007T113000Z": ("bench", "2026-10-07T11:05:00Z", "2026-10-07T11:30:00Z"),
        "pilot 20261007T120000Z": ("pilot", "2026-10-07T12:00:00Z", "2026-10-07T13:30:00Z"),
        "pilot 20261007T120000Z K8": ("pilot_arm", "2026-10-07T12:00:00Z", "2026-10-07T12:30:00Z"),
        "pilot 20261007T120000Z G8": ("pilot_arm", "2026-10-07T12:30:00Z", "2026-10-07T13:00:00Z"),
        "S2 run 1": ("session", "2026-10-07T20:00:00Z", "2026-10-07T22:00:00Z"),
        "S2 run 2": ("session", "2026-10-07T22:05:00Z", "2026-10-08T00:30:00Z"),     # crash: to the heartbeat
        "S2 run 3": ("session", "2026-10-08T01:00:00Z", "2026-10-08T09:00:00Z"),
        "S3 cell K8:gpqa_en:high:0": ("session_cell", "2026-10-08T20:00:00Z", "2026-10-08T21:00:00Z"),
        "S3 run 1": ("session", "2026-10-08T21:10:00Z", "2026-10-08T23:00:00Z"),     # running: to the heartbeat
    }
    starts = [w.start for w in ws]
    assert starts == sorted(starts)
    names = [w.name for w in ws]
    assert names.index("gate 20261007T050000Z") < names.index("gate 20261007T050000Z phase 1")   # run before parts
    assert {w.source for w in ws if w.kind == "session"} == {"results/raw/S2/session.jsonl",
                                                              "results/raw/S3/session.jsonl"}
    by = {w.name: w for w in ws}
    assert by["pilot 20261007T120000Z K8"].end - by["pilot 20261007T120000Z K8"].start == 1800.5


def test_the_record_keeps_the_discovered_windows_the_log_covers(tmp_path):
    r = results_tree(tmp_path)
    base = dt.datetime(2026, 10, 7, 4, 59, 0)
    docs = [doc(i, base=base) for i in range(1, 3 * 3600)]                # 04:59 to 07:59
    rec = record(tmp_path, docs, auto=pl.discover_windows(r))
    names = [w["name"] for w in rec["windows"]]
    assert names == ["log", "gate 20261007T050000Z", "gate 20261007T050000Z phase 1",
                     "gate 20261007T050000Z phase 2"]
    g = by_name(rec)["gate 20261007T050000Z"]
    assert g["source"] == "results/gate/gate_20261007T050000Z.json"
    assert g["coverage"] == pytest.approx((3 * 3600 - 61) / (3 * 3600), abs=1e-4)
    assert g["gaps_over_5s"]["listed"] == [{"start": "2026-10-07T07:58:59Z", "end": "2026-10-07T08:00:00Z",
                                            "seconds": 61.0}]                    # the logger stopped first
    assert by_name(rec)["gate 20261007T050000Z phase 1"]["coverage"] == 1.0
    assert rec["windows_left_out_no_overlap"] == 17


# --------------------------------------------------------------------------------------- privacy and schema


def _keys(obj) -> set[str]:
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            out |= _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _keys(v)
    return out


def test_no_host_field_is_ever_written(tmp_path, identity):
    raw = write_raw(tmp_path / "work" / "power_20261007T045959Z.plist", stream(steady(30), tail=doc(31)[:99]))
    out = tmp_path / "out"
    assert pl.main([str(raw), "--results", str(results_tree(tmp_path)), "--out", str(out),
                    "--window", "w", "2026-10-07T05:00:05Z", "20261007T050020Z"]) == 0
    (path,) = sorted(out.glob("power_*.json"))
    text = path.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert not (_keys(rec) & set(pl.HOST_KEYS)) and not forbidden_keys(rec)
    for v in HOST.values():
        assert str(v) not in text, v
    assert "/Users/" not in text and "clusters" not in text and "freq_hz" not in text
    assert rec["raw_files"][0]["model_family"] == "Mac17"                 # the family only (module docstring)
    assert set(_keys(rec)) <= set(_keys(_schema_example()))


def test_layout_prints_key_names_and_types_never_values(tmp_path, capsys):
    """--layout is the smoke's dump: key names and plist types per layout, with counts and elapsed times."""
    docs = poweravg_like(30)
    docs[5] = doc(6, elapsed_ns=1.0e9)                                    # elapsed_ns as a real: a second layout
    p = write_raw(tmp_path / "work" / "power_smoke_20261007T045959Z.plist", stream(docs, tail=doc(31)[:80]))
    assert pl.main(["--layout", str(p)]) == 0
    text = capsys.readouterr().out.strip()
    assert "\n" not in text
    out = json.loads(text)
    assert (out["documents"], out["parsed"], out["samples"], out["truncated_tail"]) == (34, 33, 33, True)
    big, small = out["layouts"]
    assert (big["documents"], big["samples"], small["documents"], small["samples"]) == (32, 32, 1, 1)
    assert big["elapsed_ms"] == {"median": 1000.0, "max": 10000.0}       # a poweravg-like document shows here
    assert big["keys"]["top"] == {"elapsed_ns": "integer", "gpu": "dict", "hw_model": "string", "is_delta": "bool",
                                  "kern_bootargs": "string", "kern_boottime": "integer", "kern_osversion": "string",
                                  "processor": "dict", "timestamp": "date"}
    assert big["keys"]["processor"] == {"ane_power": "real", "clusters": "array", "combined_power": "real",
                                        "cpu_power": "real", "gpu_power": "real"}
    assert big["keys"]["gpu"] == {"freq_hz": "real", "idle_ratio": "real"}
    assert small["keys"]["top"]["elapsed_ns"] == "real"
    types = {"bool", "integer", "real", "string", "date", "data", "dict", "array", "other"}
    for lay in out["layouts"]:
        for where in lay["keys"].values():
            assert set(where.values()) <= types
    for v in [*map(str, HOST.values()), "P0-Cluster", "3200000000", "24000", "2026-10-07"]:
        assert v not in text, v
    assert pl.main(["--layout", str(write_raw(tmp_path / "e.plist", b""))]) == 1
    with pytest.raises(SystemExit):
        pl.main(["--layout", "--check", str(p)])
    assert not list(tmp_path.rglob("*.json"))                            # --layout writes nothing


def _schema_example() -> dict:
    """Every key a record may carry (the stable schema)."""
    comp = {c: 0 for c in pl.COMPONENTS}
    win = {"name": 0, "kind": 0, "source": 0, "t_start": 0, "t_end": 0, "duration_s": 0, "n_samples": 0,
           "sampled_s": 0, "covered_s": 0, "overlap_s": 0, "coverage": 0, "energy_wh": comp, "mean_w": comp,
           "peak_w": comp, "gaps_over_5s": {"n": 0, "total_s": 0, "listed": [{"start": 0, "end": 0, "seconds": 0}]}}
    raw = {"name": 0, "dir": 0, "sha256": 0, "bytes": 0, "documents": 0, "samples": 0, "unreadable": 0,
           "truncated_tail": 0, "text_bytes_skipped": 0, "skipped": {"not_delta": 0, "no_time": 0, "no_power": 0},
           "combined_from_components": 0, "from_energy": 0, "first_sample_utc": 0, "last_sample_utc": 0,
           "span_s": 0, "elapsed_sum_s": 0, "covered_s": 0, "overlap_s": 0, "excess_s": 0,
           "long_samples": {"n": 0, "max_ms": 0}, "samplers_seen": 0,
           "interval_ms": {"declared": 0, "median": 0, "min": 0, "max": 0}, "model_family": 0}
    return {"schema": 0, "label": 0, "descriptive": 0, "utc": 0, "command": 0, "samplers_declared": 0,
            "interval_ms_declared": 0, "raw_files": [raw], "raw_policy": 0, "not_measured": 0, "caveats": 0,
            "method": {k: 0 for k in pl.METHOD}, "windows": [win], "windows_left_out_no_overlap": 0,
            "again_after": 0, "tool_sha256": 0}


def test_the_schema_is_stable(tmp_path):
    rec = record(tmp_path, steady(40), [window("w", 5, 35)], auto=[window("a", 1, 9, "bench")])
    ex = _schema_example()
    assert set(rec) == set(ex)
    assert set(rec["raw_files"][0]) == set(ex["raw_files"][0])
    assert set(rec["raw_files"][0]["skipped"]) == set(ex["raw_files"][0]["skipped"])
    assert set(rec["raw_files"][0]["long_samples"]) == {"n", "max_ms"}
    assert set(rec["method"]) == set(ex["method"]) and {"combined", "overlap_s", "double_count"} <= set(rec["method"])
    for w in rec["windows"]:
        assert set(w) == set(ex["windows"][0])
        for k in ("energy_wh", "mean_w", "peak_w"):
            assert list(w[k]) == list(pl.COMPONENTS) == ["combined", "cpu", "gpu", "ane"]
        assert set(w["gaps_over_5s"]) == {"n", "total_s", "listed"}
    assert rec["schema"] == "exp037 power v1"
    assert rec["label"] == LABEL and rec["descriptive"] == pl.DESCRIPTIVE
    assert rec["command"] == "/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist"
    assert rec["samplers_declared"] == ["cpu_power", "gpu_power", "ane_power"] and rec["interval_ms_declared"] == 1000
    assert rec["tool_sha256"] == common.sha256_file(EXP / "tools" / "power_log.py")
    assert rec["raw_files"][0]["interval_ms"] == {"declared": 1000, "median": 1000.0, "min": 1000.0, "max": 1000.0}
    assert [w["kind"] for w in rec["windows"]] == ["log", "explicit", "bench"]
    assert json.loads(json.dumps(rec)) == rec


def test_the_label_and_what_is_not_measured_agree_with_hypothesis_and_runbook():
    """CPU + GPU + ANE is not the whole SoC (review P3): the label, the list of what is not measured and the scale
    of the gap say so, in the tool and in both documents."""
    assert pl.LABEL == LABEL
    nm = " | ".join(pl.NOT_MEASURED)
    for part in ("the rest of the SoC", "memory controllers and fabric", "system-level cache", "media and display",
                 "DRAM", "wall (AC) power", "battery charging"):
        assert part in nm, part
    cav = " ".join(pl.CAVEATS)
    for part in ("28.6 W", "64.6 W", "different machine", "man page", "not the whole SoC", "1 Hz"):
        assert part in cav, part
    hyp = (EXP / "HYPOTHESIS.md").read_text(encoding="utf-8")
    sec = hyp.split("### Power and energy (descriptive)", 1)[1].split("\n## ", 1)[0]
    rb = (EXP / "RUNBOOK.md").read_text(encoding="utf-8")
    assert f'"{LABEL}"' in sec and LABEL in rb
    for part in ("28.6 W", "64.6 W", "DRAM", "memory controllers and fabric", "man page", "1 Hz", "poweravg",
                 '"power log"'):
        assert part in sec, part
    for text in (hyp, rb):
        for wrong in ("SoC power", "SoC energy", "SoC baseline", "costs no machine time", "own help"):
            assert wrong not in text, wrong


# -------------------------------------------------------------------------------------------------- check


def _now_docs(n: int, age_s: float) -> list[bytes]:
    """n samples, the newest ending age_s seconds ago."""
    end = int(time.time() - age_s)
    base = dt.datetime.fromtimestamp(end - n, tz=dt.timezone.utc).replace(tzinfo=None)
    return [doc(i, base=base) for i in range(1, n + 1)]


def test_check(tmp_path):
    ok = pl.check_raw(pl.read_raw(write_raw(tmp_path / "ok.plist", stream(steady(10)))), now=T0_EPOCH + 20)
    assert ok["ok"] and ok["problems"] == [] and ok["age_s"] == 10.0 and ok["model_family"] == "Mac17"
    assert (ok["documents"], ok["parsed"], ok["samples"]) == (10, 10, 10)
    assert ok["skipped"] == {"not_delta": 0, "no_time": 0, "no_power": 0}
    assert (ok["span_s"], ok["elapsed_sum_s"], ok["overlap_s"], ok["excess_s"]) == (10.0, 10.0, 0.0, 0.0)
    stale = pl.check_raw(pl.read_raw(tmp_path / "ok.plist"), now=T0_EPOCH + 1000)
    assert not stale["ok"] and stale["problems"][0].startswith("stale")
    assert pl.check_raw(pl.read_raw(tmp_path / "ok.plist"), max_age_s=0, now=T0_EPOCH + 10 ** 6)["ok"]
    empty = pl.check_raw(pl.read_raw(write_raw(tmp_path / "e.plist", b"")))
    assert not empty["ok"] and empty["problems"][0].startswith("empty") and "powermetrics refused" in \
        empty["problems"][0]
    one = pl.check_raw(pl.read_raw(write_raw(tmp_path / "1.plist", stream(steady(1)))), now=T0_EPOCH + 2)
    assert not one["ok"] and "at least 2" in one["problems"][0]
    no_ane = stream([doc(i, ane=None, combined=31000.0) for i in range(1, 6)])
    c = pl.check_raw(pl.read_raw(write_raw(tmp_path / "n.plist", no_ane)), now=T0_EPOCH + 6)
    assert not c["ok"] and c["problems"] == ["samplers missing: ane_power"]
    slow = stream([doc(5 * i, elapsed_ns=5_000_000_000) for i in range(1, 6)])
    c = pl.check_raw(pl.read_raw(write_raw(tmp_path / "s.plist", slow)), now=T0_EPOCH + 26)
    assert not c["ok"] and c["problems"] == ["median interval 5000.0 ms, expected about 1000 ms"]


def test_documents_that_parse_but_give_no_sample_say_key_layout_differs(tmp_path, identity, capsys):
    """A layout the parser does not know (here: no *_power, *_energy or combined_power key) is named as such, not
    as 'wait 30 s', and record mode refuses with the counts."""
    other = [plistlib.dumps({"is_delta": True, "elapsed_ns": 1_000_000_000, "timestamp": T0 + dt.timedelta(seconds=i),
                             "processor": {"package_watts": 30.0}, **HOST}) for i in range(1, 11)]
    p = write_raw(tmp_path / "work" / "power_20261007T045959Z.plist", stream(other))
    c = pl.check_raw(pl.read_raw(p), now=T0_EPOCH + 11)
    assert not c["ok"] and c["samples"] == 0 and c["skipped"] == {"no_power": 10, "no_time": 0, "not_delta": 0}
    assert c["problems"] == ['key layout differs: 10 document(s), 10 parse, 0 samples, skipped '
                             '{"no_power": 10, "no_time": 0, "not_delta": 0}; keep the log and run --layout on it']
    no_time = [plistlib.dumps({"is_delta": True, "elapsed_ns": 1_000_000_000, "processor": {"combined_power": 1.0}})
               for _ in range(5)]
    c = pl.check_raw(pl.read_raw(write_raw(tmp_path / "t.plist", stream(no_time))))
    assert c["problems"][0].startswith("key layout differs: 5 document(s), 5 parse, 0 samples")
    assert pl.main([str(p), "--no-auto", "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "no readable sample (10 document(s), 10 parse" in err and "--layout" in err


def test_check_cli(tmp_path, capsys):
    fresh = write_raw(tmp_path / "power_a.plist", stream(_now_docs(5, 3), tail=b"<?xml"))
    assert pl.main(["--check", str(fresh)]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["samples"] == 5 and out["truncated_tail"] and "skipped" in out
    old = write_raw(tmp_path / "power_b.plist", stream(_now_docs(5, 600)))
    assert pl.main(["--check", str(old)]) == 1
    assert pl.main(["--check", str(old), "--max-age", "0"]) == 0
    assert pl.main(["--check", str(tmp_path / "absent.plist")]) == 1
    assert pl.main(["--check", str(fresh), str(old)]) == 2
    assert not list(tmp_path.glob("*.json"))                             # --check writes nothing


# ---------------------------------------------------------------------------------------------------- cli


def test_cli_writes_one_record_and_refuses_a_duplicate(tmp_path, identity, capsys):
    raw = write_raw(tmp_path / "work" / "power_20261007T045959Z.plist", stream(steady(60)))
    res, out = results_tree(tmp_path), tmp_path / "results" / "power"
    assert pl.main([str(raw), "--results", str(res), "--out", str(out)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["ok"] and line["label"] == LABEL and line["log_combined_wh"] == pytest.approx(0.5, abs=1e-6)
    (first,) = sorted(out.glob("power_*.json"))
    assert re.fullmatch(r"power_\d{8}T\d{6}Z\.json", first.name)
    rec = json.loads(first.read_text(encoding="utf-8"))
    assert rec["raw_files"][0]["sha256"] == common.sha256_file(raw) and rec["again_after"] == []
    assert [w["name"] for w in rec["windows"]] == ["log", "gate 20261007T050000Z", "gate 20261007T050000Z phase 1"]
    assert pl.main([str(raw), "--results", str(res), "--out", str(out)]) == 1           # the same raw log again
    assert "already record" in capsys.readouterr().err
    assert len(list(out.glob("power_*.json"))) == 1
    assert pl.main([str(raw), "--results", str(res), "--out", str(out), "--again", "--no-auto"]) == 0
    second = [p for p in sorted(out.glob("power_*.json")) if p != first]
    assert len(second) == 1
    rec2 = json.loads(second[0].read_text(encoding="utf-8"))
    assert rec2["again_after"] == [first.name] and [w["name"] for w in rec2["windows"]] == ["log"]
    assert first.read_text(encoding="utf-8") == json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_a_log_recorded_while_still_growing_is_refused_by_name(tmp_path, identity, capsys):
    """Recorded before Ctrl-C, the log grows on: its sha256 changes, but a second record would count the first
    part twice, so the same file name is refused too (--again names the earlier record)."""
    raw = tmp_path / "work" / "power_20261007T045959Z.plist"
    out = tmp_path / "out"
    write_raw(raw, stream(steady(30), tail=doc(31)[:120]))
    assert pl.main([str(raw), "--no-auto", "--out", str(out)]) == 0
    write_raw(raw, stream(steady(90)))                                   # the logger ran on; then Ctrl-C
    assert pl.main([str(raw), "--no-auto", "--out", str(out)]) == 1
    assert "same name" in capsys.readouterr().err
    (first,) = out.glob("power_*.json")
    assert pl.main([str(raw), "--no-auto", "--out", str(out), "--again"]) == 0
    (second,) = [p for p in out.glob("power_*.json") if p != first]
    rec = json.loads(second.read_text(encoding="utf-8"))
    assert rec["again_after"] == [first.name] and rec["raw_files"][0]["samples"] == 90


def test_a_very_recent_last_sample_warns_that_the_logger_may_still_run(tmp_path, identity, capsys):
    raw = write_raw(tmp_path / "work" / "power_now.plist", stream(_now_docs(5, 2)))
    assert pl.main([str(raw), "--no-auto", "--out", str(tmp_path / "out")]) == 0       # a warning, not a refusal
    assert "still running" in capsys.readouterr().err


def test_cli_usage_and_refusals(tmp_path, monkeypatch, identity, capsys):
    raw = write_raw(tmp_path / "work" / "power_x.plist", stream(steady(20)))
    out = tmp_path / "out"
    base = [str(raw), "--no-auto", "--out", str(out)]
    assert pl.main(base + ["--window", "w", "2026-10-07T05:00:10Z", "2026-10-07T05:00:05Z"]) == 2
    assert pl.main(base + ["--window", "w", "yesterday", "2026-10-07T05:00:05Z"]) == 2
    assert pl.main(base + ["--window", "log", "1", "2"]) == 2
    assert pl.main(base + ["--window", "w", "1", "2", "--window", "w", "3", "4"]) == 2
    empty = write_raw(tmp_path / "work" / "power_y.plist", b"")
    assert pl.main([str(empty), "--no-auto", "--out", str(out)]) == 1
    assert pl.main(["--layout", str(raw), str(empty)]) == 2
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "x", "email": "y"})
    assert pl.main(base) == 1
    assert "refused" in capsys.readouterr().err
    assert not out.exists() or not list(out.glob("*.json"))


@pytest.mark.parametrize("where", ["results/power/power_20261007T050000Z.plist", "../../../power.plist"])
def test_a_raw_log_inside_the_repository_is_refused(tmp_path, identity, capsys, where):
    p = EXP / where                                                      # never created
    assert not p.exists()
    for argv in ([str(p), "--out", str(tmp_path / "out")], ["--check", str(p)], ["--layout", str(p)]):
        assert pl.main(argv) == 1
        assert "inside the repository" in capsys.readouterr().err
    assert not p.exists() and not (tmp_path / "out").exists()


def test_the_default_raw_log_is_the_newest_in_the_work_dir_never_a_smoke(tmp_path, monkeypatch, identity, capsys):
    work = tmp_path / "work"
    monkeypatch.setenv("EXP036_WORK", str(work))
    d = work / "exp037" / "power"
    assert pl.raw_dir() == d
    assert pl.main(["--check"]) == 1 and "no power_<UTC>.plist" in capsys.readouterr().err
    write_raw(d / "power_smoke_20261007T030000Z.plist", stream(_now_docs(5, 2)))          # sorts after any stamp
    assert pl.main(["--check"]) == 1 and "never taken by default" in capsys.readouterr().err
    write_raw(d / "power_20261007T040000Z.plist", stream(_now_docs(5, 900)))
    write_raw(d / "power_20261007T050000Z.plist", stream(_now_docs(5, 2)))
    assert pl.newest_raw().name == "power_20261007T050000Z.plist"
    assert pl.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["file"] == "power_20261007T050000Z.plist"
    out = tmp_path / "out"
    assert pl.main(["--no-auto", "--out", str(out)]) == 0
    (path,) = out.glob("power_*.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["raw_files"][0]["name"] == "power_20261007T050000Z.plist"
    assert rec["raw_files"][0]["dir"] == "$EXP036_WORK/exp037/power"     # redacted


def test_parse_time_forms():
    t = pl.parse_time("2026-10-07T05:01:12Z")
    assert t == pl.parse_time("20261007T050112Z") == pl.parse_time("2026-10-07T05:01:12+00:00")
    assert t == pl.parse_time(str(int(t))) == pl.parse_time(t) == pl.parse_time(dt.datetime(2026, 10, 7, 5, 1, 12))
    assert pl.iso(t) == "2026-10-07T05:01:12Z"
    for bad in (True, None, "yesterday", float("nan"), ["x"]):
        with pytest.raises((ValueError, TypeError)):
            pl.parse_time(bad)
    assert math.isclose(pl.parse_time(1.5e9), 1.5e9)


# ---------------------------------------------------------------------------------------- RUNBOOK and scope


def test_the_runbook_logger_lines_are_the_pinned_command():
    """sudo refuses any argument outside the mbp's pinned rule, so every logger line must be exactly it: the six
    steps' loggers, the pre-freeze smoke's logger and its `sudo -n -l` match test."""
    text = (EXP / "RUNBOOK.md").read_text(encoding="utf-8")
    want = f'sudo -n {pl.COMMAND} > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"'
    smoke = f'sudo -n {pl.COMMAND} > "$EXP036_WORK/exp037/power/power_smoke_$(date -u +%Y%m%dT%H%M%SZ).plist"'
    listing = f"sudo -n -l {pl.COMMAND}"
    blocks = "\n".join(re.findall(r"```bash\n(.*?)```", text, re.S))
    lines = [ln.strip() for ln in blocks.splitlines() if "powermetrics" in ln]
    assert set(lines) == {want, smoke, listing}, lines
    assert lines.count(want) >= 6 and lines.count(smoke) == 1 and lines.count(listing) == 1   # steps 9-12, 14, 16
    assert f"`{pl.COMMAND}`" in text                                     # the sudoers rule, quoted once
    assert text.count('`"$PY" tools/power_log.py --check`') >= 6         # each step checks the logger
    assert text.count('`"$PY" tools/power_log.py`') >= 6                 # ... and writes the record after it
    assert "tools/power_log.py --layout" in text and "--check --max-age 0" in text       # the smoke's reading
    assert not [ln for ln in blocks.splitlines() if "git add" in ln and ".plist" in ln]  # the raw log is never added
    assert pl._RAW_NAME.fullmatch("power_20261007T050112Z.plist")
    assert not pl._RAW_NAME.fullmatch("power_smoke_20261007T050112Z.plist")


def test_no_deciding_code_reads_a_power_record():
    """Descriptive only: the gate, runner, bench, analysis, scorers, port and reference never name it."""
    hits = []
    for d in ("gate", "runner", "bench", "analysis", "scorers", "port", "reference", "tasks"):
        for p in sorted((EXP / d).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts:
                s = p.read_text(encoding="utf-8", errors="replace")
                if "power_log" in s or "results/power" in s or "powermetrics" in s:
                    hits.append(p.relative_to(EXP).as_posix())
    assert hits == []
