"""tools/leak_check.py, tools/shingles.py and tools/hooks/pre-push (BUILD_SPEC §5.9).

Every planted leak is assembled at run time, so this file itself passes
`leak_check.py --all`. The "withheld" question below is synthetic text written
for this test; no real GPQA, RGB or AIME text appears anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tools import common, leak_check as lc, shingles

EXP = Path(__file__).resolve().parents[1]

TAILNET_IP = "100." + "101.7.9"
HOME_PATH = "/Us" + "ers/" + "alice/notes.txt"
EMAIL_PRIVATE = "someone" + "@" + "mailhost.org"
EMAIL_RESERVED = "john.doe" + "@" + "example.com"
HF_TOKEN = "hf_" + "Ab3" * 10
PRIVATE_KEY = "-----BEGIN " + "OPENSSH " + "PRIVATE KEY-----"
MDNS = "studio-box" + ".local"

QUESTION = ("Which fictional enzyme from the purple lagoon converts zorblax into quantum glitter "
            "at dawn under heavy rain, according to the synthetic test fixture?")
OPTIONS = ["Frobnicated zorblax quantum lattice", "A purple lagoon enzyme",
           "Zorblax glitter synthase complex", "None of the above"]


def _canary() -> str:
    return "canary " + "GUID " + str(uuid.uuid4())


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _ctx(exp: Path, names=None) -> lc.Context:
    s = shingles.ShingleSet(sources=["synthetic"])
    s.add_text(QUESTION)
    for o in OPTIONS:
        s.add_text(o)
        s.add_option(o)
    return lc.Context(exp, s, names=names if names is not None else {"whoami": None, "localhostname": None, "private": []})


def _kinds(res, severity="finding"):
    items = res.findings if severity == "finding" else res.warnings
    return sorted({f.kind for f in items})


@pytest.fixture
def exp(tmp_path) -> Path:
    d = tmp_path / "exp_036_kolibri_local_eval"
    d.mkdir()
    return d


# ---------------------------------------------------------------- normalisation

def test_normalisation_and_shingles():
    assert shingles.normalise_words("Hello,   WORLD! x-ray 3.14 ß") == ["hello", "world", "xray", "314", "ß"]
    a = shingles.shingle_prefixes(QUESTION)
    b = shingles.shingle_prefixes(QUESTION.upper().replace(",", " ;"))
    assert a and a == b
    assert all(len(h) == 12 for h in a)
    assert shingles.option_hash("None of the above") is None          # generic phrase
    assert shingles.option_hash("two words") is None                   # < 4 words
    assert shingles.option_hash("three plain words") is None           # < 4 words (Amendment 1, 2026-10-04)
    assert len(shingles.option_hash(OPTIONS[0])) == 64


def test_shingle_file_round_trip_is_byte_identical(tmp_path):
    p1, p2 = tmp_path / "a.sha256", tmp_path / "b.sha256"
    shingles.write_shingle_file(p1, [QUESTION], OPTIONS)
    shingles.write_shingle_file(p2, [QUESTION], list(reversed(OPTIONS)))
    assert p1.read_bytes() == p2.read_bytes()
    s = shingles.read_shingle_file(p1)
    assert s.prefixes == shingles.shingle_prefixes(QUESTION) | set().union(*(shingles.shingle_prefixes(o) for o in OPTIONS))
    assert len(s.options) == 3
    assert QUESTION.split()[3] not in p1.read_text()     # hashes only


# ---------------------------------------------------------------- planted leaks

def test_planted_gpqa_like_shingles_are_found(exp):
    ctx = _ctx(exp)
    quoted = "As the item asked: " + QUESTION.lower().replace("?", "") + " — interesting."
    _write(exp, "docs/post.md", f"intro line\n{quoted}\n")
    res = lc.check([exp / "docs" / "post.md"], ctx=ctx)
    assert "withheld_shingle" in _kinds(res)
    f = next(f for f in res.findings if f.kind == "withheld_shingle")
    assert f.line == 2 and "zorblax" not in f.detail.lower()


def test_short_option_is_found_and_generic_phrase_is_not(exp):
    ctx = _ctx(exp)
    _write(exp, "a.md", "The model chose frobnicated zorblax, quantum lattice.\n")
    _write(exp, "b.md", "The answer is none of the above for this public item.\n")
    assert "withheld_option" in _kinds(lc.check([exp / "a.md"], ctx=ctx))
    assert lc.check([exp / "b.md"], ctx=ctx).ok


def test_withheld_severity_thresholds(exp):
    ctx = _ctx(exp)
    words = shingles.normalise_words(QUESTION)
    eight = " ".join(words[:8])                     # exactly one shingle
    nine = " ".join(words[:9])                      # two shingles
    _write(exp, "one.md", f"generic {eight} end\n")
    _write(exp, "two.md", f"quote {nine} end\n")
    one, two = lc.check([exp / "one.md"], ctx=ctx), lc.check([exp / "two.md"], ctx=ctx)
    assert one.ok and "withheld_shingle" in _kinds(one, "warning")
    assert "withheld_shingle" in _kinds(two)
    # Options: a 4-word option in a public raw output is a warning, in a kit file a finding;
    # a 5-word option is a finding everywhere.
    s = shingles.ShingleSet(sources=["synthetic"])
    s.add_option("plasma lagoon zorblax cascade")
    s.add_option("fully frobnicated plasma lagoon cascade")
    c2 = lc.Context(exp, s, names={"whoami": None, "localhostname": None, "private": []})
    rec = {"key": {"task": "mmlu_en"}, "text": "so: plasma lagoon zorblax cascade."}
    _write(exp, "results/raw/S2/K8/mmlu_en_high.jsonl", json.dumps(rec) + "\n")
    _write(exp, "doc.md", "plasma lagoon zorblax cascade\n")
    raw = lc.check([exp / "results/raw/S2/K8/mmlu_en_high.jsonl"], ctx=c2)
    assert raw.ok and "withheld_option" in _kinds(raw, "warning")
    assert "withheld_option" in _kinds(lc.check([exp / "doc.md"], ctx=c2))
    rec["text"] = "so: fully frobnicated plasma lagoon cascade."
    _write(exp, "results/raw/S2/K8/mmlu_en_high.jsonl", json.dumps(rec) + "\n")
    assert "withheld_option" in _kinds(lc.check([exp / "results/raw/S2/K8/mmlu_en_high.jsonl"], ctx=c2))


def test_public_text_is_subtracted_from_the_shingle_file(tmp_path):
    public = "Which of the following statements about the fictional lagoon is correct here today?"
    withheld = "Which of the following statements about the fictional lagoon is wrong, and why so?"
    p = tmp_path / "w.sha256"
    stats_all = shingles.write_shingle_file(tmp_path / "all.sha256", [withheld], ["the fictional lagoon is"])
    shingles.write_shingle_file(p, [withheld], ["the fictional lagoon is"], public_texts=[public])
    s = shingles.read_shingle_file(p)
    assert len(s.prefixes) < stats_all["prefixes"]
    assert not (s.prefixes & shingles.shingle_prefixes(public))
    assert not s.options                                  # the option also occurs in public text
    assert "removed as also public" in p.read_text().splitlines()[1]


def test_ips_paths_tokens_keys_emails_hosts(exp):
    ctx = _ctx(exp)
    text = "\n".join([
        f"tailnet {TAILNET_IP} here",
        "the documented range 100.64.0.0/10 is fine",
        f"path {HOME_PATH}",
        "shared /Users/Shared/x is not a home",
        f"token {HF_TOKEN}",
        PRIVATE_KEY,
        f"mail {EMAIL_PRIVATE}",
        f"reserved {EMAIL_RESERVED}",
        "allowed hello@localfirstai.eu and noreply@anthropic.com",
        "remote git@github.com:miktam/local-first-ai.git",
        f"host {MDNS}",
        "env/exp036.local.env and threading.local() are not hosts",
        _canary(),
    ])
    _write(exp, "notes.md", text + "\n")
    res = lc.check([exp / "notes.md"], ctx=ctx)
    found = {(f.kind, f.line) for f in res.findings}
    assert ("tailnet_ip", 1) in found and not any(k == "tailnet_ip" and ln == 2 for k, ln in found)
    assert ("home_path", 3) in found and not any(ln == 4 for _, ln in found)
    assert ("hf_token", 5) in found
    assert ("private_key", 6) in found
    assert ("email", 7) in found
    assert ("mdns_host", 11) in found
    assert ("canary", 13) in found
    assert not any(ln in (9, 10, 12) for _, ln in found)
    assert {(w.kind, w.line) for w in res.warnings} == {("email", 8)}


def test_raw_output_matches_are_warnings_but_withheld_text_is_a_finding(exp):
    ctx = _ctx(exp)
    rec = {"key": {"arm": "K8", "task": "ifbench", "effort": "high", "item": "1", "pass": 0},
           "text": f"Contact {EMAIL_RESERVED} or {EMAIL_PRIVATE} at {TAILNET_IP}."}
    _write(exp, "results/raw/S2/K8/ifbench_high.jsonl", json.dumps(rec) + "\n")
    res = lc.check([exp / "results/raw/S2/K8/ifbench_high.jsonl"], ctx=ctx)
    assert res.ok
    assert {"email", "tailnet_ip"} <= set(_kinds(res, "warning"))
    rec["text"] = "Reasoning: " + QUESTION
    _write(exp, "results/raw/S2/K8/ifbench_high.jsonl", json.dumps(rec) + "\n")
    res = lc.check([exp / "results/raw/S2/K8/ifbench_high.jsonl"], ctx=ctx)
    assert "withheld_shingle" in _kinds(res)


def test_withheld_set_records_carry_no_text_or_gold(exp):
    ctx = _ctx(exp)
    good = {"key": {"arm": "K8", "task": "gpqa_de", "effort": "high", "item": "x1", "pass": 0},
            "text_sha256": "0" * 64, "completion_ids_sha256": "1" * 64, "truncated": False}
    bad = dict(good, text="Antwort: B")
    _write(exp, "results/raw/S2/K8/gpqa_de_high.jsonl", json.dumps(good) + "\n")
    assert lc.check([exp / "results/raw/S2/K8/gpqa_de_high.jsonl"], ctx=ctx).ok
    _write(exp, "results/raw/S2/K8/gpqa_de_high.jsonl", json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    res = lc.check([exp / "results/raw/S2/K8/gpqa_de_high.jsonl"], ctx=ctx)
    assert [(f.kind, f.line) for f in res.findings] == [("withheld_field", 2)]
    # A summary nesting a withheld task, and a withheld manifest with gold.
    _write(exp, "results/pilot_summary_x.json", json.dumps({"K8": {"rgb_cb": {"diagnoses": [{"text": "x"}]}}}))
    _write(exp, "tasks/manifests/gpqa_diamond_en.json", json.dumps([{"id": "a", "item_sha256": "0", "gold": "C"}]))
    for rel in ("results/pilot_summary_x.json", "tasks/manifests/gpqa_diamond_en.json"):
        assert "withheld_field" in _kinds(lc.check([exp / rel], ctx=ctx)), rel
    # Public sets may carry text and gold; synthetic test fixtures are not records.
    _write(exp, "tasks/manifests/mmlu_prox_en.json", json.dumps([{"id": "a", "gold": "C", "category": "law"}]))
    _write(exp, "tests/fixtures/scorers/rgb_cases.json", json.dumps([{"task": "rgb_cb", "text": "stub"}]))
    for rel in ("tasks/manifests/mmlu_prox_en.json", "tests/fixtures/scorers/rgb_cases.json"):
        assert lc.check([exp / rel], ctx=ctx).ok, rel


def test_refused_types_and_sizes(exp):
    ctx = _ctx(exp)
    (exp / "report.pdf").write_bytes(b"%PDF-1.4 tiny")
    _write(exp, "big.txt", "a" * (5_000_001))
    _write(exp, "results/raw/S2/K8/mmlu_en_high.jsonl", "b" * (6_000_000))
    assert _kinds(lc.check([exp / "report.pdf"], ctx=ctx)) == ["refused_type"]
    assert _kinds(lc.check([exp / "big.txt"], ctx=ctx)) == ["too_large"]
    assert lc.check([exp / "results/raw/S2/K8/mmlu_en_high.jsonl"], ctx=ctx).ok


def test_allowlist_and_shingle_file_format(exp):
    ctx = _ctx(exp)
    _write(exp, "tests/fixtures/leak/planted.md", f"{TAILNET_IP} {HOME_PATH}\n")
    assert lc.check([exp / "tests/fixtures/leak/planted.md"], ctx=ctx).ok
    shingles.write_shingle_file(exp / "tools" / "withheld_shingles.sha256", [QUESTION], OPTIONS)
    assert lc.check([exp / "tools" / "withheld_shingles.sha256"], ctx=ctx).ok
    with open(exp / "tools" / "withheld_shingles.sha256", "a") as f:
        f.write("this line is text, not a hash\n")
    assert _kinds(lc.check([exp / "tools" / "withheld_shingles.sha256"], ctx=ctx)) == ["shingle_file_format"]


def test_runtime_names(exp):
    names = {"whoami": "zq_user7", "localhostname": "zq-host", "private": ["Doeson"]}
    ctx = _ctx(exp, names)
    _write(exp, "doc.md", "built on zq-host by zq_user7; reviewed by A. doeson\n")
    _write(exp, "results/preflight_x.json", json.dumps({"label": "zq-host"}) + "\n")
    res = lc.check([exp / "doc.md"], ctx=ctx)
    assert {"username", "private_name"} <= set(_kinds(res))
    assert "hostname" in _kinds(res, "warning") and "hostname" not in _kinds(res)
    assert "hostname" in _kinds(lc.check([exp / "results/preflight_x.json"], ctx=ctx))


def test_no_withheld_source_exits_3(exp, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXP036_DATA", str(tmp_path / "no_data"))
    monkeypatch.setenv("EXP036_PRIVATE", str(tmp_path / "no_private"))
    with pytest.raises(lc.NoWithheldSource):
        lc.check([], all_files=True, exp_dir=exp)
    monkeypatch.setattr(lc, "SHINGLE_FILE_REL", "tools/does_not_exist.sha256")
    assert lc.main(["--all"]) == 3
    assert lc.main(["--all", "--no-gpqa-source", "--json"]) in (0, 1)
    out = capsys.readouterr().out
    assert '"withheld_check": "skipped (--no-gpqa-source)"' in out


def test_rgb_source_is_read_on_the_fly(tmp_path, monkeypatch):
    rgb = tmp_path / "data" / "RGB-src" / "data"
    rgb.mkdir(parents=True)
    row = {"id": 0, "query": "Which synthetic team won the imaginary lagoon cup in the year twenty fifty?",
           "answer": [["The Purple Zorblaxes"]], "positive": ["doc"], "negative": ["doc"]}
    (rgb / "en.json").write_text(json.dumps(row) + "\n")
    s, warns = shingles.collect_from_sources(tmp_path / "data", tmp_path / "private")
    assert s and any("RGB" in x for x in s.sources)
    assert s.matches(shingles.normalise_words("well: which synthetic team won the imaginary lagoon cup in"))
    # A 3-word RGB answer is not hashed (Amendment 1, 2026-10-04: options start at 4 words).
    # RGB raw outputs are withheld anyway; a bare short answer reveals nothing of the dataset.
    assert not s.matches(shingles.normalise_words("it was the purple zorblaxes, clearly"))


def test_the_tools_area_is_clean():
    """Pre-push checklist item 5, on this area's own files (warnings allowed).
    The whole-directory run is the integrator's pre-push step."""
    paths = [EXP / "tools", EXP / "env", *sorted((EXP / "tests").glob("test_tools_*.py"))]
    paths += [p for p in (EXP / "README.md", EXP / ".gitignore", EXP / "NOTICE.addendum.tools.md") if p.exists()]
    res = lc.check(paths, no_gpqa_source=True, exp_dir=EXP)
    assert res.ok, "\n".join(f.fmt() for f in res.findings)


# ---------------------------------------------------------------- git modes and the hook

def _git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env)


@pytest.fixture
def repo(tmp_path):
    """A repository with the experiment path, a bare remote and one pushed commit."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    work = tmp_path / "work_repo"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.name", "Miktam")
    _git(work, "config", "user.email", "hello@localfirstai.eu")
    exp = work / "tasks" / "chronos" / "exp_036_kolibri_local_eval"
    _write(exp, "notes.md", f"already public line {MDNS}\nsecond line\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "base")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "-q", "-u", "origin", "main")
    return work, exp


def test_range_mode_checks_every_unpushed_commit(repo):
    work, exp = repo
    ctx = _ctx(exp)
    # A line added and removed again before the push is still in the pushed history.
    _write(exp, "notes.md", f"already public line {MDNS}\nsecond line\nleaky {TAILNET_IP}\n")
    _git(work, "commit", "-q", "-am", "add leak")
    _write(exp, "notes.md", f"already public line {MDNS}\nsecond line changed\n")
    _git(work, "commit", "-q", "-am", "remove leak")
    res = lc.check([], ranges=["@{u}..HEAD"], ctx=ctx)
    kinds = [(f.kind, f.line) for f in res.findings]
    assert ("tailnet_ip", 3) in kinds
    assert not any(k == "mdns_host" for k, _ in kinds), "a line already upstream was flagged"
    assert all(f.commit and f.commit != "index" for f in res.findings)


def test_staged_mode(repo):
    work, exp = repo
    ctx = _ctx(exp)
    _write(exp, "new.md", f"staged {HOME_PATH}\n")
    _git(work, "add", "-A")
    res = lc.check([], staged=True, ctx=ctx)
    assert [(f.kind, f.commit) for f in res.findings] == [("home_path", "index")]
    _write(exp, "results/raw/S2/K8/rgb_cb_high.jsonl",
           json.dumps({"key": {"task": "rgb_cb"}, "text": "x"}) + "\n")
    (exp / "x.parquet").write_bytes(b"PAR1")
    _git(work, "add", "-f", "-A")
    kinds = _kinds(lc.check([], staged=True, ctx=ctx))
    assert {"withheld_field", "refused_type", "home_path"} <= set(kinds)


def test_patch_parser_handles_plus_lines_inside_hunks():
    patch = ("\x00commit abc\n\ndiff --git a/f.md b/f.md\n--- a/f.md\n+++ b/f.md\n"
             "@@ -0,0 +1,2 @@\n+++ looks like a header\n+normal\n")
    out = list(lc.parse_patch(patch))
    assert out == [("abc", "f.md", [(1, "++ looks like a header"), (2, "normal")])]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_pre_push_hook_blocks_a_leak(repo):
    work, exp = repo
    shutil.copytree(EXP / "tools", exp / "tools",
                    ignore=shutil.ignore_patterns("__pycache__", "withheld_shingles.sha256"))
    shingles.write_shingle_file(exp / "tools" / "withheld_shingles.sha256", [QUESTION], OPTIONS)
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "kit")
    hooks = Path(_git(work, "rev-parse", "--git-path", "hooks").stdout.strip())
    hooks = hooks if hooks.is_absolute() else work / hooks
    hooks.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXP / "tools" / "hooks" / "pre-push", hooks / "pre-push")
    os.chmod(hooks / "pre-push", 0o755)
    env = dict(os.environ, PY=sys.executable)
    ok = subprocess.run(["git", "push", "-q", "origin", "main"], cwd=work, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr + ok.stdout
    _write(exp, "post.md", "Item text: " + QUESTION + "\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "leak")
    blocked = subprocess.run(["git", "push", "-q", "origin", "main"], cwd=work, env=env, capture_output=True,
                             text=True)
    assert blocked.returncode != 0
    assert "withheld_shingle" in blocked.stdout + blocked.stderr


# ---------------------------------------------------------------- review fixes 2026-10-03

PASSAGE = ("The synthetic harbour council of Glimmerport voted on Tuesday to rebuild the copper lighthouse "
           "after the winter storms, said a spokesperson for the invented municipal utility.")
WEB_DE = ("Im erfundenen Dorf Wolkenbach eröffnet am Samstag ein kleiner Laden für handgemachte Kerzen, "
          "berichtet die ausgedachte Lokalzeitung ohne jeden echten Bezug.")


def _local_world(tmp_path, exp):
    """A synthetic $EXP036_DATA with RGB-shaped documents and a FineWeb-2-shaped parquet (no real text)."""
    data = tmp_path / "data"
    rgbd = data / "RGB-src" / "data"
    rgbd.mkdir(parents=True)
    row = {"id": 0, "query": "Which invented council voted?", "answer": ["Glimmerport"],
           "positive": [PASSAGE], "negative": ["An unrelated synthetic passage about nothing in particular at all."]}
    (rgbd / "en.json").write_text(json.dumps(row) + "\n")
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    fw = data / "fineweb-2" / "data" / "deu_Latn" / "test"
    fw.mkdir(parents=True)
    pq.write_table(pa.table({"text": [WEB_DE, "kurz"]}), str(fw / "000_00000.parquet"))
    ctx = _ctx(exp)
    assert ctx.enable_local(data)
    return ctx


def test_local_corpora_catch_rgb_documents_and_fineweb_rows(tmp_path, exp):
    ctx = _local_world(tmp_path, exp)
    p1 = _write(exp, "docs/notes.md", "Notes\n\n" + PASSAGE + "\n")
    p2 = _write(exp, "docs/web.md", "Zitat: " + WEB_DE + "\n")
    p3 = _write(exp, "results/raw/S2/K8/mmlu_en_high.jsonl", json.dumps({"task": "mmlu_en", "answer": PASSAGE}) + "\n")
    p4 = _write(exp, "results/raw/S2/K8/rgb_cb_high.jsonl", json.dumps({"task": "rgb_cb", "note": PASSAGE}) + "\n")
    res = lc.check([p1, p2], ctx=ctx)
    assert _kinds(res) == ["local_corpus"] and len(res.findings) == 2
    ctx = _local_world(tmp_path / "b", exp)
    assert lc.check([p3], ctx=ctx).ok                   # public raw output: never matched against the corpora
    ctx = _local_world(tmp_path / "c", exp)
    assert "local_corpus" in _kinds(lc.check([p4], ctx=ctx))   # a withheld-set raw file carries no text


def test_local_corpora_ignore_public_kit_text_and_digit_runs(tmp_path, exp):
    _write(exp, "LICENSE-APACHE-2.0", PASSAGE + "\n")    # the kit's own public text holds the same words
    ctx = _local_world(tmp_path, exp)
    p = _write(exp, "docs/licence_quote.md", PASSAGE + "\n")
    assert lc.check([p], ctx=ctx).ok
    assert shingles.all_digits("1 2 3 4 5 6 7 8") and not shingles.all_digits("1 2 3 4 5 6 7 x")
    assert shingles.shingle_prefixes("1 2 3 4 5 6 7 8 9") == set()


def test_eight_word_options_and_short_questions_are_findings(exp):
    s = shingles.ShingleSet()
    eight = "zorblax glitter lagoon enzyme converts quantum purple dawn"
    s.add_text(eight)
    s.add_option(eight)
    ctx = lc.Context(exp, s, names={"whoami": None, "localhostname": None, "private": []})
    p = _write(exp, "docs/q.md", "Failed item: " + eight + "\n")
    assert "withheld_option" in _kinds(lc.check([p], ctx=ctx))
    s2 = shingles.ShingleSet()
    shingles._add_rows(s2, [{"Question": "Which zorblax enzyme glows?", "Explanation": "x"}])
    assert shingles.option_hash("Which zorblax enzyme glows?") in s2.options


def test_binary_non_utf8_and_archives_are_findings_in_the_kit(exp):
    ctx = _ctx(exp)
    p1 = exp / "docs" / "u16.txt"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(("Q: " + QUESTION).encode("utf-16"))
    p2 = exp / "results" / "raw.jsonl.gz"
    p2.parent.mkdir(parents=True, exist_ok=True)
    import gzip

    p2.write_bytes(gzip.compress(b"{}"))
    p3 = exp / "docs" / "nul.md"
    p3.write_bytes(b"\x00" + QUESTION.encode())
    assert "binary_file" in _kinds(lc.check([p1], ctx=ctx))
    assert set(_kinds(lc.check([p2], ctx=ctx))) >= {"refused_type"}
    assert "binary_file" in _kinds(lc.check([p3], ctx=ctx))


def test_tailscale_ipv6_and_json_escaped_home_paths(exp):
    ctx = _ctx(exp)
    p = _write(exp, "docs/net.md", "reachable at fd7a:115c:a1e0:" + "ab12:4843:cd96:6264:fb54\n")
    q = _write(exp, "docs/path.json", '{"p": "\\/Us' + 'ers\\/alice\\/models"}\n')
    assert _kinds(lc.check([p], ctx=ctx)) == ["tailnet_ip"]
    assert _kinds(lc.check([q], ctx=ctx)) == ["home_path"]


def test_the_checker_itself_is_not_exempt_and_fixtures_keep_the_token_rules(exp):
    ctx = _ctx(exp)
    assert lc.check([EXP / "tools" / "leak_check.py"], ctx=lc.Context(EXP, shingles.ShingleSet(),
                    names={"whoami": None, "localhostname": None, "private": []})).ok
    p = _write(exp, "tools/leak_check.py", "# debug " + HF_TOKEN + "\n")
    assert "hf_token" in _kinds(lc.check([p], ctx=ctx))
    f = _write(exp, "tests/fixtures/leak/planted_token.md", HF_TOKEN + "\n")
    assert "hf_token" in _kinds(lc.check([f], ctx=ctx))
