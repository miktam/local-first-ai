"""exp_037 Amendment 1, gate fix 6.6 (diagnostics/g1_resume_20261007/PLAN.md 6.6 and 6.8).

Whether run_cell uses the scorers' reasoning split must not depend on scorers.reasoning._CHECKED, the cache of
passed tokenizer checks keyed by id(tokenizer), or on any other id()-keyed state.

Gate run 20261007T110355Z failed G1 on
tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records. Earlier in G1's process,
tests/test_bench_batch_flip.py checks a real Kolibri tokenizer through scorers and frees it, and its id stays in
_CHECKED. A later StubTok can reuse that id. runner.generate._scorer_split then took the scorers' split for it, and
the resumed run's records got split_by "scorers.reasoning.split_ids" instead of "runner.generate.split_ids"
(diagnostics/g1_resume_20261007/out/mbp_cachestate_20261007T171033Z.json, outcome "hit").

The trigger is seeded, so one invocation decides (PLAN.md 6.8, deterministic trigger): the resumed run's StubTok is
put into _CHECKED as ("kolibri", id(stub)), the key check_tokenizer writes. The oracle is symbolic and independent
of every gate value: the family table's turn-start, delimiter and EOS ids (scorers.reasoning.FAMILIES) against the
tokenizer's. _CHECKED is restored after every test.
"""

from __future__ import annotations

import pytest

import test_runner_tiny as T
from test_runner_tiny import model  # noqa: F401  (fixture: the tiny vendor checkpoint, as test_runner_tiny loads it)

from runner import generate, jsonl
from scorers import reasoning


class TableTok:
    """A tokenizer stub whose turn-start, delimiter and EOS ids are the family table's, as a real tokenizer's are."""

    unk_token_id = None

    def __init__(self, family: str, eos_shift: int = 0):
        spec = reasoning.family_spec(family)
        self.ids = {spec.turn_start: spec.turn_start_id, spec.open: spec.open_id, spec.close: spec.close_id}
        self.ids.update(zip(spec.eos, (e + eos_shift for e in spec.eos_ids)))
        self.eos_token_ids = set(spec.eos_ids)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return self.ids.get(t)


def _snapshot():
    return set(reasoning._CHECKED)


def _restore(saved):
    reasoning._CHECKED.clear()
    reasoning._CHECKED.update(saved)


def _diff(got: dict, want: dict) -> str:
    """Every differing item of the resumed file (got) against the reference (want), field by field."""
    lines = []
    for k in sorted(set(got) | set(want)):
        a, b = got.get(k), want.get(k)
        if a == b:
            continue
        if a is None or b is None:
            lines.append(f"{k}: present only in {'the reference' if a is None else 'the resumed file'}")
            continue
        for f in sorted(set(a) | set(b)):
            if a.get(f) != b.get(f):
                lines.append(f"{k}: {f}: resumed {a.get(f)!r} != reference {b.get(f)!r}")
    return "records differ:\n" + "\n".join(lines)


def _cases(family: str):
    """(prompt_ids, completion_ids) pairs in the family's ids: plain, closed and open prompts; closed, unclosed and
    none splits; a leading blank (id 11) and a trailing EOS."""
    s = reasoning.family_spec(family)
    eos = s.eos_ids[0]
    plain = [s.turn_start_id, 5, 6]
    closed = [s.turn_start_id, 5, s.open_id, s.close_id]
    opened = [s.turn_start_id, s.open_id]
    return [
        (plain, [s.open_id, 7, 8, s.close_id, 9, 10, eos]),
        (plain, [11, s.open_id, 7, s.close_id, 9]),
        (plain, [9, 10, eos]),
        (plain, [s.open_id, 7, 8]),
        (closed, [9, s.open_id, 10, eos]),
        (opened, [7, s.close_id, 9, eos]),
        (opened, [7, 8]),
    ]


def _assert_scorers_split(family: str, split, is_blank) -> None:
    """The split _scorer_split returns is scorers.reasoning's id-level split for this family (what it returned before
    the fix for every tokenizer whose ids match the table)."""
    assert split is not None
    eos = list(reasoning.family_spec(family).eos_ids)
    for prompt, completion in _cases(family):
        state = reasoning.prompt_state_ids(family, prompt)
        assert split(prompt, completion, eos, is_blank) == reasoning.split_ids(family, state, completion, is_blank, eos)


# ------------------------------------------------------- the failing test's scenario, with the trigger seeded


def test_resume_with_the_stub_in_the_check_cache_gives_identical_records(model, tmp_path, monkeypatch):  # noqa: F811
    """test_runner_tiny's resume scenario, step by step, through its own run(), with the resumed run's StubTok in
    _CHECKED. Before the fix the resumed records (q003-q005) differ from the reference in split_by; after it every
    record is equal, and _scorer_split gives None for the stub whatever the cache holds."""
    fam = reasoning.family_spec("kolibri").name
    saved = _snapshot()
    stub = T.StubTok()  # held for the whole test, so no other object can share its id
    key = (fam, id(stub))
    try:
        reasoning._CHECKED.clear()
        reasoning._CHECKED.add(key)  # the seeded trigger: the key check_tokenizer would hold for a passed check

        its = T.items(6, max_tokens=lambda k: 8 + 2 * k)
        ref = tmp_path / "ref.jsonl"
        T.run(model, ref, its, B=1, cap=20)

        p = tmp_path / "raw/S2/K8/mmlu_en_high.jsonl"

        def stop_after_three():
            recs = [r for r in jsonl.read_jsonl(p) if r.get("type") == "record"] if p.exists() else []
            return len(recs) >= 3

        summ = T.run(model, p, its, B=1, cap=20, stop=stop_after_three)
        assert summ["stopped"] and summ["n_done"] == 3
        with open(p, "ab") as f:  # killed while writing the 4th line
            f.write(b'{"completion_ids": [1, 2, 3], "key": {"it')
        done = jsonl.completed_keys("K8", "mmlu_en", "high", root=tmp_path / "raw")
        assert len(done) == 3
        assert reasoning._CHECKED == {key}
        with monkeypatch.context() as m:
            m.setattr(T, "StubTok", lambda: stub)  # run() receives the seeded stub where it calls StubTok()
            summ = T.run(model, p, its, B=1, cap=20, done=done)
        assert summ["n_todo"] == 3 and summ["n_done"] == 3
        recs, bad = jsonl.scan(p)
        assert bad == 1 and [r["type"] for r in recs if r["type"] != "record"] == ["header", "resume"]
        got, want = T.content(T.records(p)), T.content(T.records(ref))
        assert got == want, _diff(got, want)
        assert jsonl.completed_keys("K8", "mmlu_en", "high", root=tmp_path / "raw") == {(i["id"], 0) for i in its}

        # The symbolic oracle: StubTok's ids are not the table's, so every record carries the runner's split.
        assert any(reasoning.token_id(stub, t) != i for t, i in TableTok(fam).ids.items())
        assert {r["split_by"] for r in T.records(p).values()} == {"runner.generate.split_ids"}
        assert generate._scorer_split(fam, stub) is None  # seeded
        reasoning._CHECKED.clear()
        assert generate._scorer_split(fam, stub) is None  # empty
    finally:
        _restore(saved)


# ------------------------------------------------- the decision follows the family table, never the cache


@pytest.mark.parametrize("family", sorted(reasoning.FAMILIES))
def test_scorer_split_follows_the_family_table_whatever_the_cache_holds(family):
    """For every family: a tokenizer on the table gets the scorers' split and one off it gets None, with the cache
    empty, seeded with the three tokenizers' keys, or seeded with every family's keys for them. The decision equals the
    uncached check_tokenizer's (pass or raise), i.e. what _scorer_split decided before the fix whenever no stale or
    seeded id was involved."""
    saved = _snapshot()
    on, off, near = TableTok(family), T.StubTok(), TableTok(family, eos_shift=1)
    is_blank = lambda t: t == 11  # noqa: E731
    try:
        states = [
            set(),
            {(family, id(on)), (family, id(off)), (family, id(near))},
            {(f, id(x)) for f in reasoning.FAMILIES for x in (on, off, near)},
        ]
        for state in states:
            _restore(state)
            _assert_scorers_split(family, generate._scorer_split(family, on), is_blank)
            assert generate._scorer_split(family, off) is None
            assert generate._scorer_split(family, near) is None  # EOS ids count, as in check_tokenizer

        reasoning._CHECKED.clear()
        reasoning.check_tokenizer(family, on)  # the uncached check passes: the scorers' split before and after
        for tok in (off, near):
            reasoning._CHECKED.clear()
            with pytest.raises(ValueError):
                reasoning.check_tokenizer(family, tok)  # the uncached check raises: None before and after
    finally:
        _restore(saved)


def test_scorer_split_unchanged_for_the_real_kolibri_tokenizer(tokenizer_dir):
    """The real Kolibri tokenizer, as mlx_lm loads it for run_cell (TokenizerWrapper): its ids match the table, so it
    gets the scorers' split with the cache empty or holding its key, as before the fix."""
    from mlx_lm.tokenizer_utils import load as load_tok

    saved = _snapshot()
    tok = load_tok(tokenizer_dir)
    is_blank = lambda t: generate._decode(tok, [t]).strip() == ""  # noqa: E731  (run_cell's is_blank)
    try:
        for state in (set(), {("kolibri", id(tok))}):
            _restore(state)
            _assert_scorers_split("kolibri", generate._scorer_split("kolibri", tok), is_blank)
        reasoning._CHECKED.clear()
        reasoning.check_tokenizer("kolibri", tok)  # the uncached check passes: the scorers' split before and after
    finally:
        _restore(saved)  # no stale id of this tokenizer is left behind
