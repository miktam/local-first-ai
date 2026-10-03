"""Run allenai/IFBench's own checkers in place. Executed by scorers/ifbench_adapter.py inside the IFBench
venv (BUILD_SPEC §5.6, §8); it imports nothing from the kit and only the standard library besides IFBench.

    <ifbench-venv>/bin/python ifbench_driver.py --ifbench-src DIR --inputs IN.jsonl \\
        --responses RESP.jsonl --out OUT.json

IN.jsonl is IFBench's input format ({key, prompt, instruction_id_list, kwargs}), RESP.jsonl IFBench's
response format ({prompt, response}). The scoring is IFBench's own code path, in the order its run_eval.py
uses: evaluation_lib.read_prompt_list, read_prompt_to_response_dict, then
test_instruction_following_strict over every input, then test_instruction_following_loose over the same
(mutated) inputs. Order matters: the strict pass drops None-valued kwargs in place, which the loose pass
relies on. `random` is re-seeded before every check so that a checker that falls back to a random
argument cannot make two runs differ.

No network: socket connections and nltk.download raise before IFBench is imported, and every NLTK
resource IFBench asks for must already be in $NLTK_DATA (checked first; a missing one exits 3).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys

NLTK_RESOURCES = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)
SEED = 36


def _deny_network(*_args, **_kwargs):
    raise RuntimeError("exp036 IFBench driver: network access is disabled")


def _block_network() -> None:
    socket.socket.connect = _deny_network  # type: ignore[method-assign]
    socket.socket.connect_ex = _deny_network  # type: ignore[method-assign]
    socket.create_connection = _deny_network  # type: ignore[assignment]
    socket.getaddrinfo = _deny_network  # type: ignore[assignment]


def _check_nltk() -> None:
    if not os.environ.get("NLTK_DATA"):
        print("exp036 IFBench driver: NLTK_DATA is not set", file=sys.stderr)
        sys.exit(3)
    import nltk

    def _no_download(*_a, **_k):
        raise RuntimeError("exp036 IFBench driver: nltk.download is disabled; put the data in $NLTK_DATA")

    nltk.download = _no_download  # type: ignore[assignment]
    missing = []
    for res in NLTK_RESOURCES:
        try:
            nltk.data.find(res)
        except LookupError:
            missing.append(res)
    if missing:
        print(f"exp036 IFBench driver: NLTK resources missing from $NLTK_DATA: {missing}", file=sys.stderr)
        sys.exit(3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ifbench-src", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    # Nothing from the kit may shadow IFBench's imports: drop this script's directory from sys.path.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    _block_network()
    _check_nltk()
    sys.path.insert(0, os.path.abspath(args.ifbench_src))
    import evaluation_lib  # IFBench's own module, from the pinned checkout

    inputs = evaluation_lib.read_prompt_list(args.inputs)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(args.responses)

    results: dict[str, dict] = {}
    for phase, func in (
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ):
        for inp in inputs:
            random.seed(SEED)
            out = func(inp, prompt_to_response)
            rec = results.setdefault(str(inp.key), {"instruction_id_list": list(inp.instruction_id_list)})
            rec[phase] = [bool(x) for x in out.follow_instruction_list]
            rec[f"prompt_{phase}"] = bool(out.follow_all_instructions)
            rec["has_response"] = evaluation_lib.get_response_for_prompt(inp, prompt_to_response) is not None

    module_file = os.path.abspath(evaluation_lib.__file__)
    import ifbench

    payload = {
        "evaluation_lib": module_file,
        "ifbench_package": os.path.abspath(os.path.dirname(ifbench.__file__)),
        "n_inputs": len(inputs),
        "python": sys.version.split()[0],
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
