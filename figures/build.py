#!/usr/bin/env python3
"""
Regenerate the localfirstai.eu figures from the files published in this repository.

  python3 figures/build.py --blog ~/local-first-ai-blog            # rebuild every figure in place
  python3 figures/build.py --blog ~/local-first-ai-blog --check    # report what would change, write nothing

Each module in figures/posts/ names its post (POST = "<file>.md") and returns {name: html} from figures(). In the post,
a figure lives between <!-- lfa-fig:<name> --> and <!-- /lfa-fig:<name> -->; the markers are placed once, by hand, where
the figure belongs, and build.py only ever replaces what is between them. The shared style is written into the blog's
layouts/partials/extend_head.html between <!-- lfa-viz-css --> markers.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import lfa_viz  # noqa: E402


def load(mod_path: Path):
    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def put_between(text: str, start: str, end: str, body: str) -> tuple[str, bool]:
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(text):
        raise KeyError(start)
    new = pat.sub(lambda _: f"{start}\n{body}\n{end}", text, count=1)
    return new, new != text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    blog = Path(a.blog).expanduser()
    head = blog / "layouts/partials/extend_head.html"
    t = head.read_text()
    block = f"<style>\n{lfa_viz.CSS}\n</style>"
    if "<!-- lfa-viz-css -->" not in t:
        t = t.rstrip("\n") + "\n<!-- lfa-viz-css -->\n<!-- /lfa-viz-css -->\n"
    t2, changed = put_between(t, "<!-- lfa-viz-css -->", "<!-- /lfa-viz-css -->", block)
    print(f"{'would update' if a.check else 'updated'} head css" if changed else "head css unchanged")
    if changed and not a.check:
        head.write_text(t2)
    for mod_path in sorted((HERE / "posts").glob("*.py")):
        if a.only and a.only not in mod_path.stem:
            continue
        m = load(mod_path)
        post = blog / "content/posts" / m.POST
        text = post.read_text()
        figs = m.figures(REPO)
        for name, body in figs.items():
            try:
                text, ch = put_between(text, f"<!-- lfa-fig:{name} -->", f"<!-- /lfa-fig:{name} -->", body)
            except KeyError:
                print(f"  MISSING marker lfa-fig:{name} in {m.POST}")
                continue
            print(f"  {m.POST} · {name}: {'changed' if ch else 'unchanged'}")
        if not a.check:
            post.write_text(text)


if __name__ == "__main__":
    main()
