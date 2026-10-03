# SPDX-License-Identifier: MIT
"""Build and verify the gate texts (HYPOTHESIS Phase 0 "Gate text"; BUILD_SPEC §5.3).

Committed under gate/texts/ (the gate consumes the ids):
  T1  EN, our exp_035 post (2026-09-22), first 1,536 Kolibri tokens of the body
  T2  EN, our Malaga-AI post (2026-09-30), first 1,536 tokens of the body
  T3  DE, prose written by Claude for exp_036 (1,687 tokens before the cut)
  T4  DE, Grundgesetz Art. 1-19 (amtliches Werk, § 5 UrhG)
  T7  EN chat render at effort high with a closed think block, exactly 1,536 tokens
  T8  DE chat render at effort none, exactly 1,536 tokens
  G5  the 8 prompts of the G5 greedy check (4 EN, 4 DE; efforts none and high; >= 600 tokens)
  src/T{1,2,3,4}_source.txt   the full source texts (T9 uses them "in full")
  src/SOURCES.json            where each source came from (path, sha256, URL, rule)
  MANIFEST.json               ids sha256, text sha256, counts, rules, and the
                              FineWeb-2 row indices / token counts / sha256 of T5, T6, T9

Never committed (web text), written to $EXP036_WORK/gate_texts/ by rule:
  T5, T6  the first two FineWeb-2 deu_Latn test documents with >= 1,536 Kolibri
          tokens at file index >= 5,000 (disjoint from H5's first 5,000)
  T9      16,384 tokens: T1, T4 and T3 sources in full, then T5, T6 and the
          following documents in file order, joined with the ids of "\\n\\n"

Ids come from the raw `tokenizers` tokenizer with add_special_tokens=False
(G0 checks that the harness tokenizer gives the same ids). A cut that ends
inside a multi-byte character (U+FFFD in the decoded text) advances the source
start by one word and cuts again (BUILD_SPEC §5.3).

CLI (mini, build time; sources default to the committed src/ files):
  build_gate_text.py [--fineweb PARQUET] [--blog POSTS_DIR] [--gg-zip ZIP] [--tok DIR]
  build_gate_text.py --check                  rebuild in memory, compare with gate/texts
mbp (RUNBOOK step 7):
  build_gate_text.py --work-only --fineweb PARQUET
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate import common
from gate.common import TEXTS_DIR, ids_sha256, sha256_bytes, sha256_file

N_TOKENS = 1536
T9_TOKENS = 16384
FINEWEB_START = 5000
FINEWEB_MIN_TOKENS = 1536
G5_MIN_TOKENS = 600
SRC_DIR = TEXTS_DIR / "src"
MANIFEST = TEXTS_DIR / "MANIFEST.json"
SEPARATOR = "\n\n"

PLAIN = {
    # id: (file stem, language)
    "T1": ("T1_exp035_post", "en"),
    "T2": ("T2_malaga_ai_post", "en"),
    "T3": ("T3_de_prose_claude", "de"),
    "T4": ("T4_grundgesetz_art1_19", "de"),
}
BLOG_POSTS = {
    "T1": "2026-09-22-we-trained-it-three-times-then-stopped.md",
    "T2": "2026-09-30-the-part-that-looked-fine.md",
}
GG_URL = "https://www.gesetze-im-internet.de/gg/xml.zip"
GG_ZIP_SHA256 = "b0dbbb71c7d04ed323f16c1926d1826e640103a531458dd52892864b7705fc37"
FINEWEB_REL = "fineweb-2/data/deu_Latn/test/000_00000.parquet"  # under $EXP036_DATA
FINEWEB_SHA256 = "f370564f527a10cf125f267d8f10175d6a6366390480620e90a469bbf80a2a46"
FINEWEB_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
T3_LABEL = "written by Claude for exp_036"

# T7 / T8 (chat renders). The user turn and the assistant content are
# paragraphs of our own committed sources, taken after the 1,536-token cut of
# T2 (T7) and T4 (T8); the T7 reasoning paragraph is written by Claude for exp_036.
T7_INSTRUCTION = "Here is part of a report from a local AI meetup. Continue it in the same voice.\n\n"
T7_REASONING = (
    "The user wants the report to continue in the same voice. The excerpt walks through a community "
    "showcase of evaluation projects, one talk at a time: what was measured, what was found, and what the "
    "speaker's own caveats said. The continuation should keep that structure, stay plain and factual, and "
    "not add claims that are not in the talks."
)
T8_INSTRUCTION = "Hier ist ein Auszug aus dem Grundgesetz. Setze den Text mit den folgenden Artikeln fort.\n\n"
CHAT_USER_MIN_TOKENS = 450

G5_INSTRUCTION = {
    "en": "Read the following excerpt from a blog post and summarise its main points in a few sentences.\n\n",
    "de": "Lies den folgenden Auszug und fasse die wichtigsten Punkte in wenigen Sätzen zusammen.\n\n",
}
# (id, source, language, effort, start as a fraction of the source's paragraphs)
G5_SPECS = (
    ("G5_en_none_1", "T1", "en", "none", 0.25),
    ("G5_en_high_1", "T1", "en", "high", 0.60),
    ("G5_en_high_2", "T2", "en", "high", 0.25),
    ("G5_en_none_2", "T2", "en", "none", 0.60),
    ("G5_de_none_1", "T4", "de", "none", 0.25),
    ("G5_de_high_1", "T4", "de", "high", 0.60),
    ("G5_de_high_2", "T3", "de", "high", 0.00),
    ("G5_de_none_2", "T3", "de", "none", 0.40),
)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def blog_body(text: str) -> str:
    """Everything after the closing '---' line of the front matter."""
    if not text.startswith("---\n"):
        raise ValueError("post does not start with a front-matter block")
    end = text.index("\n---\n", 4)
    return text[end + len("\n---\n"):]


def grundgesetz_art1_19(zip_path: Path) -> tuple[str, dict]:
    """Art 1 ... Art 19 (with 12a, 16a, 17a) in document order: the article
    label on its own line, then each <P> paragraph on its own line; articles
    separated by a blank line; footnotes excluded."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected one XML file, found {names}")
        raw = z.read(names[0])
    root = ET.fromstring(raw)
    arts, on = [], False
    for norm in root:
        label = norm.find("metadaten").findtext("enbez")
        if label == "Art 1":
            on = True
        if on:
            content = norm.find("textdaten/text/Content")
            paras = ["".join(p.itertext()).strip() for p in content.findall("P")]
            arts.append(label + "\n" + "\n".join(p for p in paras if p))
        if label == "Art 19":
            break
    if not arts or not arts[-1].startswith("Art 19"):
        raise ValueError("Art 1 ... Art 19 not found")
    return "\n\n".join(arts) + "\n", {
        "xml_file": names[0],
        "xml_sha256": sha256_bytes(raw),
        "builddate": root.get("builddate"),
        "articles": [a.split("\n", 1)[0] for a in arts],
    }


def write_sources(blog_dir: Path | None, gg_zip: Path | None) -> dict:
    """(Re)write src/T1, T2, T4 from their origins and src/SOURCES.json.
    T3 is our own committed text and is only hashed."""
    origins = common.read_json(SRC_DIR / "SOURCES.json") if (SRC_DIR / "SOURCES.json").is_file() else {}
    if blog_dir is not None:
        for tid, name in BLOG_POSTS.items():
            post = Path(blog_dir) / name
            body = blog_body(post.read_text(encoding="utf-8"))
            (SRC_DIR / f"{tid}_source.txt").write_text(body, encoding="utf-8")
            origins[tid] = {
                "origin": f"local-first-ai-blog/content/posts/{name}",
                "origin_sha256": sha256_file(post),
                "rule": "front matter removed: the body is everything after the closing '---' line",
                "published": name[:10],
            }
    if gg_zip is not None:
        zsha = sha256_file(gg_zip)
        if zsha != GG_ZIP_SHA256:
            raise SystemExit(f"{gg_zip}: sha256 {zsha} != pinned {GG_ZIP_SHA256}")
        text, info = grundgesetz_art1_19(Path(gg_zip))
        (SRC_DIR / "T4_source.txt").write_text(text, encoding="utf-8")
        origins["T4"] = {
            "origin": GG_URL,
            "origin_sha256": zsha,
            "rule": "Art 1 to Art 19 (incl. 12a, 16a, 17a); label line, one line per <P>, blank line between articles; footnotes excluded",
            "licence": "amtliches Werk, § 5 UrhG (public domain)",
            **info,
        }
    origins["T3"] = {
        "origin": "gate/texts/src/T3_source.txt",
        "label": T3_LABEL,
        "rule": "German prose written for exp_036 after Kolibri's 2026-06-18 cutoff; Andrei's choice 2026-10-03",
    }
    for tid in PLAIN:
        origins[tid]["source_sha256"] = sha256_file(SRC_DIR / f"{tid}_source.txt")
    common.write_json(SRC_DIR / "SOURCES.json", origins)
    return origins


def read_source(tid: str) -> str:
    return (SRC_DIR / f"{tid}_source.txt").read_text(encoding="utf-8")


def paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---------------------------------------------------------------------------
# Cutting and rendering
# ---------------------------------------------------------------------------


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def decode(tok, ids) -> str:
    return tok.decode(list(ids), skip_special_tokens=False)


def cut_plain(tok, text: str, n: int = N_TOKENS) -> dict:
    """The first n tokens of `text`; on a cut inside a multi-byte character
    the start advances by one word (BUILD_SPEC §5.3)."""
    start, advanced = 0, 0
    while advanced < 64:
        src = text[start:]
        ids = encode(tok, src)
        if len(ids) < n:
            raise ValueError(f"only {len(ids)} tokens after advancing {advanced} words; need {n}")
        ids = ids[:n]
        dec = decode(tok, ids)
        if "�" not in dec:
            if not src.startswith(dec):
                raise ValueError("decoded cut is not a prefix of the source")
            return {"ids": ids, "text": dec, "start_char": start, "words_advanced": advanced,
                    "source_tokens": len(encode(tok, text))}
        m = re.match(r"\s*\S+", src)
        start += m.end()
        advanced += 1
    raise ValueError("could not find a clean cut")


def _source_tail_after_cut(tok, tid: str) -> list[str]:
    """Paragraphs of a source that start after its 1,536-token gate cut."""
    text = read_source(tid)
    cut = cut_plain(tok, text)
    tail = text[cut["start_char"] + len(cut["text"]):]
    # Skip the paragraph the cut fell into.
    nxt = re.search(r"\n\s*\n", tail)
    return paragraphs(tail[nxt.end():] if nxt else "")


def render_ids(tok, messages, effort: str, add_generation_prompt: bool) -> tuple[str, list[int]]:
    from gate import vendor_template

    text = vendor_template.render(messages, add_generation_prompt=add_generation_prompt, reasoning_effort=effort)
    return text, encode(tok, text)


def build_chat(tok, tid: str) -> dict:
    """T7 (EN, effort high, closed think block) or T8 (DE, effort none): a user
    turn of >= CHAT_USER_MIN_TOKENS tokens, then an assistant turn whose
    content is cut at the token level so the whole render (no generation
    prompt) is exactly N_TOKENS tokens."""
    if tid == "T7":
        src, effort, instr, reasoning, lang = "T2", "high", T7_INSTRUCTION, T7_REASONING, "en"
    else:
        src, effort, instr, reasoning, lang = "T4", "none", T8_INSTRUCTION, None, "de"
    paras = _source_tail_after_cut(tok, src)
    k, user = 0, instr
    while len(encode(tok, user)) < CHAT_USER_MIN_TOKENS:
        user += ("" if user.endswith("\n\n") else "\n\n") + paras[k]
        k += 1
    content_full = "\n\n".join(paras[k:])
    c_ids = encode(tok, content_full)

    def messages_for(content):
        assistant = {"role": "assistant", "content": content}
        if reasoning is not None:
            assistant["reasoning"] = reasoning
        return [{"role": "user", "content": user}, assistant]

    full_text, full_ids = render_ids(tok, messages_for(content_full), effort, False)
    if len(full_ids) < N_TOKENS:
        raise ValueError(f"{tid}: only {len(full_ids)} tokens before the cut")
    m0 = len(c_ids) - (len(full_ids) - N_TOKENS)
    for delta in range(0, 16):
        for m in (m0 - delta, m0 + delta):
            content = decode(tok, c_ids[:m])
            if "�" in content:
                continue
            text, ids = render_ids(tok, messages_for(content), effort, False)
            if len(ids) == N_TOKENS:
                return {
                    "id": tid, "lang": lang, "effort": effort, "messages": messages_for(content),
                    "add_generation_prompt": False, "text": text, "ids": ids,
                    "rule": (f"user turn: instruction + paragraphs of {src}_source after its {N_TOKENS}-token "
                             f"cut until >= {CHAT_USER_MIN_TOKENS} tokens; assistant content: the following "
                             f"paragraphs, cut at the token level so the render is exactly {N_TOKENS} tokens"),
                }
    raise ValueError(f"{tid}: no content cut gives exactly {N_TOKENS} tokens")


def build_g5_prompts(tok) -> list[dict]:
    out = []
    for pid, src, lang, effort, frac in G5_SPECS:
        paras = paragraphs(read_source(src))
        k = int(frac * len(paras))
        user = G5_INSTRUCTION[lang]
        while True:
            if k >= len(paras):
                raise ValueError(f"{pid}: {src} ran out of paragraphs before {G5_MIN_TOKENS} tokens")
            user += ("" if user.endswith("\n\n") else "\n\n") + paras[k]
            k += 1
            messages = [{"role": "user", "content": user}]
            text, ids = render_ids(tok, messages, effort, True)
            if len(ids) >= G5_MIN_TOKENS:
                break
        out.append({"id": pid, "lang": lang, "effort": effort, "source": src, "start_fraction": frac,
                    "messages": messages, "text": text, "ids": ids, "n_tokens": len(ids),
                    "ids_sha256": ids_sha256(ids)})
    return out


# ---------------------------------------------------------------------------
# FineWeb-2 (T5, T6, T9): never committed
# ---------------------------------------------------------------------------


def iter_fineweb(parquet: Path, start: int = 0):
    """(row index, text) in file order from `start` (pyarrow imported lazily)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet))
    row = 0
    for batch in pf.iter_batches(columns=["text"], batch_size=512):
        for text in batch.column(0).to_pylist():
            if row >= start:
                yield row, text
            row += 1


def build_web(tok, parquet: Path) -> tuple[dict, dict]:
    """(public, work): `public` holds only rules, row indices, counts and
    sha256 (for MANIFEST.json); `work` holds the ids and texts for
    $EXP036_WORK/gate_texts/."""
    picked, public, work = [], {}, {}
    rows_iter = iter_fineweb(parquet, FINEWEB_START)
    for row, text in rows_iter:
        n = len(encode(tok, text))
        if n >= FINEWEB_MIN_TOKENS:
            picked.append((row, text, n))
            if len(picked) == 2:
                break
    for tid, (row, text, n) in zip(("T5", "T6"), picked):
        cut = cut_plain(tok, text)
        work[tid] = {"id": tid, "row": row, "ids": cut["ids"], "text": cut["text"]}
        public[tid] = {
            "id": tid, "lang": "de", "kind": "web", "row": row, "doc_tokens": n,
            "doc_text_sha256": sha256_bytes(text.encode("utf-8")),
            "n_tokens": N_TOKENS, "ids_sha256": ids_sha256(cut["ids"]),
            "text_sha256": sha256_bytes(cut["text"].encode("utf-8")),
            "words_advanced": cut["words_advanced"],
            "rule": (f"the {'first' if tid == 'T5' else 'second'} document at file index >= {FINEWEB_START} "
                     f"with >= {FINEWEB_MIN_TOKENS} Kolibri tokens; its first {N_TOKENS} tokens"),
        }
    sep = encode(tok, SEPARATOR)
    ids, components = [], []

    def add(what, piece, extra):
        nonlocal ids
        if ids:
            ids += sep
        ids += piece
        components.append({"what": what, "n_tokens": len(piece), **extra})

    for tid in ("T1", "T4", "T3"):
        src = read_source(tid)
        add(f"{tid} source in full", encode(tok, src), {"source_sha256": sha256_bytes(src.encode("utf-8"))})
    for row, text, n in picked:
        add("FineWeb-2 document", encode(tok, text), {"row": row, "doc_text_sha256": sha256_bytes(text.encode("utf-8"))})
    last = picked[-1][0]
    for row, text in iter_fineweb(parquet, last + 1):
        if len(ids) >= T9_TOKENS:
            break
        add("FineWeb-2 document", encode(tok, text), {"row": row, "doc_text_sha256": sha256_bytes(text.encode("utf-8"))})
    ids = ids[:T9_TOKENS]
    work["T9"] = {"id": "T9", "ids": ids}
    public["T9"] = {
        "id": "T9", "lang": "en+de", "kind": "long", "n_tokens": len(ids), "ids_sha256": ids_sha256(ids),
        "components": components, "separator": SEPARATOR,
        "rule": (f"T1, T4 and T3 sources in full, then T5's and T6's documents in full, then the following "
                 f"deu_Latn test documents in file order; each tokenised alone, joined with the ids of "
                 f"{SEPARATOR!r}, cut at {T9_TOKENS} tokens"),
    }
    return public, work


def write_work(work: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, entry in work.items():
        common.write_json(out_dir / f"{tid}.ids.json", {"id": tid, "ids": entry["ids"], "ids_sha256": ids_sha256(entry["ids"]),
                                                         "row": entry.get("row")})
        if "text" in entry:
            (out_dir / f"{tid}.txt").write_text(entry["text"], encoding="utf-8")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def tokenizer_info(tok_dir: Path) -> dict:
    return {
        "source": "Aleph-Alpha/Kolibri-1-BF16 @ 7a8f290e7858825c3cf5e4c447ba68345de9f1d3",
        "tokenizer_json_sha256": sha256_file(Path(tok_dir) / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(Path(tok_dir) / "tokenizer_config.json"),
        "encode": "tokenizers.Tokenizer.encode(text, add_special_tokens=False)",
    }


def build(tok_dir: Path, fineweb: Path | None = None, previous: dict | None = None) -> tuple[dict, dict, dict]:
    """(manifest, files, work): files maps a gate/texts file name to its
    content (str); nothing is written here."""
    tok = common.load_raw_tokenizer(tok_dir)
    origins = common.read_json(SRC_DIR / "SOURCES.json")
    files: dict[str, str] = {}
    texts: dict[str, dict] = {}
    for tid, (stem, lang) in PLAIN.items():
        src = read_source(tid)
        cut = cut_plain(tok, src)
        ids_sha = ids_sha256(cut["ids"])
        text_sha = sha256_bytes(cut["text"].encode("utf-8"))
        files[f"{stem}.ids.json"] = common.dumps({"id": tid, "ids": cut["ids"], "ids_sha256": ids_sha,
                                                  "n_tokens": len(cut["ids"]), "text_sha256": text_sha})
        files[f"{stem}.txt"] = cut["text"]
        texts[tid] = {
            "id": tid, "lang": lang, "kind": "plain", "files": [f"{stem}.ids.json", f"{stem}.txt"],
            "n_tokens": len(cut["ids"]), "ids_sha256": ids_sha, "text_sha256": text_sha,
            "source": f"src/{tid}_source.txt", "source_sha256": sha256_file(SRC_DIR / f"{tid}_source.txt"),
            "source_tokens": cut["source_tokens"], "words_advanced": cut["words_advanced"],
            "origin": origins.get(tid, {}),
            "rule": f"first {N_TOKENS} tokens of the source (advance one word on a U+FFFD cut)",
        }
    for tid, name in (("T7", "T7_chat_en_high.json"), ("T8", "T8_chat_de_none.json")):
        chat = build_chat(tok, tid)
        entry = {k: chat[k] for k in ("id", "lang", "effort", "messages", "add_generation_prompt", "ids", "rule")}
        entry.update(n_tokens=len(chat["ids"]), ids_sha256=ids_sha256(chat["ids"]),
                     text=chat["text"], text_sha256=sha256_bytes(chat["text"].encode("utf-8")))
        files[name] = common.dumps(entry)
        texts[tid] = {"id": tid, "lang": chat["lang"], "kind": "chat", "files": [name],
                      "effort": chat["effort"], "n_tokens": len(chat["ids"]),
                      "ids_sha256": entry["ids_sha256"], "text_sha256": entry["text_sha256"], "rule": chat["rule"]}
    g5 = build_g5_prompts(tok)
    files["G5_prompts.json"] = common.dumps({"prompts": g5, "min_tokens": G5_MIN_TOKENS,
                                             "rule": "vendor template, add_generation_prompt=True; paragraphs from the start fraction until >= 600 tokens"})
    manifest = {
        "version": "exp036-gate-texts-1",
        "n_tokens_per_text": N_TOKENS,
        "texts": texts,
        "g5_prompts": {"file": "G5_prompts.json", "ids_sha256": [p["ids_sha256"] for p in g5],
                       "n_tokens": [p["n_tokens"] for p in g5]},
        "tokenizer": tokenizer_info(tok_dir),
        "vendor_template_sha256": sha256_file(common.VENDOR_JINJA),
        "ids_sha256_rule": common.IDS_SHA256_RULE,
        "web": {
            "parquet": f"$EXP036_DATA/{FINEWEB_REL}",
            "parquet_sha256": FINEWEB_SHA256,
            "revision": FINEWEB_REVISION,
            "licence": "ODC-By 1.0; subject to Common Crawl terms. No web text is committed.",
            "work_dir": "$EXP036_WORK/gate_texts",
        },
    }
    work = {}
    if fineweb is not None:
        if sha256_file(fineweb) != FINEWEB_SHA256:
            raise SystemExit(f"{fineweb}: sha256 differs from the pinned {FINEWEB_SHA256}")
        public, work = build_web(tok, fineweb)
        manifest["web"].update(public)
    elif previous is not None:
        for tid in ("T5", "T6", "T9"):
            if tid in previous.get("web", {}):
                manifest["web"][tid] = previous["web"][tid]
    files["MANIFEST.json"] = common.dumps(manifest)
    return manifest, files, work


def compare_web(manifest: dict, public: dict) -> list[str]:
    """Differences between MANIFEST's T5/T6/T9 entries and a fresh build."""
    diffs = []
    for tid in ("T5", "T6", "T9"):
        want = manifest.get("web", {}).get(tid)
        if want is None:
            continue
        for key in ("row", "doc_tokens", "doc_text_sha256", "ids_sha256", "n_tokens", "text_sha256"):
            if key in want and want.get(key) != public[tid].get(key):
                diffs.append(f"{tid}.{key}: manifest {want.get(key)} != built {public[tid].get(key)}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tok", type=Path, default=None, help="Kolibri tokenizer dir (default: $EXP036_TOK or the BF16 snapshot)")
    ap.add_argument("--fineweb", type=Path, default=None, help="the pinned FineWeb-2 deu_Latn test parquet")
    ap.add_argument("--blog", type=Path, default=None, help="mini only: the blog's content/posts dir (rewrites src/T1, src/T2)")
    ap.add_argument("--gg-zip", type=Path, default=None, help="mini only: gesetze-im-internet.de gg/xml.zip (rewrites src/T4)")
    ap.add_argument("--work-only", action="store_true", help="write T5, T6, T9 to $EXP036_WORK/gate_texts only")
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with gate/texts")
    args = ap.parse_args(argv)
    tok_dir = args.tok or common.tokenizer_dir()
    previous = common.read_json(MANIFEST) if MANIFEST.is_file() else None

    if args.work_only:
        if args.fineweb is None:
            ap.error("--work-only needs --fineweb")
        tok = common.load_raw_tokenizer(tok_dir)
        if sha256_file(args.fineweb) != FINEWEB_SHA256:
            print(f"{args.fineweb}: sha256 differs from the pinned parquet", file=sys.stderr)
            return 1
        public, work = build_web(tok, args.fineweb)
        out_dir = common.work_dir() / "gate_texts"
        write_work(work, out_dir)
        have = previous is not None and all(t in previous.get("web", {}) for t in ("T5", "T6", "T9"))
        if have:
            diffs = compare_web(previous, public)
            if diffs:
                print("T5/T6/T9 differ from MANIFEST.json:\n  " + "\n  ".join(diffs), file=sys.stderr)
                return 1
            print(f"T5, T6, T9 written to {common.redact_path(out_dir)}; they match MANIFEST.json")
        else:
            common.require_identity()
            path = common.DEFAULT_RESULTS_DIR / f"gate_texts_{common.utc_stamp()}.json"
            common.write_new_json(path, {"web": public, "parquet_sha256": FINEWEB_SHA256, "utc": common.utc_iso()})
            print(f"T5, T6, T9 written to {common.redact_path(out_dir)}; indices recorded in {common.redact_path(path)}")
        return 0

    if args.blog is not None or args.gg_zip is not None:
        write_sources(args.blog, args.gg_zip)
    manifest, files, work = build(tok_dir, args.fineweb, previous)
    if args.check:
        bad = []
        for name, content in sorted(files.items()):
            path = TEXTS_DIR / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                bad.append(name)
        if bad:
            print("differs from gate/texts: " + ", ".join(bad), file=sys.stderr)
            return 1
        print(f"gate/texts reproduce ({len(files)} files)")
        return 0
    for name, content in files.items():
        (TEXTS_DIR / name).write_text(content, encoding="utf-8")
    if work:
        write_work(work, common.work_dir() / "gate_texts")
    print(f"wrote {len(files)} files to gate/texts; MANIFEST sha256 {sha256_file(MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
