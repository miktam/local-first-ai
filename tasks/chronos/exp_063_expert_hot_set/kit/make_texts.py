"""Build the eight texts (four classes, halves a = rank, b = score). Writes the texts outside the repo and a manifest
(file names, sha256, bytes) inside it. a and b always come from different source files."""
import glob, hashlib, json, os, re, sys
H = os.path.expanduser("~"); OUT = f"{H}/models/exp063/texts"; N = 60000
L = f"{H}/REPOS/library/sources"; C = f"{H}/models/llama.cpp-b12676"
def strip(s): return re.sub(r"\A---\n.*?\n---\n", "", s, flags=re.S)
es = sorted(glob.glob(f"{H}/REPOS/esgrimasanpedro/content/**/*.es.md", recursive=True))
ru = sorted(glob.glob(f"{L}/general-psychology-mgu/transcripts/*.md"))
SRC = {
 "en_book": ([f"{L}/james/principles-of-psychology/04-habit.md"], [f"{L}/james/principles-of-psychology/22-reasoning.md"]),
 "ru_lecture": (ru[:8], ru[-8:]),
 "es_site": (es[0::2], es[1::2]),
 "cpp_code": ([f"{C}/src/llama-model.cpp"], [f"{C}/ggml/src/ggml.c"]),
}
man = {}
for k, halves in SRC.items():
    for half, files in zip("ab", halves):
        s = "\n\n".join(strip(open(f, encoding="utf8").read()) for f in files)[:N]
        p = f"{OUT}/{k}_{half}.txt"; open(p, "w", encoding="utf8").write(s)
        man[f"{k}_{half}"] = {"chars": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest(),
                              "files": [os.path.relpath(f, H) for f in files]}
json.dump(man, open(sys.argv[1], "w"), indent=1); print({k: v["chars"] for k, v in man.items()})
