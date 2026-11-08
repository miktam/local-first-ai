date
ollama pull hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M
ollama pull hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M
hf download NousResearch/Propaganda_Eval --repo-type dataset --local-dir ~/models/propaganda-eval
ollama list | grep -i -E "qwen3.8-27b|arete"
shasum -a 256 ~/models/propaganda-eval/data/items.jsonl ~/models/propaganda-eval/judge/JUDGE_PROMPT.md ~/models/propaganda-eval/judge/score.py
wc -l ~/models/propaganda-eval/data/items.jsonl
ollama ps
date
