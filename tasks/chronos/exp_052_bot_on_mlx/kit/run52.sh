#!/bin/zsh -e
# exp_052: the registered run, in order. From tasks/exp052, inside tmux (session exp052-run).
# The router (gemma4:e4b on Ollama) stays loaded throughout, as it would if only the answering model moved.
cd "$(dirname "$0")/.."
TS=$(date -u +%Y%m%dT%H%M%SZ)
R=runs/$TS
mkdir -p $R
export HF_HUB_OFFLINE=1
A=(python3.14 -I kit/arms52.py --frozen data/frozen.jsonl)
{
  echo "exp_052 run $TS  ollama $(curl -s localhost:11434/api/version)  loaded before: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
  curl -s localhost:11434/api/generate -d '{"model":"gemma4:26b","keep_alive":-1}' >/dev/null   # G loaded as live
  $A --arm G --out $R/G || { echo "FAILED: G"; exit 1 }
  ollama stop gemma4:26b
  python3.14 -m mlx_lm server --model mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit --port 8090 --max-tokens 6144 \
      --log-level WARNING > $R/mlx_server.log 2>&1 &
  MLXPID=$!
  for i in {1..90}; do curl -s -o /dev/null 127.0.0.1:8090/v1/models && break; sleep 2; done
  curl -s 127.0.0.1:8090/v1/chat/completions -d '{"model":"mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit","messages":[{"role":"user","content":"Warm-up. One word."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
  $A --arm M --out $R/M || { echo "FAILED: M"; kill $MLXPID; exit 1 }
  kill $MLXPID; sleep 3
  curl -s localhost:11434/api/generate -d '{"model":"gemma4:26b","keep_alive":-1}' >/dev/null
  $A --arm G --out $R/G_drift --first 10 || { echo "FAILED: drift"; exit 1 }
  python3.14 -I kit/score52.py data/frozen.jsonl $R
  python3.14 -I kit/read52.py data/frozen.jsonl $R
  echo "exp_052 run $TS finished $(date -u +%H:%M:%SZ); loaded after: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
} 2>&1 | tee $R/run.log
