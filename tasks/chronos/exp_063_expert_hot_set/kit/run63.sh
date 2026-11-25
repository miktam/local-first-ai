#!/bin/zsh
# usage: run63.sh <gemma|inkling> <text>...   one imatrix file per text; counts extracted to runs/<stamp>/
set -e
M=$1; shift; H=$HOME; W=$H/models/exp063; HERE=${0:A:h}; OUT=${OUT:?set OUT to the run dir}
case $M in gemma) F=$H/.ollama/models/blobs/sha256-4c856523d61d*;; inkling) F=$H/models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf;; esac
F=$(echo ${~F}); mkdir -p $OUT $W/im
for t in $@; do
  $W/build/bin/llama-imatrix -m $F -f $W/texts/$t.txt -o $W/im/$M-$t.gguf --no-ppl -c 512 --chunks 12 -ngl 99 > $OUT/$M-$t.log 2>&1
  $H/models/exp036-mini/venv314/bin/python -I $HERE/extract.py $W/im/$M-$t.gguf $OUT/$M-$t.json
  echo "$M $t done"
done
