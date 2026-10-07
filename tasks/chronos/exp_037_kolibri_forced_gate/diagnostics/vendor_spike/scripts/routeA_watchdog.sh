#!/bin/bash
V="${SPIKE_DIR:?set SPIKE_DIR to the scratch folder of the spike}"  # exp_037 copy: was the scratch path on the mini
DEADLINE=$(date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-06T04:37:20Z" +%s)
$V/scripts/routeA_install.sh > $V/routeA/install.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if [ $(date -u +%s) -ge $DEADLINE ]; then echo "DEADLINE_HIT $(date -u)" >> $V/routeA/install.log; pkill -P $PID; kill $PID; sleep 2; pkill -f "vllm-0.29.0" ; exit 124; fi
  sleep 5
done
wait $PID; RC=$?; echo "EXIT $RC $(date -u)" >> $V/routeA/install.log; exit $RC
