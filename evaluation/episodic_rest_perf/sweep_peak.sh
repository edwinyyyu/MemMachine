#!/bin/bash
# Ceiling probe: ramped high-concurrency arms around the optimized build's
# peak, after sweep_delay.sh showed arm-start SYN bursts past
# kern.ipc.somaxconn=128 stall connections and drag window averages.
# Arms are "c:queries:procs:ramp_s"; ramp staggers each child's worker
# starts so connections open gradually. Env knobs as in sweep_delay.sh.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
PY=${PY:-$REPO/.venv/bin/python}
SERVER=${SERVER:-$(dirname "$PY")/memmachine-server}
HOT_SRC=${HOT_SRC:-$REPO/packages/server/src}
STOCK_SRC=${STOCK_SRC:-}
DELAY_MS=${DELAY_MS:-200}
PROJECT=${PROJECT:-isolated1}
R=$S/results
mkdir -p "$R"
cd "$S"

pkill -f memmachine-server 2>/dev/null; pkill -f mock_openai.py 2>/dev/null; sleep 1
DELAY_MS=$DELAY_MS nohup "$PY" "$S/mock_openai.py" > "$R/mock_peak.log" 2>&1 &

( while true; do
    echo "== $(date +%H:%M:%S)"
    docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' bench2-qdrant bench2-pg 2>/dev/null
    ps -A -r -o %cpu= -o comm= | head -10
    sleep 8
  done ) > "$R/sysload_peak.log" 2>&1 &
SAMPLER=$!

phase() {  # $1 pythonpath ('' = venv's installed package)  $2 tag  $3 workers  $4 "c:q:procs:ramp ..."
  local pp=$1 tag=$2 w=$3 arms=$4 spid
  if [ -n "$pp" ]; then
    MEMORY_CONFIG=$S/config.yml MEMMACHINE_WORKERS=$w PYTHONPATH=$pp \
      nohup "$SERVER" > "$R/server_$tag.log" 2>&1 &
  else
    MEMORY_CONFIG=$S/config.yml MEMMACHINE_WORKERS=$w \
      nohup "$SERVER" > "$R/server_$tag.log" 2>&1 &
  fi
  spid=$!
  for i in $(seq 1 90); do
    curl -s -o /dev/null http://127.0.0.1:8091/docs && break; sleep 1
  done
  grep -q "address already in use" "$R/server_$tag.log" && echo "[$tag] BIND FAILED -- ABORT PHASE"
  curl -s http://127.0.0.1:8091/api/v2/memories/search -H 'Content-Type: application/json' \
    -d "{\"org_id\":\"benchorg\",\"project_id\":\"$PROJECT\",\"top_k\":10,\"query\":\"warm\",\"types\":[\"episodic\"]}" \
    | "$PY" -c "
import json, sys
n = len(json.load(sys.stdin)['content']['episodic_memory']['long_term_memory']['episodes'])
assert n == 10, n
print('[$tag] hits ok (10 episodes)')"
  for arm in $arms; do
    IFS=: read -r c q p ramp <<< "$arm"
    "$PY" "$S/multi_client.py" --project "$PROJECT" --c "$c" --queries "$q" \
      --procs "$p" --ramp "$ramp" 2>&1 | sed "s/^/[$tag] /"
  done
  kill "$spid" 2>/dev/null
  for i in $(seq 1 20); do pgrep -f memmachine-server >/dev/null || break; sleep 1; done
  pkill -f memmachine-server 2>/dev/null; sleep 1
}

phase "$HOT_SRC"   "Hw1" 1 "256:27000:4:2"
phase "$HOT_SRC"   "Hw2" 2 "256:36000:4:2 320:38000:5:2 448:40000:7:3"
phase "$HOT_SRC"   "Hw4" 4 "256:36000:4:2 320:38000:5:2 448:40000:7:3 512:40000:8:3"
phase "$STOCK_SRC" "Sw4" 4 "128:9000:2:2"

kill "$SAMPLER" 2>/dev/null
pkill -f mock_openai.py 2>/dev/null
echo "PEAK SWEEP DONE"
