#!/bin/bash
# {stock,optimized} x workers x concurrency REST search sweep with a delayed
# mock embedder. Workload: project $PROJECT (seed it first -- see README),
# types=episodic, top_k=10 (the core derives vector_search_limit=50).
#
# Env:
#   PY         python of a venv with memmachine-server installed
#              (default: <repo>/.venv/bin/python)
#   HOT_SRC    PYTHONPATH overlay for the optimized build
#              (default: <repo>/packages/server/src, i.e. this branch)
#   STOCK_SRC  PYTHONPATH overlay for the baseline build; '' uses the venv's
#              installed package -- point it at a checkout of the branch's
#              merge-base for a true stock baseline (see README)
#   DELAY_MS   mock embedder latency (default 200)
#   PROJECT    seeded project to search (default isolated1)
#
# Every phase boots a FRESH server; a failed bind leaves the previous server
# serving, so each phase asserts on "address already in use" and every boot
# asserts the searched project actually returns hits.
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
DELAY_MS=$DELAY_MS nohup "$PY" "$S/mock_openai.py" > "$R/mock_d$DELAY_MS.log" 2>&1 &

( while true; do
    echo "== $(date +%H:%M:%S)"
    docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' bench2-qdrant bench2-pg 2>/dev/null
    ps -A -o %cpu= -o command= | grep -E 'memmachine-server|multi_client|rest_bench|mock_openai' \
      | grep -v grep | sort -rn | head -14
    sleep 8
  done ) > "$R/sysload.log" 2>&1 &
SAMPLER=$!

phase() {  # $1 pythonpath ('' = venv's installed package)  $2 tag  $3 workers  $4 "c:queries:procs ..."
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
    IFS=: read -r c q p <<< "$arm"
    "$PY" "$S/multi_client.py" --project "$PROJECT" --c "$c" --queries "$q" --procs "$p" 2>&1 \
      | sed "s/^/[$tag] /"
  done
  kill "$spid" 2>/dev/null
  for i in $(seq 1 20); do pgrep -f memmachine-server >/dev/null || break; sleep 1; done
  pkill -f memmachine-server 2>/dev/null; sleep 1
}

# Queries per arm target a 15-25s steady window at the rate that build/W/c is
# expected to sustain; adjust freely, the harness measures whatever runs.
for rep in 1 2; do
  phase "$STOCK_SRC" "S${rep}w1" 1 "16:1500:1 64:3000:2 128:3000:2 256:3000:4"
  phase "$HOT_SRC"   "H${rep}w1" 1 "16:1500:1 64:6000:2 128:9000:2 256:9000:4"
  phase "$STOCK_SRC" "S${rep}w2" 2 "64:5500:2 128:5500:2 256:5500:4 384:5500:6"
  phase "$HOT_SRC"   "H${rep}w2" 2 "64:6000:2 128:11000:2 256:18000:4 384:18000:6"
  phase "$STOCK_SRC" "S${rep}w4" 4 "64:6000:2 128:9000:2 256:11000:4 384:11000:6"
  phase "$HOT_SRC"   "H${rep}w4" 4 "64:6000:2 128:11000:2 256:18000:4 384:20000:6"
done

kill "$SAMPLER" 2>/dev/null
pkill -f mock_openai.py 2>/dev/null
echo "SWEEP DONE"
