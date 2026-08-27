#!/bin/zsh
# Full benchmark matrix, sequential (both DBs share the docker VM).
# Knobs recorded in labels: n=count c=concurrency b=batch thr=neo4j index
# threshold k/vsl=search knobs e=expand q=query count d=768 dims throughout.
set -e
PY="${PY:-$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python}"
cd "$(dirname "$0")"

stage() { echo "\n##### $(date +%H:%M:%S) $1 #####" }

### Phase 1: typical-session scale, shipped defaults (thr=10000)
stage "decl ingest 12k c1 (crosses 10k index cliff)"
$PY bench_declarative.py --mode ingest --label ingest_decl_n12000_c1_b1_thr10000 \
  --session decl_12k --n 12000 --concurrency 1 --threshold 10000 2>/dev/null
stage "event ingest 12k c1"
$PY bench_event.py --mode ingest --label ingest_ev_n12000_c1_b1 \
  --session ev_12k --n 12000 --concurrency 1 2>/dev/null

stage "decl ingest 2k c1 (stays sub-threshold: exact-scan regime)"
$PY bench_declarative.py --mode ingest --label ingest_decl_n2000_c1_b1_thr10000 \
  --session decl_2k --n 2000 --concurrency 1 --threshold 10000 2>/dev/null
stage "event ingest 2k c1"
$PY bench_event.py --mode ingest --label ingest_ev_n2000_c1_b1 \
  --session ev_2k --n 2000 --concurrency 1 2>/dev/null

stage "decl query 2k (exact-scan regime) c1 / c16"
$PY bench_declarative.py --mode query --label query_decl_2k_q200_c1_k10_e0_thr10000 \
  --session decl_2k --queries 200 --warmup 10 --k 10 --expand 0 --threshold 10000 2>/dev/null
$PY bench_declarative.py --mode query --label query_decl_2k_q200_c16_k10_e0_thr10000 \
  --session decl_2k --queries 200 --warmup 10 --k 10 --expand 0 --concurrency 16 --threshold 10000 2>/dev/null
stage "event query 2k c1 / c16"
$PY bench_event.py --mode query --label query_ev_2k_q200_c1_vsl50_e0 \
  --session ev_2k --queries 200 --warmup 10 --vsl 50 --expand 0 2>/dev/null
$PY bench_event.py --mode query --label query_ev_2k_q200_c16_vsl50_e0 \
  --session ev_2k --queries 200 --warmup 10 --vsl 50 --expand 0 --concurrency 16 2>/dev/null

### Phase 2: concurrent multi-session ingest (8 sessions, 2k each)
stage "decl ingest 16k c8 one-session-per-worker"
$PY bench_declarative.py --mode ingest --label ingest_decl_n16000_c8_spw_b1_thr10000 \
  --session decl_ms --n 16000 --concurrency 8 --sessions-per-worker --threshold 10000 2>/dev/null
stage "event ingest 16k c8 one-session-per-worker"
$PY bench_event.py --mode ingest --label ingest_ev_n16000_c8_spw_b1 \
  --session ev_ms --n 16000 --concurrency 8 --sessions-per-worker 2>/dev/null

### Phase 3: large session 50k, neo4j best case (thr=1: indexes from start)
stage "decl ingest 50k c4 thr1 (best case: indexed from start)"
$PY bench_declarative.py --mode ingest --label ingest_decl_n50000_c4_b1_thr1 \
  --session decl_50k --n 50000 --concurrency 4 --threshold 1 2>/dev/null
stage "event ingest 50k c4"
$PY bench_event.py --mode ingest --label ingest_ev_n50000_c4_b1 \
  --session ev_50k --n 50000 --concurrency 4 2>/dev/null

stage "decl queries 50k (ANN regime)"
for c in 1 16 32; do
  $PY bench_declarative.py --mode query --label query_decl_50k_q300_c${c}_k10_e0_thr1 \
    --session decl_50k --queries 300 --warmup 20 --k 10 --expand 0 \
    --concurrency $c --threshold 1 --wait-indexes 2>/dev/null
done
$PY bench_declarative.py --mode query --label query_decl_50k_q300_c1_k10_e6_thr1 \
  --session decl_50k --queries 300 --warmup 20 --k 10 --expand 6 --threshold 1 --wait-indexes 2>/dev/null
$PY bench_declarative.py --mode query --label query_decl_50k_q300_c16_k10_e6_thr1 \
  --session decl_50k --queries 300 --warmup 20 --k 10 --expand 6 --concurrency 16 --threshold 1 --wait-indexes 2>/dev/null
$PY bench_declarative.py --mode query --label query_decl_50k_q300_c1_k10_e0_filtered_thr1 \
  --session decl_50k --queries 300 --warmup 20 --k 10 --expand 0 --filtered --threshold 1 --wait-indexes 2>/dev/null
$PY bench_declarative.py --mode query --label query_decl_50k_q300_c16_k10_e0_filtered_thr1 \
  --session decl_50k --queries 300 --warmup 20 --k 10 --expand 0 --filtered --concurrency 16 --threshold 1 --wait-indexes 2>/dev/null

stage "event queries 50k"
for c in 1 16 32; do
  $PY bench_event.py --mode query --label query_ev_50k_q300_c${c}_vsl50_e0 \
    --session ev_50k --queries 300 --warmup 20 --vsl 50 --expand 0 --concurrency $c 2>/dev/null
done
$PY bench_event.py --mode query --label query_ev_50k_q300_c1_vsl50_e6 \
  --session ev_50k --queries 300 --warmup 20 --vsl 50 --expand 6 2>/dev/null
$PY bench_event.py --mode query --label query_ev_50k_q300_c16_vsl50_e6 \
  --session ev_50k --queries 300 --warmup 20 --vsl 50 --expand 6 --concurrency 16 2>/dev/null
$PY bench_event.py --mode query --label query_ev_50k_q300_c1_vsl50_e0_filtered \
  --session ev_50k --queries 300 --warmup 20 --vsl 50 --expand 0 --filtered 2>/dev/null
$PY bench_event.py --mode query --label query_ev_50k_q300_c16_vsl50_e0_filtered \
  --session ev_50k --queries 300 --warmup 20 --vsl 50 --expand 0 --filtered --concurrency 16 2>/dev/null

### Phase 4: mixed read/write at 50k
stage "decl mixed 50k r16 w4 60s"
$PY bench_declarative.py --mode mixed --label mixed_decl_50k_r16_w4_60s_k10_e0_thr1 \
  --session decl_50k --readers 16 --writers 4 --duration 60 --k 10 --threshold 1 2>/dev/null
stage "event mixed 50k r16 w4 60s"
$PY bench_event.py --mode mixed --label mixed_ev_50k_r16_w4_60s_vsl50_e0 \
  --session ev_50k --readers 16 --writers 4 --duration 60 --vsl 50 2>/dev/null

stage "ALL DONE"
