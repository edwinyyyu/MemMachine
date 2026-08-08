#!/bin/bash
# segment_store baseline with methodology matched to the temporal_scoring runs:
# TextSegmenter(max_chunk_length=4000) + WholeTextDeriver, text-embedding-3-small,
# vector_search_limit=50, max_num_segments=6, expand_context=0, answerer gpt-5-mini
# (--answer-prompt simple), judge gpt-5-mini mem0-classic, skip cat5.
# Two retrieval arms from ONE ingest: cosine-only (control) and BM25-additive (the
# segment_store signature). cosine arm should ~match temporal_scoring's cosine OFF.
set -e
cd /Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/locomo
P=/Users/eyu/edwinyyyu/mmcc/segment_store
DATA=../../data/locomo10.json
SD=baseline_tw_seg.db; VD=baseline_tw_vec.db
EMB=text-embedding-3-small

echo "=== INGEST (text@4000 + whole) === $(date)"
uv run --project $P python locomo_ingest.py --data-path $DATA \
  --segment-db $SD --vector-db $VD --segmenter text --deriver whole \
  --max-text-chunk-length 4000 --embedding-model $EMB > baseline-ingest.log 2>&1
echo "ingest exit=$? $(date)"

for ARM in cos bm25; do
  if [ "$ARM" = "cos" ]; then FUSE="--bm25-fusion none"; else FUSE="--bm25-fusion additive --bm25-fusion-weight 0.5"; fi
  echo "=== SEARCH $ARM ($FUSE) === $(date)"
  uv run --project $P python locomo_search.py --data-path $DATA \
    --target-path baseline-$ARM-search.json --segment-db $SD --vector-db $VD \
    --embedding-model $EMB --vector-search-limit 50 --max-num-segments 6 \
    --expand-context 0 --model gpt-5-mini --answer-prompt simple $FUSE \
    > baseline-$ARM-search.log 2>&1
  echo "search $ARM exit=$? $(date)"
done

echo "=== JUDGE both (gpt-5-mini mem0-classic, skip cat5) === $(date)"
uv run --project $P python locomo_evaluate.py --data-path baseline-cos-search.json \
  --target-path baseline-cos-eval.json --judge-model gpt-5-mini \
  --judge-variant mem0-classic --skip-category-5 > baseline-cos-eval.log 2>&1 &
J1=$!
uv run --project $P python locomo_evaluate.py --data-path baseline-bm25-search.json \
  --target-path baseline-bm25-eval.json --judge-model gpt-5-mini \
  --judge-variant mem0-classic --skip-category-5 > baseline-bm25-eval.log 2>&1 &
J2=$!
wait $J1; echo "judge cos exit=$?"; wait $J2; echo "judge bm25 exit=$?"
echo "BASELINE COMPLETE $(date)"
