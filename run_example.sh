#!/usr/bin/env bash
#
# Q-Gate end-to-end example: gating + fusion (Stage 2) followed by keyframe
# selection + downstream QA (Stage 3).
#
# This starts from a dataset JSON that already carries all three raw score
# streams (score_logic / score_relevance / score_narrative). Producing those
# streams (Stage 1) needs the heavy expert models and, for score_logic, the
# external YOLO-World repo — see the README "Score streams" section.
#
# Prerequisites:
#   - pip install -r requirements.txt
#   - cp .env.example .env  and set OPENAI_API_KEY (+ OPENAI_BASE_URL if using a proxy)
#   - videos/subtitles downloaded under ./Datasets (see the README "Datasets layout")
#
# Usage (defaults shown; override any via environment variables):
#   bash run_example.sh
#   DATASET=videomme BACKEND=qwen3_32b K=32 bash run_example.sh
#   RAW_SCORES=./Score/narrative/lvb_..._narrative.json bash run_example.sh
set -euo pipefail

# --- Load environment (OPENAI_API_KEY etc.) if a .env file is present ---
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

# --- Configurable knobs (all env-overridable) ---
DATASET="${DATASET:-longvideobench}"   # longvideobench | videomme
BACKEND="${BACKEND:-gpt4}"             # gpt4 | qwen3_32b | internvl3_8b | internvl_26b
K="${K:-8}"                            # number of keyframes
TAU="${TAU:-0.5}"                      # fusion softmax temperature
SCORE_KEY="${SCORE_KEY:-score_fusion}" # fused Q-Gate distribution

# GPT-4o needs an API key; local backends need their *_BASE_URL server instead.
if [ "${BACKEND}" = "gpt4" ]; then
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY (copy .env.example to .env and fill it in)}"
fi

# fuse_scores.py names its output using Python's float form of the temperature
# (0.5 -> "0.5", 1 -> "1.0"); normalize TAU the same way so the Stage-3 lookup matches.
TAU="$(python -c "print(float('${TAU}'))")"

# Input carrying all three raw streams for this dataset (the narrative producer's
# output already contains them). Override RAW_SCORES to point at your file.
RAW_SCORES="${RAW_SCORES:-./Score/${DATASET}_raw_scores.json}"
FUSION_DIR="./Score/fusion"
RESULTS_DIR="./results"

# Preflight: the raw-scores file is the output of Stage 1 (the narrative producer),
# not something this script creates. Fail early with guidance instead of a bare
# FileNotFoundError deep inside fuse_scores.py.
if [ ! -f "${RAW_SCORES}" ]; then
    echo "ERROR: raw-scores file not found: ${RAW_SCORES}" >&2
    echo "This file is the Stage-1 output that carries all three streams" >&2
    echo "(score_logic / score_relevance / score_narrative). Produce the streams" >&2
    echo "first (see README 'Score streams'), then point RAW_SCORES at the narrative" >&2
    echo "producer's output, e.g.:" >&2
    echo "  RAW_SCORES=./Score/narrative/${DATASET}_..._narrative.json bash run_example.sh" >&2
    exit 1
fi

echo "==> Stage 2: query-modulated gating + score fusion (backend=${BACKEND}, tau=${TAU})"
python fuse_scores.py \
    --input_json "${RAW_SCORES}" \
    --output_dir "${FUSION_DIR}" \
    --strategy llm --backend "${BACKEND}" --temperature "${TAU}"

# fuse_scores.py writes "<input_basename>_fusion_llm_T<tau>.json" into FUSION_DIR.
RAW_BASENAME="$(basename "${RAW_SCORES}" .json)"
FUSED="${FUSION_DIR}/${RAW_BASENAME}_fusion_llm_T${TAU}.json"

echo "==> Stage 3: top-K keyframe selection + downstream QA (dataset=${DATASET}, K=${K})"
python run_qa.py \
    --input_json "${FUSED}" \
    --dataset "${DATASET}" --output_dir "${RESULTS_DIR}" \
    --backend "${BACKEND}" --frame_num "${K}" \
    --score_key "${SCORE_KEY}" \
    --use_subtitles --filter_with_subtitles

echo "==> Done. QA results and the accuracy summary (.txt) are in ${RESULTS_DIR}/"
