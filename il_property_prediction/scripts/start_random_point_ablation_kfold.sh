#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/venvs/mipgraph/bin/python}"
RUN_ID="${RUN_ID:-ablation_random_point_kfold_seed42}"
LOG_DIR="${LOG_DIR:-tmp/training_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablation_random_point_kfold_seed42}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALIDATE_EVERY="${VALIDATE_EVERY:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VARIANTS="${VARIANTS:-all}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
OUT_LOG="${LOG_DIR}/${RUN_ID}.out.log"
ERR_LOG="${LOG_DIR}/${RUN_ID}.err.log"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "already running: pid=${old_pid}"
    echo "stdout=${OUT_LOG}"
    echo "stderr=${ERR_LOG}"
    exit 0
  fi
fi

if [[ -n "${WAIT_FOR_PID}" ]]; then
  nohup bash -c '
    set -euo pipefail
    wait_for_pid="$1"
    project_dir="$2"
    python_bin="$3"
    output_root="$4"
    folds="$5"
    variants="$6"
    epochs="$7"
    patience="$8"
    batch_size="$9"
    shift 9
    validate_every="$1"
    num_workers="$2"
    while kill -0 "${wait_for_pid}" 2>/dev/null; do
      sleep 60
    done
    cd "${project_dir}"
    exec "${python_bin}" scripts/run_random_point_ablation_kfold.py \
      --config configs/default.yaml \
      --output-root "${output_root}" \
      --folds "${folds}" \
      --variants "${variants}" \
      --epochs "${epochs}" \
      --patience "${patience}" \
      --batch-size "${batch_size}" \
      --validate-every "${validate_every}" \
      --num-workers "${num_workers}" \
      --skip-existing
  ' bash "${WAIT_FOR_PID}" "${PROJECT_DIR}" "${PYTHON_BIN}" "${OUTPUT_ROOT}" "${FOLDS}" "${VARIANTS}" "${EPOCHS}" "${PATIENCE}" "${BATCH_SIZE}" "${VALIDATE_EVERY}" "${NUM_WORKERS}" \
    > "${OUT_LOG}" 2> "${ERR_LOG}" < /dev/null &
  pid="$!"
  echo "${pid}" > "${PID_FILE}"
  echo "queued: pid=${pid} wait_for_pid=${WAIT_FOR_PID}"
  echo "stdout=${OUT_LOG}"
  echo "stderr=${ERR_LOG}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "variants=${VARIANTS}"
  exit 0
fi

nohup "${PYTHON_BIN}" scripts/run_random_point_ablation_kfold.py \
  --config configs/default.yaml \
  --output-root "${OUTPUT_ROOT}" \
  --folds "${FOLDS}" \
  --variants "${VARIANTS}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --validate-every "${VALIDATE_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --skip-existing \
  > "${OUT_LOG}" 2> "${ERR_LOG}" < /dev/null &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started: pid=${pid}"
echo "stdout=${OUT_LOG}"
echo "stderr=${ERR_LOG}"
echo "output_root=${OUTPUT_ROOT}"
echo "variants=${VARIANTS}"
