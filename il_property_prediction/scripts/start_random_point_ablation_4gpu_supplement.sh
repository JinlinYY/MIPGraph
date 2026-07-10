#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/venvs/mipgraph/bin/python}"
RUN_ID="${RUN_ID:-ablation_random_point_kfold_seed42_4gpu_supplement}"
LOG_DIR="${LOG_DIR:-tmp/training_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablation_random_point_kfold_seed42_supplement}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VALIDATE_EVERY="${VALIDATE_EVERY:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SHARD_VARIANTS="${SHARD_VARIANTS:-unimol2_bilinear,unimol2_cross_transformer_global_desc,unimol2_cross_transformer_global_fg_desc|unimol2_cross_transformer_unfreeze1,unimol2_cross_transformer_unfreeze2|unimol2_cross_transformer_topk1,unimol2_cross_transformer_topk3,unimol2_cross_transformer_no_condition_film|unimol2_cross_transformer_no_moe_prior,unimol2_cross_transformer_no_moe_load_balance,unimol2_cross_transformer_no_moe_regularizers}"

mkdir -p "${LOG_DIR}"
IFS='|' read -r -a SHARDS <<< "${SHARD_VARIANTS}"

for gpu_id in "${!SHARDS[@]}"; do
  variants="${SHARDS[$gpu_id]}"
  if [[ -z "${variants}" ]]; then
    continue
  fi
  shard_name="${RUN_ID}_gpu${gpu_id}"
  pid_file="${LOG_DIR}/${shard_name}.pid"
  out_log="${LOG_DIR}/${shard_name}.out.log"
  err_log="${LOG_DIR}/${shard_name}.err.log"
  if [[ -s "${pid_file}" ]]; then
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "already running: gpu=${gpu_id} pid=${old_pid} variants=${variants}"
      continue
    fi
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" nohup "${PYTHON_BIN}" scripts/run_random_point_ablation_kfold.py \
    --config configs/default.yaml \
    --output-root "${OUTPUT_ROOT}" \
    --folds "${FOLDS}" \
    --variants "${variants}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --validate-every "${VALIDATE_EVERY}" \
    --num-workers "${NUM_WORKERS}" \
    --skip-existing \
    > "${out_log}" 2> "${err_log}" < /dev/null &
  pid="$!"
  echo "${pid}" > "${pid_file}"
  echo "started: gpu=${gpu_id} pid=${pid} variants=${variants}"
  echo "stdout=${out_log}"
  echo "stderr=${err_log}"
done

echo "output_root=${OUTPUT_ROOT}"
