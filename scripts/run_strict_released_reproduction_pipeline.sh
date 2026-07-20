#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
LOG_DIR="${PROJECT_ROOT}/logs"
READY_MARKER="${PROJECT_ROOT}/outputs/vision_opd_official_existing_checkpoints_complete"
EXPECTED_STEP="${EXPECTED_STEP:-65}"
MAX_TRAIN_ATTEMPTS="${MAX_TRAIN_ATTEMPTS:-100}"
MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-20}"

FOUR_EXP="Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
NINE_EXP="Vision-OPD-Qwen3.5-9B-released-b96-r8-gradaccum-sp8"
FOUR_CKPT="${PROJECT_ROOT}/checkpoints/${FOUR_EXP}"
NINE_CKPT="${PROJECT_ROOT}/checkpoints/${NINE_EXP}"
FOUR_MERGED="${PROJECT_ROOT}/merged_models/${FOUR_EXP}"
NINE_MERGED="${PROJECT_ROOT}/merged_models/${NINE_EXP}"
FOUR_BASE="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B"
NINE_BASE="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b"
FOUR_SERVED="Vision-OPD-Qwen3.5-4B-released-b96-r8-official"
NINE_SERVED="Vision-OPD-Qwen3.5-9B-released-b96-r8-official"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/merged_models"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

latest_step() {
  local marker="$1/latest_checkpointed_iteration.txt"
  if [[ -f "${marker}" ]]; then
    tr -cd '0-9' < "${marker}"
  else
    printf '0'
  fi
}

run_training() {
  local label="$1" wrapper="$2" checkpoint_dir="$3" train_log="$4"
  local attempt step status
  for attempt in $(seq 1 "${MAX_TRAIN_ATTEMPTS}"); do
    step="$(latest_step "${checkpoint_dir}")"
    if (( step >= EXPECTED_STEP )); then
      log "${label} reached strict checkpoint ${step}/${EXPECTED_STEP}."
      return 0
    fi
    log "Starting ${label} strict training attempt ${attempt} at ${step}/${EXPECTED_STEP}."
    ray stop --force >> "${train_log}" 2>&1 || true
    status=0
    bash "${wrapper}" >> "${train_log}" 2>&1 || status=$?
    step="$(latest_step "${checkpoint_dir}")"
    if (( status == 0 && step >= EXPECTED_STEP )); then
      log "${label} strict training completed at checkpoint ${step}."
      return 0
    fi
    log "${label} attempt ${attempt} exited with status ${status}; latest checkpoint is ${step}."
    sleep $((attempt < 10 ? attempt * 30 : 300))
  done
  return 1
}

merge_latest() {
  local checkpoint_dir="$1" target="$2"
  local step source
  step="$(latest_step "${checkpoint_dir}")"
  source="${checkpoint_dir}/global_step_${step}"
  if [[ -f "${target}/config.json" ]] && compgen -G "${target}/model*.safetensors" >/dev/null; then
    if python scripts/validate_merged_model.py "${target}"; then
      log "Verified existing merged model ${target}."
      return 0
    fi
    log "Existing merged model failed validation; preserving it as incomplete."
  fi
  if [[ -d "${target}" ]] && [[ -n "$(find "${target}" -mindepth 1 -print -quit)" ]]; then
    mv "${target}" "${target}.incomplete.$(date -u +'%Y%m%dT%H%M%SZ')"
  fi
  TARGET_DIR="${target}" bash scripts/merge_checkpoint.sh "${source}"
  python scripts/validate_merged_model.py "${target}"
}

evaluate_model() {
  local path="$1" name="$2" tp="$3" dp="$4" eval_log="$5"
  local attempt
  for attempt in $(seq 1 "${MAX_EVAL_ATTEMPTS}"); do
    log "Evaluating ${name}, attempt ${attempt}."
    ray stop --force >> "${eval_log}" 2>&1 || true
    if MODEL_PATH="${path}" SERVED_MODEL_NAME="${name}" MODEL_TP="${tp}" MODEL_DP="${dp}" \
      bash scripts/evaluate_official_single_model.sh >> "${eval_log}" 2>&1; then
      log "${name} completed strict official evaluation."
      return 0
    fi
    sleep $((attempt < 5 ? attempt * 60 : 300))
  done
  return 1
}

python scripts/validate_lazy_image_equivalence.py \
  --model "${FOUR_BASE}" --data data/train.parquet \
  --chat-template chat_templates/perception_chat_template_qwen35.jinja --samples 3 \
  >> "${LOG_DIR}/strict_released_4b_lazy_image_validation.log" 2>&1
run_training 4B scripts/run_vision_opd_released_b96_r8_gradaccum_4b.sh \
  "${FOUR_CKPT}" "${LOG_DIR}/strict_released_4b_training.log"
python scripts/validate_strict_runtime_config.py \
  --log "${LOG_DIR}/strict_released_4b_training.log" --backbone 4b \
  >> "${LOG_DIR}/strict_released_4b_runtime_config_validation.log" 2>&1
python scripts/validate_strict_training_metrics.py \
  --tensorboard-dir "${PROJECT_ROOT}/tensorboard_log/Vision-OPD/${FOUR_EXP}" \
  --expected-step "${EXPECTED_STEP}" \
  >> "${LOG_DIR}/strict_released_4b_metric_validation.log" 2>&1
python scripts/validate_strict_checkpoint.py "${FOUR_CKPT}" --step "${EXPECTED_STEP}" \
  >> "${LOG_DIR}/strict_released_4b_checkpoint_validation.log" 2>&1
merge_latest "${FOUR_CKPT}" "${FOUR_MERGED}" \
  >> "${LOG_DIR}/strict_released_4b_merge.log" 2>&1

if [[ ! -f "${READY_MARKER}" ]]; then
  for attempt in $(seq 1 "${MAX_EVAL_ATTEMPTS}"); do
    log "Running official baseline alignment and existing-checkpoint evaluation, attempt ${attempt}."
    if bash scripts/continue_official_eval_after_judge_download.sh \
      >> "${LOG_DIR}/official_post_4b_prerequisite_pipeline.log" 2>&1; then
      break
    fi
    sleep $((attempt < 5 ? attempt * 60 : 300))
  done
fi
[[ -f "${READY_MARKER}" ]]

evaluate_model "${FOUR_MERGED}" "${FOUR_SERVED}" 1 6 \
  "${LOG_DIR}/strict_released_4b_evaluation.log"
python scripts/verify_official_opd_alignment.py --project-root "${PROJECT_ROOT}" \
  --model "${FOUR_SERVED}" --backbone 4b \
  | tee "${LOG_DIR}/strict_released_4b_alignment.log"
touch outputs/vision_opd_strict_released_4b_complete

python scripts/validate_lazy_image_equivalence.py \
  --model "${NINE_BASE}" --data data/train.parquet \
  --chat-template chat_templates/perception_chat_template_qwen35.jinja --samples 3 \
  >> "${LOG_DIR}/strict_released_9b_lazy_image_validation.log" 2>&1
run_training 9B scripts/run_vision_opd_released_b96_r8_gradaccum_9b.sh \
  "${NINE_CKPT}" "${LOG_DIR}/strict_released_9b_training.log"
python scripts/validate_strict_runtime_config.py \
  --log "${LOG_DIR}/strict_released_9b_training.log" --backbone 9b \
  >> "${LOG_DIR}/strict_released_9b_runtime_config_validation.log" 2>&1
python scripts/validate_strict_training_metrics.py \
  --tensorboard-dir "${PROJECT_ROOT}/tensorboard_log/Vision-OPD/${NINE_EXP}" \
  --expected-step "${EXPECTED_STEP}" \
  >> "${LOG_DIR}/strict_released_9b_metric_validation.log" 2>&1
python scripts/validate_strict_checkpoint.py "${NINE_CKPT}" --step "${EXPECTED_STEP}" \
  >> "${LOG_DIR}/strict_released_9b_checkpoint_validation.log" 2>&1
merge_latest "${NINE_CKPT}" "${NINE_MERGED}" \
  >> "${LOG_DIR}/strict_released_9b_merge.log" 2>&1
evaluate_model "${NINE_MERGED}" "${NINE_SERVED}" 2 3 \
  "${LOG_DIR}/strict_released_9b_evaluation.log"
python scripts/verify_official_opd_alignment.py --project-root "${PROJECT_ROOT}" \
  --model "${NINE_SERVED}" --backbone 9b \
  | tee "${LOG_DIR}/strict_released_9b_alignment.log"
touch outputs/vision_opd_strict_released_9b_complete

python scripts/summarize_official_evaluation.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
touch outputs/vision_opd_strict_released_reproduction_complete
log "Strict released 4B and 9B reproduction pipeline completed."
