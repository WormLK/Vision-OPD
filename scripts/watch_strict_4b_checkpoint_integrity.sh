#!/usr/bin/env bash
set -u

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
PYTHON="/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_VERIFY_ATTEMPTS="${MAX_VERIFY_ATTEMPTS:-10}"
last_verified=0

count_nonempty() {
  local directory="$1" pattern="$2"
  find "${directory}" -maxdepth 1 -type f -name "${pattern}" -size +0c 2>/dev/null | wc -l
}

latest_step() {
  [[ -f "${TRACKER}" ]] && tr -cd '0-9' < "${TRACKER}" || printf '0'
}

verify_checkpoint() {
  local step="$1" checkpoint actor teacher attempt
  local actor_model actor_optim actor_extra teacher_model
  checkpoint="${CHECKPOINT_ROOT}/global_step_${step}"
  actor="${checkpoint}/actor"
  teacher="${actor}/teacher"

  for attempt in $(seq 1 "${MAX_VERIFY_ATTEMPTS}"); do
    actor_model="$(count_nonempty "${actor}" 'model_world_size_8_rank_*.pt')"
    actor_optim="$(count_nonempty "${actor}" 'optim_world_size_8_rank_*.pt')"
    actor_extra="$(count_nonempty "${actor}" 'extra_state_world_size_8_rank_*.pt')"
    teacher_model="$(count_nonempty "${teacher}" 'model_world_size_8_rank_*.pt')"
    if [[ "${actor_model}" == 8 && "${actor_optim}" == 8 && "${actor_extra}" == 8 \
      && "${teacher_model}" == 8 && -s "${checkpoint}/data.pt" \
      && -s "${actor}/huggingface/config.json" && -s "${teacher}/huggingface/config.json" ]]; then
      if ! "${PYTHON}" "${PROJECT_ROOT}/scripts/validate_strict_checkpoint.py" \
        "${CHECKPOINT_ROOT}" --step "${step}"; then
        printf '%s step=%s status=invalid_semantics\n' \
          "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${step}"
        return 1
      fi
      printf '%s step=%s status=complete actor_model=%s actor_optim=%s actor_extra=%s teacher_model=%s\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${step}" "${actor_model}" "${actor_optim}" \
        "${actor_extra}" "${teacher_model}"
      return 0
    fi
    sleep 30
  done

  printf '%s step=%s status=incomplete actor_model=%s actor_optim=%s actor_extra=%s teacher_model=%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${step}" "${actor_model}" "${actor_optim}" \
    "${actor_extra}" "${teacher_model}"
  return 1
}

while (( last_verified < 65 )); do
  step="$(latest_step)"
  if (( step > last_verified )); then
    verify_checkpoint "${step}" || exit 1
    last_verified="${step}"
  fi
  sleep "${INTERVAL_SECONDS}"
done
