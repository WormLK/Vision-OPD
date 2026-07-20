#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-full-repro-20260713}"
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh
