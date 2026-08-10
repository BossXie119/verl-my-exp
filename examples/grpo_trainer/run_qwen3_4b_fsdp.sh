#!/usr/bin/env bash
# GRPO | Qwen3-4B | FSDP training | DAPO-Math (EN) train | configurable test set
#
# Test set source is controlled by TEST_SOURCE:
#   - split  (default): randomly hold out a fraction of the DAPO-English train set
#   - aime24          : original AIME24 eval set
# This recipe normalizes the parquet files to a shared prompt format and uses
# math_verify-based rule reward with DAPO overlong handling.

set -xeuo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")

# ---- user-adjustable ----
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-/aiarena/group/rlgroup1/duanyuanrui/models/Qwen3-4B-Instruct-2507}

RAW_TRAIN_FILE=${RAW_TRAIN_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}
RAW_AIME24_FILE=${RAW_AIME24_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/RLVR-Linearity-Dataset/aime24.parquet}
SPLIT_TEST_RATIO=${SPLIT_TEST_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-42}
# Test source: split (default, hold out from DAPO-English train set) or aime24 (original AIME24 eval set).
TEST_SOURCE=${TEST_SOURCE:-split}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-${REPO_ROOT}/examples/data_preprocess/prepare_dapo_en_aime24_for_grpo.py}
REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_ROOT}/examples/grpo_trainer/reward_math_verify_answer_boxed.py}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-4}

# Validation sampling: n>1 + sampling makes best/worst/maj@N metrics meaningful.
VAL_N=${VAL_N:-4}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.9}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math_en}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_math_verify_aime24_grpo}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${REPO_ROOT}/outputs/rollouts/${EXPERIMENT_NAME}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${REPO_ROOT}/outputs/val/${EXPERIMENT_NAME}}
# Number of samples to dump per step (rollout + validation). 4 = one full GRPO group (one question x N responses).
ROLLOUT_DUMP_MAX_SAMPLES=${ROLLOUT_DUMP_MAX_SAMPLES:-4}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
# ---- end user-adjustable ----

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        NGPUS_PER_NODE=16
        ROLLOUT_GPU_MEM_UTIL=0.9
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

# Resolve data files and preprocess args according to the test source.
case "${TEST_SOURCE}" in
    split)
        TRAIN_FILE=${TRAIN_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train_grpo_qwen3_4b_split.parquet}
        TEST_FILE=${TEST_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/test_grpo_qwen3_4b_split.parquet}
        PREPARE_ARGS=(--test-ratio "${SPLIT_TEST_RATIO}" --seed "${SPLIT_SEED}")
        ;;
    aime24)
        TRAIN_FILE=${TRAIN_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train_grpo_qwen3_4b.parquet}
        TEST_FILE=${TEST_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/RLVR-Linearity-Dataset/aime24_grpo_qwen3_4b.parquet}
        PREPARE_ARGS=(--test-input "${RAW_AIME24_FILE}")
        ;;
    *)
        echo "Unsupported TEST_SOURCE=${TEST_SOURCE}. Expected 'split' or 'aime24'." >&2
        exit 1
        ;;
esac

if [ "${FORCE_PREPROCESS}" = "1" ] || [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
    python3 "${PREPARE_SCRIPT}" \
        --train-input "${RAW_TRAIN_FILE}" \
        --train-output "${TRAIN_FILE}" \
        --test-output "${TEST_FILE}" \
        "${PREPARE_ARGS[@]}"
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} 
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.rollout_dump_max_samples=${ROLLOUT_DUMP_MAX_SAMPLES}
)

EXTRA=(
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
