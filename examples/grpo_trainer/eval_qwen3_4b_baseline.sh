#!/usr/bin/env bash
# Inference-only evaluation | Qwen3-4B | DAPO-Math (EN) split and/or AIME24
#
# Measures the baseline accuracy and response length *without any training*, so the
# length-shaped runs from run_qwen3_4b_fsdp.sh have something to be compared against.
#
# This reuses the training pipeline with trainer.val_only=True (see ray_trainer.py:1419):
# one validation pass, then exit. Doing it this way keeps the prompt template, the answer
# parser, the reward components and the length metrics identical to training, so the
# before/after numbers are actually comparable.
#
# EVAL_SOURCE selects the eval set: split | aime24 | both
#
# Set MODEL_PATH to a trained checkpoint's `huggingface/` directory to reuse this script
# for the post-training evaluation.

set -xeuo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../..")

# ---- user-adjustable ----
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-/aiarena/group/rlgroup1/duanyuanrui/models/Qwen3-4B-Instruct-2507}

RAW_TRAIN_FILE=${RAW_TRAIN_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}
RAW_AIME24_FILE=${RAW_AIME24_FILE:-/aiarena/group/rlgroup1/duanyuanrui/data/RLVR-Linearity-Dataset/aime24.parquet}
SPLIT_TEST_RATIO=${SPLIT_TEST_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-42}
EVAL_SOURCE=${EVAL_SOURCE:-aime24}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-${REPO_ROOT}/examples/data_preprocess/prepare_dapo_en_aime24_for_grpo.py}
REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_ROOT}/examples/grpo_trainer/reward_math_verify_answer_boxed.py}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

# Sampling for evaluation. Must stay identical to the val_kwargs of the training script,
# otherwise the baseline and the post-training numbers are not comparable.
VAL_N=${VAL_N:-8}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.9}
# GREEDY=1 gives the single deterministic greedy answer per question. This is a different
# quantity from mean@N under sampling (which is the pass@1 *expectation*), and is the
# cheapest way to get a stable, reproducible number.
GREEDY=${GREEDY:-0}
VAL_DO_SAMPLE=True
if [ "${GREEDY}" = "1" ]; then
    VAL_N=1
    VAL_DO_SAMPLE=False
fi
# Prompts per generation call (each is expanded VAL_N times). Caps the memory peak;
# null would put the whole eval set in a single batch. Larger = more work queued in vLLM
# at once = better throughput, until host memory becomes the limit.
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}

ROLLOUT_TP=${ROLLOUT_TP:-1}
# Keep this well below 1.0. Before validation runs, ray_trainer.py:1408 syncs the FSDP
# actor weights into vLLM, and FSDP has to unshard full parameter tensors on the GPU to do
# it. If vLLM has already claimed most of the card, that unshard OOMs.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
# Transfer buffer for the weight sync; smaller means a lower transient peak.
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-512}
# vLLM scheduler limits. max_num_seqs is the concurrency ceiling; the effective one is
# whatever the KV cache can hold, so raising it only helps if there is spare cache.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

# Baseline measurement: no length shaping, so the reported score is acc + format only.
# Set to 0.5 if you want the shaped score used during training instead.
LEN_TARGET=${LEN_TARGET:-1024}
LEN_PENALTY_COEF=${LEN_PENALTY_COEF:-0}
TRUNCATED_WRONG_PENALTY=${TRUNCATED_WRONG_PENALTY:-0}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math_en}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-baseline_eval_${EVAL_SOURCE}_$([ "${GREEDY}" = "1" ] && echo greedy || echo n${VAL_N})}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${REPO_ROOT}/outputs/eval/${EXPERIMENT_NAME}}
# null dumps every sample, which is what lets you recompute the truncation ratio and the
# non-truncated mean length offline from the JSONL.
ROLLOUT_DUMP_MAX_SAMPLES=${ROLLOUT_DUMP_MAX_SAMPLES:-null}
LOGGER=${LOGGER:-'["console"]'}
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

# Data files, kept byte-identical to the ones the training script uses so that the
# baseline is measured on exactly the same questions.
DAPO_DIR=$(dirname "${RAW_TRAIN_FILE}")
SPLIT_TRAIN_FILE=${SPLIT_TRAIN_FILE:-${DAPO_DIR}/train_grpo_qwen3_4b_split.parquet}
SPLIT_TEST_FILE=${SPLIT_TEST_FILE:-${DAPO_DIR}/test_grpo_qwen3_4b_split.parquet}
FULL_TRAIN_FILE=${FULL_TRAIN_FILE:-${DAPO_DIR}/train_grpo_qwen3_4b.parquet}
AIME24_TEST_FILE=${AIME24_TEST_FILE:-$(dirname "${RAW_AIME24_FILE}")/aime24_grpo_qwen3_4b.parquet}

# The trainer always builds a train dataloader even with val_only=True, so a train file
# must exist. It is never iterated.
case "${EVAL_SOURCE}" in
    split)
        TRAIN_FILE=${SPLIT_TRAIN_FILE}
        VAL_FILES=${SPLIT_TEST_FILE}
        PREPARE_ARGS=(--train-output "${SPLIT_TRAIN_FILE}" --test-output "${SPLIT_TEST_FILE}"
                      --test-ratio "${SPLIT_TEST_RATIO}" --seed "${SPLIT_SEED}")
        REQUIRED_FILES=("${SPLIT_TRAIN_FILE}" "${SPLIT_TEST_FILE}")
        ;;
    aime24)
        TRAIN_FILE=${FULL_TRAIN_FILE}
        VAL_FILES=${AIME24_TEST_FILE}
        PREPARE_ARGS=(--train-output "${FULL_TRAIN_FILE}" --test-output "${AIME24_TEST_FILE}"
                      --test-input "${RAW_AIME24_FILE}")
        REQUIRED_FILES=("${FULL_TRAIN_FILE}" "${AIME24_TEST_FILE}")
        ;;
    both)
        # Per-eval-set metrics rely on the `data_source` column differing between the two
        # files; if they happen to share a value the metrics are merged into one group.
        TRAIN_FILE=${SPLIT_TRAIN_FILE}
        VAL_FILES="[${SPLIT_TEST_FILE},${AIME24_TEST_FILE}]"
        PREPARE_ARGS=()
        REQUIRED_FILES=("${SPLIT_TRAIN_FILE}" "${SPLIT_TEST_FILE}" "${AIME24_TEST_FILE}")
        ;;
    *)
        echo "Unsupported EVAL_SOURCE=${EVAL_SOURCE}. Expected 'split', 'aime24' or 'both'." >&2
        exit 1
        ;;
esac

NEED_PREPROCESS=${FORCE_PREPROCESS}
for f in "${REQUIRED_FILES[@]}"; do
    [ -f "${f}" ] || NEED_PREPROCESS=1
done

if [ "${NEED_PREPROCESS}" = "1" ]; then
    if [ "${EVAL_SOURCE}" = "both" ]; then
        echo "EVAL_SOURCE=both requires the parquet files to exist already." >&2
        echo "Run this script once with EVAL_SOURCE=split and once with EVAL_SOURCE=aime24 first." >&2
        exit 1
    fi
    python3 "${PREPARE_SCRIPT}" --train-input "${RAW_TRAIN_FILE}" "${PREPARE_ARGS[@]}"
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILES}"
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    # Never iterated (val_only), kept small so the unused train dataloader is cheap.
    data.train_batch_size=8
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    # No optimizer step happens, but the actor worker is still constructed.
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    # False so the reference-policy worker is not created at all: no KL means no ref logprobs.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    # Inference-only, so chunked prefill interleaves prefill with decode instead of
    # stalling it. free_cache_engine stays True: the KV cache has to be released around
    # the pre-validation weight sync, otherwise the FSDP unshard has nowhere to allocate.
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.len_target=${LEN_TARGET}
    +reward.custom_reward_function.reward_kwargs.len_penalty_coef=${LEN_PENALTY_COEF}
    +reward.custom_reward_function.reward_kwargs.truncated_wrong_penalty=${TRUNCATED_WRONG_PENALTY}
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=1024
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    # One validation pass, then return before the training loop starts.
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.total_epochs=1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.rollout_dump_max_samples=${ROLLOUT_DUMP_MAX_SAMPLES}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"

set +x
echo "Per-sample results: ${VALIDATION_DATA_DIR}/0.jsonl"
echo "Each line carries acc / em / fuzzy / format_strict / truncated / response_length,"
echo "which is everything needed to recompute the non-truncated mean length,"
echo "the truncation ratio and the EM / fuzzy accuracy offline."



