#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION_2   # not XFORMERS
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=2            # NVLink → IB hierarchy
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_CROSS_NIC=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GDR_LEVEL=2             # you already set this
export NCCL_MIN_NCHANNELS=4            # often helps multi-NIC; tune 2–8
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1   # recommended with FlashAttn
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ##############################
# Define file paths
# ##############################
HOME_DIR="/home"
PROJ_DIR=${HOME_DIR}/sequential-decision-processors
verl_workdir=${PROJ_DIR}/verl_dead_agent
DATA_DIR=${PROJ_DIR}/data/verl-agent
ENGINE=${1:-vllm}
TRAIN_PARQUET=${DATA_DIR}/text/train.parquet
VAL_PARQUET=${DATA_DIR}/text/test.parquet
DATA_OUT_DIR=${PROJ_DIR}/rej_sampling_data

# ##############################
# Main training args
# ##############################
env_seed=42
env_name=tales_twx_alfworld
twx_max_steps=40
alfworld_max_steps=25
prompt_template=base_with_verbs_context
tokenizer_type=qwen3
valid_seen=False
load_env_seeds=True
model_path=Qwen/Qwen3-32B
run_type=rejection_sampling
wandb_project_name=sdp_alfworld_rejsampling
experiment_name=q32b-11-26-rft-all
train_prompt_bsz=64
val_prompt_bsz=64
rollout_n=1
max_prompt_length=$((512 * 3))
max_response_length=$((512 * 2))
max_total_length=$((max_prompt_length + max_response_length))
num_nodes=1
micro_bs_per_gpu=$((64 / (num_nodes * 8)))
num_cpus_per_env_worker=0.25
save_freq=-1
test_freq=20
total_epochs=1
# One 64-sample batches per environment (twx & alfworld)
train_steps=4

rollout_save_dir=${DATA_OUT_DIR}
mkdir -p "${rollout_save_dir}"

# ##############################
# Prep synthetic text inputs (same modality as training)
# ##############################
uv run -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir ${DATA_DIR} \
    --train_data_size 1024 \
    --val_data_size 128

# ##############################
# Single UV run covering both envs via multi-env manager
# ##############################
uv run -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-sum-norm' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype='auto' \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy="HYBRID_SHARD" \
    +actor_rollout_ref.actor.fsdp_config.backward_prefetch="BACKWARD_PRE" \
    \
    env.env_name=${env_name} \
    env.seed=${env_seed} \
    env.max_steps=${twx_max_steps} \
    env.rollout.n=${rollout_n} \
    +env.multi_env_scheduler.env_overrides.twx.max_steps=${twx_max_steps} \
    +env.multi_env_scheduler.env_overrides.alfworld.max_steps=${alfworld_max_steps} \
    +env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    +env.prompt_template=${prompt_template} \
    +env.reward_mode="goal-only" \
    +env.num_envs_per_batch=${rollout_n} \
    +env.tokenizer=${tokenizer_type} \
    +env.valid_seen=${valid_seen} \
    +env.load_env_seeds=${load_env_seeds} \
    \
    +intermediary.enabled=False \
    \
    +trainer.run_type=${run_type} \
    +trainer.rollout_data_dir=${rollout_save_dir} \
    trainer.total_training_steps=${train_steps} \
    trainer.logger=['wandb'] \
    trainer.log_val_generations=0 \
    trainer.project_name=${wandb_project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=1