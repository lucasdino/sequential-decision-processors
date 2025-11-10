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
PROJ_DIR=$HOME_DIR/sequential-decision-processors
verl_workdir=$PROJ_DIR/verl_dead_agent/
DATA_DIR=$PROJ_DIR/data/verl-agent/text
ENGINE=${1:-vllm}
TRAIN_PARQUET=$DATA_DIR/train.parquet
VAL_PARQUET=$DATA_DIR/test.parquet
REJ_SAMPLING_DATA_DIR=$PROJ_DIR/rej_sampling_data



# ##############################
# Main training args we'll change
# ##############################
env_seed=2
env_name=tales_alfworld
env_max_steps=50
prompt_template=basecase
model_path=$PROJ_DIR/models/Qwen2.5-7B-Instruct
wandb_project_name=sdp_alfworld_rejsampling
experiment_name=sdp-q25-7b-rejsampling
total_epochs=100
save_freq=-1
test_freq=10
group_size=2
num_cpus_per_env_worker=0.1
train_prompt_bsz=32
val_prompt_bsz=32
max_prompt_length=$((512 * 7))
max_response_length=512
max_total_length=$((max_prompt_length + max_response_length))
num_nodes=1
micro_bs_per_gpu=$((32 / (num_nodes*8)))


# We only use data preparation to indicate the modality and the data size.
uv run -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir ${DATA_DIR} \
    --train_data_size ${train_prompt_bsz} \
    --val_data_size ${val_prompt_bsz}

uv run -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    \
    \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz}\
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.dtype='auto' \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=60000\
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy="HYBRID_SHARD" \
    +actor_rollout_ref.actor.fsdp_config.backward_prefetch="BACKWARD_PRE" \
    \
    env.env_name=${env_name} \
    env.seed=${env_seed} \
    env.max_steps=${env_max_steps} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    +env.prompt_template=${prompt_template} \
    +env.reward_mode="goal-only" \
    +env.num_envs_per_batch=1 \
    \
    \
    +trainer.rejection_sampling=True \
    +trainer.rollout_data_dir=${REJ_SAMPLING_DATA_DIR} \
    trainer.total_training_steps=1000 \
    trainer.logger=["console","wandb"] \
    trainer.log_val_generations=30 \
    trainer.project_name=${wandb_project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=1 \


echo "All done!"