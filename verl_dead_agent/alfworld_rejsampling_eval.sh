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
uv run -m verl_agent_sdp.examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir ${DATA_DIR} \
    --train_data_size ${train_prompt_bsz} \
    --val_data_size ${val_prompt_bsz}

uv run -m verl_dead_agent.verl.trainer.main_eval \
    data.train_files=${TRAIN_PARQUET} \
    data.path=${VAL_PARQUET} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.val_batch_size=${val_prompt_bsz} \
    \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    env.env_name=${env_name} \
    env.seed=${env_seed} \
    env.max_steps=${env_max_steps} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    +env.prompt_template=${prompt_template} \
    +env.reward_mode="goal-only" \
    \
    trainer.logger=["console","wandb"] \
    trainer.project_name=${wandb_project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \

echo "All done!"