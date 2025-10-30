#!/bin/bash
#SBATCH --job-name=Alfworld_AC_SC_RIC2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # 1 task (Python process) on each node
#SBATCH --gpus-per-task=8       # 8 GPUs exposed to that task
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm/%x/%x-%j.out
#SBATCH --error=slurm/%x/%x-%j.err

# Positive reward 'death' penalty, deterministic 3-turn drift, self-critique
# PPO

echo "Starting job $SLURM_JOB_ID on $SLURM_NNODES nodes with $SLURM_NTASKS_PER_NODE tasks per node."

# Putting this at top to make my life easier
env_seed=2

prompt_template=alfworld_ac
prompt_template=alfworld_ac_sctq_ric
# prompt_template=alfworld_with_help_sctq_inst_first_with_think
# prompt_template=sctq_inst_first_with_think
# prompt_template=inst_first_with_think
prompt_template=basecase
# prompt_template=sctq_inst_first
# prompt_template=sctq_inst_first_with_think_extended

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"


declare -A pids
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION_2   # not XFORMERS
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=2          # NVLink → IB hierarchy
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



verl_workdir=$HOME/

DATA_DIR=$HOME/data/verl-agent/text
ENGINE=${1:-vllm}
TRAIN_PARQUET=$DATA_DIR/train.parquet
VAL_PARQUET=$DATA_DIR/test.parquet

# To fix a triton cache issue, setting a node-local cache directory
export TRITON_CACHE_DIR=/tmp/triton-cache/$SLURM_NODEID
mkdir -p $TRITON_CACHE_DIR

# =================== Ray start ===================
# ray stop at all nodes
# srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=4 ray stop

srun --nodes=$SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE ray stop
sleep 5
srun --nodes=$SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE rm -rf /tmp/ray/ray_current_cluster
sleep 5

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

echo "Waiting for Ray head to be ready at ${address_head}…"
until ${CONDA_BIN_PATH}ray status --address="${address_head}" &>/dev/null; do
  echo "  head not up yet, retrying in 2s…"
  sleep 2
done
echo "Ray head is up!"


export address_head="${head_node_ip}:${port}"


# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10

# Make sure the workers are actually started and terminate if not.
echo "Checking Ray cluster status..."
RAY_STATUS_OUTPUT=$(${CONDA_BIN_PATH}ray status --address="${head_node_ip}:${port}")
echo "$RAY_STATUS_OUTPUT"

echo "✓ Ray cluster properly initialized with $ACTIVE_NODES nodes and $TOTAL_GPUS GPUs"


conda activate verl-agent
echo "${CONDA_BIN_PATH}"

# Copying Taylor because this is much neater:
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up modelforward, but note memory overflow
offload=False

n_resp_per_prompt_val=1
total_epochs=200
save_freq=15
test_freq=15
max_ckpt_to_keep=1
enable_curriculum=True
val_before_train=True
train_prompt_bsz=32
val_prompt_bsz=32
max_prompt_length=$((1024 * 15))
max_response_length=512
max_total_length=$((max_prompt_length + max_response_length))
num_nodes=$SLURM_NNODES
micro_bs_per_gpu=$((32 / (num_nodes*8)))
micro_batch_per_gpu=32

pwd

export HF_TOKEN=""

# echo "Checking that the environment is set up correctly…"
# "${CONDA_BIN_PATH}python" verl-dead-agent/agent_system/environments/env_manager.py

# Need to uncomment this if running on a new map for first time.
echo "Preprocessing data…"
"${CONDA_BIN_PATH}python" -m verl-dead-agent.examples.data_preprocess.prepare \
    --mode text \
    --train_data_size ${train_prompt_bsz} \
    --val_data_size ${val_prompt_bsz}

# Organizing this a bit more

"${CONDA_BIN_PATH}python" -m verl-dead-agent.verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=${TRAIN_PARQUET} \
    data.val_files=${VAL_PARQUET} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz}\
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    \
    \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    \
    actor_rollout_ref.actor.use_torch_compile=False \
    \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    \
    +actor_rollout_ref.actor.use_entropy_advantage=False \
    +actor_rollout_ref.actor.entropy_advantage_alpha=0.01 \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_total_length} \
    \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy="HYBRID_SHARD" \
    +actor_rollout_ref.actor.fsdp_config.backward_prefetch="BACKWARD_PRE" \
    +critic.model.fsdp_config.sharding_strategy="HYBRID_SHARD" \
    +critic.model.fsdp_config.backward_prefetch="BACKWARD_PRE" \
    \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.dtype='auto' \
    \
    actor_rollout_ref.rollout.max_num_batched_tokens=60000\
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    critic.optim.lr=3e-6 \
    critic.model.path=Qwen/Qwen3-8B\
    critic.ppo_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    critic.model.enable_activation_offload=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.reshard_after_forward=False \
    critic.model.use_remove_padding=True \
    critic.use_dynamic_bsz=True \
    \
    algorithm.use_kl_in_reward=False \
    env.env_name=tales_alfworld \
    env.seed=${env_seed} \
    env.max_steps=25 \
    +env.prompt_template=${prompt_template} \
    +env.reward_mode="goal-only" \
    +env.num_envs_per_batch=1 \
    trainer.logger=["console","wandb"] \
    trainer.log_val_generations=30 \
    trainer.project_name="verl_agent_alfworld" \
    trainer.experiment_name="ppo_qwen3-8b_alfworld"_${prompt_template} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.save_full_model=True \
    +trainer.consolidate_checkpoints=True \
    +trainer.save_only_on_master=True \