#!/bin/bash
#SBATCH --job-name=Alfworld_AC_SC_RIC2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # 1 task (Python process) on each node
#SBATCH --gpus-per-task=8       # 8 GPUs exposed to that task
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm/%x/%x-%j.out
#SBATCH --error=slurm/%x/%x-%j.err




# ##############################
# OPTIONAL - Multi-node runs
# ##############################
# echo "Starting job $SLURM_JOB_ID on $SLURM_NNODES nodes with $SLURM_NTASKS_PER_NODE tasks per node."
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"
declare -A pids
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port



# ##############################
# Setup flags
# ##############################
export worker_num=$SLURM_NNODES
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
verl_workdir=$HOME/verl_dead_agent/
DATA_DIR=$HOME/data/verl-agent/text
ENGINE=${1:-vllm}
TRAIN_PARQUET=$DATA_DIR/train.parquet
VAL_PARQUET=$DATA_DIR/test.parquet

# To fix a triton cache issue, setting a node-local cache directory
export TRITON_CACHE_DIR=/tmp/triton-cache/$SLURM_NODEID
mkdir -p $TRITON_CACHE_DIR



# ##############################
# Ray start
# ##############################
srun --nodes=$SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE ray stop
sleep 5
srun --nodes=$SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE rm -rf /tmp/ray/ray_current_cluster
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



# ##############################
# Setup Env
# ##############################
pwd
source .venv/bin/activate

env_seed=2
prompt_template=basecase
model_path=$HOME/models/Qwen2.5-7B-Instruct
wandb_project_name=sdp_alfworld_rejsampling
experiment_name=sdp-q25-3b-r
total_epochs=100
save_freq=-1
test_freq=10
max_ckpt_to_keep=1
val_before_train=False
train_prompt_bsz=32
val_prompt_bsz=32
max_prompt_length=$((512 * 7))
max_response_length=512
max_total_length=$((max_prompt_length + max_response_length))
num_nodes=$SLURM_NNODES
micro_bs_per_gpu=$((32 / (num_nodes*8)))


# We only use data preparation to indicate the modality and the data size.
echo "Preprocessing data…"
"${CONDA_BIN_PATH}python" -m verl-dead-agent.examples.data_preprocess.prepare \
    --mode text \
    --train_data_size ${train_prompt_bsz} \
    --val_data_size ${val_prompt_bsz}

"${CONDA_BIN_PATH}python" -m verl-dead-agent.verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
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
    actor_rollout_ref.model.path=${model_path} \
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
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-sum-norm' \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bs_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_total_length} \
    \
    +actor_rollout_ref.actor.fsdp_config.sharding_strategy="HYBRID_SHARD" \
    +actor_rollout_ref.actor.fsdp_config.backward_prefetch="BACKWARD_PRE" \
    \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_total_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.dtype='auto' \
    actor_rollout_ref.rollout.n=8 \
    \
    actor_rollout_ref.rollout.max_num_batched_tokens=60000\
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    \
    algorithm.use_kl_in_reward=False \
    env.env_name=tales_alfworld \
    env.seed=${env_seed} \
    env.max_steps=50 \
    +env.prompt_template=${prompt_template} \
    +env.reward_mode="goal-only" \
    +env.num_envs_per_batch=1 \
    trainer.logger=["console","wandb"] \
    trainer.log_val_generations=30 \
    trainer.project_name=${wandb_project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=${val_before_train} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${num_nodes} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=1 \