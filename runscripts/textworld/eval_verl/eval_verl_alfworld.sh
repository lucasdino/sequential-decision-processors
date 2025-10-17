HYDRA_FULL_ERROR=1

export RAY_DISABLE_DASHBOARD=1
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS
ROOT_DIR="$HOME/Desktop/UCSD/Research"
PROJ_DIR="$ROOT_DIR/sequential-decision-processors"


num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
val_data_size=32

# We only use data preparation to indicate the modality and the data size.
python3 -m verl_agent_sdp.examples.data_preprocess.prepare \
    --local_dir $PROJ_DIR/data/verl-agent \
    --mode 'text' \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$PROJ_DIR/data/verl-agent/text/test.parquet \
    data.output_path=$PROJ_DIR/data/verl-agent/text/test-output.parquet \
    data.batch_size=1 \
    data.n_samples=1 \
    model.path=$ROOT_DIR/models/Qwen2.5-0.5B-Instruct \
    rollout.prompt_length=128 \
    rollout.response_length=128 \
    rollout.max_model_len=256 \
    rollout.max_num_seqs=1 \
    rollout.gpu_memory_utilization=0.35 \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.tensor_model_parallel_size=1 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False

# python3 -m verl_agent_sdp.verl.trainer.main_eval \
#     data.path=$HOME/data/verl-agent/text/test.parquet \
#     actor_rollout_ref.model.path=$HOME/../models/Qwen2.5-0.5B-Instruct \
#     trainer.project_name='verl_agent_sdp' \
#     trainer.experiment_name='eval_qwen2.5_0.5b' \
#     trainer.n_gpus_per_node=1 \
#     trainer.logger=['console'] \
#     env.env_name=alfworld/AlfredTWEnv \
#     env.seed=0 \
#     env.max_steps=50 \
#     env.rollout.n=$group_size \
#     env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \ 