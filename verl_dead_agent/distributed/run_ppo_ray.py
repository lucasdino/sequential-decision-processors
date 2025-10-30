# -*- coding: utf-8 -*-
"""
Launch PPO with one vLLM worker per GPU.

Usage
-----
python -m verl_dead_agent.distributed.run_ppo_ray \
       +hydra=override,your,current,flags
"""
import os, itertools, ray
from functools import partial
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ───────────────────────────────── vLLM worker ──────────────────────────────
@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(self,
                 model_name: str,
                 max_model_len: int = 8192,
                 gpu_mem_util: float = 0.70):
        """Each actor gets a **single** GPU (Ray sets CUDA_VISIBLE_DEVICES)."""
        from vllm import LLM                         # local import – after CUDA vis fix
        self.engine = LLM(
            model=model_name,
            tensor_parallel_size=1,                 # <- single‑GPU replica
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            dtype="bfloat16",
        )

    def generate(self, prompts, sampling_dict):
        from vllm import SamplingParams
        sparams = SamplingParams(**sampling_dict)
        outs = self.engine.generate(prompts, sparams)
        return [o.outputs[0].text for o in outs]

# ───────────────────────────── patch helper ────────────────────────────────
def make_distributed_generate(workers, default_sampling):
    """Return a function that round‑robins prompts across workers."""
    rr = itertools.cycle(workers)

    def _generate(prompts, **kw):
        sampling = {**default_sampling, **kw}
        # Futures per prompt → round‑robin over workers
        futs = [next(rr).generate.remote([p], sampling) for p in prompts]
        # Each future returns a *one‑item* list; flatten
        return [x[0] for x in ray.get(futs)]

    return _generate

# ─────────────────────────────────── main ──────────────────────────────────
def main():
    # 1. attach to Ray cluster that Slurm launched
    ray.init(address="auto", ignore_reinit_error=True)

    # 2. spin up 1 worker per GPU
    n_gpus = int(ray.available_resources().get("GPU", 0))
    model_path = "Qwen/Qwen3-8B"
    workers = [VLLMWorker.remote(model_path) for _ in range(n_gpus)]

    # 3. patch the rollout LLM so every .generate() goes remote
    from verl_dead_agent.verl.llm.llm_wrapper import LLMWrapper  # <‑­ existing class that calls vllm
    default_sampling = dict(temperature=0.7, top_p=0.95, max_tokens=1024)
    LLMWrapper.generate = make_distributed_generate(workers, default_sampling)

    # 4. build hydra config exactly as before and launch training
    with initialize_config_dir("verl-dead-agent/conf"):          # root of your configs
        cfg = compose(config_name="ppo_qwen3_8b")                # or whatever you call it
    from verl_dead_agent.verl.trainer.main_ppo import train      # unchanged
    train(cfg)


if __name__ == "__main__":
    main()
