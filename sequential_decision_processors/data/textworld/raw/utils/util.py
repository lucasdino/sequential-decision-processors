import os, sys, math, random, json
from datetime import datetime
from typing import Callable, Optional, Tuple, Dict, Any, List
from contextlib import contextmanager
from collections import deque

from .dataclass import SamplingArgs


# =======================================
# Helper to limit unwanted print statements
# =======================================
@contextmanager
def silent_io():
    devnull = open(os.devnull, "w")
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = devnull
        yield
    finally:
        sys.stdout, sys.stderr = _out, _err
        devnull.close()


# =======================================
# Sampling Manager
# =======================================
class Rollout():
    def __init__(self, rollout_length: int, gen_args: Dict, move_idx: int):
        self.rollout_length = rollout_length
        self.gen_args = gen_args
        self.move_idx = move_idx
        self.rollout = []
        self.full = False

    def append_data(self, data):
        self.rollout.extend(data)
        if len(self.rollout) >= self.rollout_length+1:
            self.full = True

    def get_final_data(self):
        sample = {
            "data": self.rollout[:self.rollout_length+1],
            "info": {
                "move_idx": self.move_idx,
                "generation_args": self.gen_args
            }
        }
        return sample


class Textworld_Sampling_Manager():
    def __init__(self, wrapper, sampling_args: List[SamplingArgs]):
        """ Sampling manager that sits around our wrapper and generates samples. """
        self.wrapper = wrapper
        self.sampling_args = sampling_args


    def generate_data(self, save_path, max_iters = None, print_every = 10):
        self.generated_samples = [[] for _ in range(len(self.sampling_args))]
        self.still_generating = [True] * len(self.sampling_args)        

        iters = 0
        while any(self.still_generating) and (max_iters is None or iters < max_iters):
            iters += 1
            self._generate_samples()
            # Print updates
            if iters % print_every == 0:
                counts = " | ".join(f"{a.name}: {len(self.generated_samples[i])}/{a.total_samples}" for i, a in enumerate(self.sampling_args))
                print(f"[iter {iters}] {counts} | active={sum(self.still_generating)}")

        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each set of generated samples to CSV
        for arg_idx, args in enumerate(self.sampling_args):
            num_samples = min(len(self.generated_samples[arg_idx]), args.total_samples)
            file_name = f"{args.name}_{timestamp}_{num_samples}.jsonl"
            file_path = os.path.join(save_path, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                for sample in self.generated_samples[arg_idx][:num_samples]:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # ====================================================
    # Data generation helpers
    # ====================================================
    def _generate_samples(self) -> None:
        """ Main data sampling function. Generates data samples for each sampling_arg """
        # Start by resetting the environment
        self.wrapper.generate_new_game(randomize_gen_args=True)
        self.wrapper.load_env()

        # Then loop through our sampling_args
        for arg_idx, args in enumerate(self.sampling_args):
            # Don't generate data if we've reached our total number of samples
            if self.still_generating[arg_idx] == False:
                continue

            self.wrapper.reset_env()
            sampled_indices = self._get_sample_indices(self.wrapper.game_state.full_gold_path, args)
            
            # Generate our data
            rollouts, next_sample_idx, remaining_samples = [], sampled_indices.popleft(), args.total_samples
            for move_idx in range(len(self.wrapper.game_state.full_gold_path)):
                # Break if we've generated our desired # of samples
                if remaining_samples == 0:
                    break
                
                # Check for index matches and create new trace
                if next_sample_idx == move_idx:
                    r = Rollout(rollout_length=args.rollout_length, gen_args=self.wrapper.game_state.generation_args, move_idx=move_idx)
                    r.append_data(self.wrapper.get_env_state(full_state=True))
                    rollouts.append(r)
                    next_sample_idx = sampled_indices.popleft() if len(sampled_indices) > 0 else -1
                
                # Step env and get intermediate data
                _ = self.wrapper.step_env()
                usr, env = self.wrapper.get_env_state(full_state=False)
                
                # Update traces
                idx = 0
                while idx < len(rollouts):
                    rollouts[idx].append_data([usr, env])
                    if rollouts[idx].full:
                        r = rollouts.pop(idx)
                        remaining_samples -= 1
                        self.generated_samples[arg_idx].append(r.get_final_data())
                        if remaining_samples == 0:
                            break
                    else:
                        idx += 1

            # Once done, catch any partial rollouts since we want these
            final_rollouts = [rollout.get_final_data() for rollout in rollouts]
            self.generated_samples.extend(final_rollouts[:remaining_samples])
            if len(self.generated_samples[arg_idx]) >= args.total_samples:
                self.still_generating[arg_idx] = False


    def _get_sample_indices(self, gold_path: List[str], sampling_args) -> deque[int]:
        """
        Uniform over all ascending start sets with gap >= d (= L - overlap), O(n) time.
        """
        actions_per_rollout = math.ceil(sampling_args.rollout_length / 2)
        L = int(actions_per_rollout)
        O = int(sampling_args.max_step_overlap)
        d = max(1, L - O)

        max_start = len(gold_path) - L + 1
        if max_start <= 0:
            return deque()

        # Max number of starts that fit with gap d (tight pack)
        capacity = 1 + (max_start - 1) // d
        n = min(int(sampling_args.max_samples_per_env), capacity)
        if n <= 0:
            return deque()

        # Slack domain size after removing mandatory gaps
        S = max_start - d * (n - 1)
        if S <= 0:
            # No slack: only one packing (0, d, 2d, ...)
            xs = [i * d for i in range(n) if i * d < max_start]
            return deque(xs)

        # ---- Uniform multiset sampling (combinations with replacement) ----
        # Pick z_1 < ... < z_n from {0, 1, ..., S+n-2}, then set y_i = z_i - i
        # so 0 <= y_1 <= ... <= y_n < S.
        z = sorted(random.sample(range(S + n - 1), n))
        ys = [z_i - i for i, z_i in enumerate(z)]

        # Map back to original starts with gaps reinstated
        xs = [y + i * d for i, y in enumerate(ys)]

        # Sanity checks
        assert all(0 <= x < max_start for x in xs)
        assert all(xs[i+1] - xs[i] >= d for i in range(len(xs) - 1))

        return deque(xs)
    

# =======================================
# Various other helpers
# =======================================
def print_env_state(env_state: List[Tuple[str, str]]):
    for role, output in env_state:
        print(f"{role}: {output}\n\n")