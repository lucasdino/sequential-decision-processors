import os
from . import utils as util


NUM_GENS = 3
MAX_SAMPLES_SINGLE = 25_000
MAX_SAMPLES_MULTI = 10_000
SAVE_PATH = f"{os.getcwd()}/sequential_decision_processors/data/textworld/raw/generated_data"


def generate_scienceworld_data(num_gens=1, max_samples_single=25, max_samples_multi=25):
    sampling_args = [
        util.SamplingArgs(name="scienceworld_singlemove", max_samples_per_env=20, rollout_length=1, total_samples=max_samples_single, max_step_overlap=0),
        util.SamplingArgs(name="scienceworld_multimove", max_samples_per_env=8, rollout_length=5, total_samples=max_samples_multi, max_step_overlap=1),
    ]
    scienceworld_wrapper = util.Scienceworld_Wrapper_Env()
    data_sampler = util.Textworld_Sampling_Manager(wrapper = scienceworld_wrapper, sampling_args=sampling_args)
    for i in range(num_gens):
        print(f"Generating Scienceworld Data -- Generation #{i+1}")
        try:
            data_sampler.generate_data(save_path=SAVE_PATH, max_iters=None)
        except Exception as e:
            if isinstance(e, util.NoFurtherGenerations):
                print(f"No further generations available.")
                break
            else:
                raise e

def generate_cooking_data(num_gens=1, max_samples_single=25, max_samples_multi=25):
    sampling_args = [
        util.SamplingArgs(name="cooking_singlemove", max_samples_per_env=20, rollout_length=1, total_samples=max_samples_single, max_step_overlap=0),
        util.SamplingArgs(name="cooking_multimove", max_samples_per_env=8, rollout_length=5, total_samples=max_samples_multi, max_step_overlap=1),
    ]
    cooking_wrapper = util.Textworld_Cooking_Wrapper_Env()
    data_sampler = util.Textworld_Sampling_Manager(wrapper = cooking_wrapper, sampling_args=sampling_args)
    for i in range(num_gens):
        print(f"Generating Cooking Data -- Generation #{i+1}")
        data_sampler.generate_data(save_path=SAVE_PATH, max_iters=None)


def main():
    # generate_scienceworld_data(num_gens=NUM_GENS, max_samples_single=MAX_SAMPLES_SINGLE, max_samples_multi=MAX_SAMPLES_MULTI)
    generate_cooking_data(num_gens=NUM_GENS, max_samples_single=MAX_SAMPLES_SINGLE, max_samples_multi=MAX_SAMPLES_MULTI)


if __name__ == "__main__":
    main()