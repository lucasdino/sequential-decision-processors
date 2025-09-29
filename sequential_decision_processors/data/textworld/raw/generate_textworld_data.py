import os
from . import utils as util




def main():
    # Define our inputs
    num_gens = 5
    save_path = f"{os.getcwd()}/generated_data"
    sampling_args = [
        util.SamplingArgs(name="cooking_singlemove", max_samples_per_env=20, rollout_length=1, total_samples=25_000, max_step_overlap=0),
        util.SamplingArgs(name="cooking_multimove", max_samples_per_env=8, rollout_length=5, total_samples=10_000, max_step_overlap=1),
    ]
    cooking_wrapper = util.Textworld_Cooking_Wrapper_Env()

    # Generate our data
    data_sampler = util.Textworld_Sampling_Manager(
        wrapper = cooking_wrapper,
        sampling_args=sampling_args
    )

    for i in range(num_gens):
        data_sampler.generate_data(save_path=save_path, max_iters=None)


if __name__ == "__main__":
    main()