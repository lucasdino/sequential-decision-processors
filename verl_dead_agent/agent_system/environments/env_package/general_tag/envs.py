import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import ray

from agent_system.environments.env_package.general_tag.general_tag.agents.environment import get_environment


def load_config_file(path):
    print(path)
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def compute_reward(info, done):
    reward = 0
    if info['won']:
        reward = 10
    elif info['lost']:
        reward = -5
    else:
        # If the agent made some number of moves that are actually valid in the environment
        if 'moves' in info.keys():
            if info['moves'] > 0 and done:
                reward = 0.3 * info['moves']

    return float(reward)

@ray.remote(num_cpus=0.25)
class GeneralWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, env_file_name=None):
        if env_file_name is not None:
            self.env = base_env.init_n_env([env_file_name])
            print("Loading single env: ", env_file_name)
        else:
            self.env = base_env.init_env()  # Each worker holds only one sub-environment
        print("Fixing twx batch:", self.env)
        self.env.seed(seed)
        self.config = config
        self.seed = seed
    
    def step(self, action):
        """Execute a step in the environment"""
        if len(action) > 50:
            print("Action too long, truncated to 50 chars: ", action)
            action = action[:50]
        elif 'restart' in action:
            action = action.replace("restart", "")
        elif 'exit' in action:
            action = action.replace("exit", "")
        elif 'save' in action:
            action = action.replace("save", "")
        # Not sure why this action is crashing the game but try to deal with it:
        special_chars = ['\\', '/', '<', '>', '|', '*', '?', '"', '\'', '`', '$', '#', '&', ';', '(', ')', '{', '}', '[', ']', '%', '@', '+', '=', '-', ':', ',', '.', '!', '^', 'action']
        for char in special_chars:
            action = action.replace(char, "")
        try:
            # Test if it's valid UTF-8
            action_bytes = action.encode('utf-8')
            # print(" Action '{}' was truncated to '{}'.".format(action, action_bytes.decode()))
        except UnicodeEncodeError:
            # Convert to bytes first
            print("Encoded error tripped, original action: ", action)
            action = action.encode('latin-1')

        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        # print("Obs: ", obs)
        infos['observation_text'] = obs
        for i, done in enumerate(dones):
            # Sanity check: make sure that if the game is 'won' or 'lost', the done flag is set to True.
            if infos['won'][i] or infos['lost'][i]:
                assert done, "Game should be done if won or lost."
        # print("Step completed with action:", action)
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        # print("Env reset")
        # print(infos)
        # print("========================== Episode Reset ==========================")
        return obs, infos
    
    def getobs(self):
        return None

class GeneralEnvs(gym.Env):
    def __init__(self, general_config_path, seed=0, env_num=1, group_n=1, is_train=True, main_config = None, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(general_config_path)

        env_type = config['env']['type']
        self.main_config = main_config
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else 'test', main_config = main_config)
        
        self.multi_modal = False
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = GeneralWorker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(len(self.workers))]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, dones[0]))

        return text_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        print("Len of results from reset:", len(results))
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        return text_obs_list, info_list

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_general_envs(general_config_path, seed, env_num, group_n, is_train=True, main_config = None, env_kwargs={}):
    return GeneralEnvs(general_config_path, seed, env_num, group_n, is_train, main_config=main_config, env_kwargs=env_kwargs)