import os, ast
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import ray

from agent_system.environments.env_package.general_tag.general_tag.agents.environment import get_environment
from agent_system.environments.env_package.general_tag.general_tag.agents.environment.utils import clean_cookingworld_obs, clean_alfworld_obs, get_necessary_context

ALFWORLD_VERBS = ['go to', 'open', 'close', 'take _ from _', 'move _ to _', 'use', 'heat _ with _', 'cool _ with _', 'clean _ with _', 'slice _ with _', 'inventory', 'look', 'examine']


def load_config_file(path):
    print(path)
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def compute_reward(info, done):
    reward = 0
    if info['env_provided']['won']:
        reward = 1
    elif info['env_provided']['lost']:
        reward = -0.5
    # else:
    #     # If the agent made some number of moves that are actually valid in the environment
    #     if 'moves' in info.keys():
    #         if info['moves'] > 0 and done:
    #             reward = 0.3 * info['moves']
    return float(reward)




@ray.remote(num_cpus=0.25)
class GeneralWorker:
    """ Ray remote actor that replaces the worker function. Each actor holds one environment instance. """
    def __init__(self, config, base_env, env_file_name=None):
        # Old code from chris - currently just going thru else branch
        if env_file_name is not None:
            self.env = base_env.init_n_env([env_file_name])
            print("Loading single env: ", env_file_name)
        else:
            self.env = base_env.init_env()  
        
        # General instantiation
        self.necessary_context = None
        self.my_seed = None
        self.cur_step = 1
        self.config = config
        self.env_type_string = base_env.main_config['env']['env_name'] if base_env.main_config else None
    
    def step(self, action):
        """Execute a step in the environment"""
        if len(action) > 100:
            print("Action too long, truncated to 100 chars: ", action)
            action = action[:100]
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
        
        # Test if it's valid UTF-8
        try:
            action_bytes = action.encode('utf-8')
            # print(" Action '{}' was truncated to '{}'.".format(action, action_bytes.decode()))
        except UnicodeEncodeError:
            print("Encoded error tripped, original action: ", action)
            action = action.encode('latin-1')

        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        obs = self._process_obs(obs)
        proc_infos = self._process_infos(infos)
        proc_infos['observation_text'] = obs[0]
        self.cur_step += 1
        return obs, scores, dones, proc_infos
    
    def reset(self, seed):
        """Reset the environment"""
        self.cur_step = 1
        self.my_seed = seed
        if self.env_type_string == "tales_alfworld":
            obs, infos = self.env.reset(game_file=seed)
        else:
            obs, infos = self.env.reset(seed)

        obs = self._process_obs(obs)
        infos = self._process_infos(infos)
        infos['observation_text'] = obs[0]
        return obs, infos
    
    def getobs(self):
        return None

    # ==========================
    # Updated to allow env specific edits
    # ==========================
    def _process_infos(self, infos):
        cleaned_infos = dict()
        if self.env_type_string and "textworld" in self.env_type_string:
            return infos
        elif self.env_type_string and "alfworld" in self.env_type_string:
            infos = {k: v[0] for k, v in infos.items()}   # need to do this to match other envs
            cleaned_infos['env_provided'] = infos
            cleaned_infos['state_info'] = {
                "verbs": ALFWORLD_VERBS,
                "necessary_context": self.necessary_context
            }
            cleaned_infos['run_info'] = {
                "seed": self.my_seed,
                "proc_id": os.getpid(),
                "step": self.cur_step
            }
            return cleaned_infos
        elif self.env_type_string and "scienceworld" in self.env_type_string:
            return infos
        elif self.env_type_string and "twx" in self.env_type_string:
            cleaned_infos['env_provided'] = infos
            cleaned_infos['state_info'] = {
                "verbs": infos['verbs'],
                "necessary_context": self.necessary_context
            }
            cleaned_infos['run_info'] = {
                "seed": self.my_seed,
                "proc_id": os.getpid(),
                "step": self.cur_step
            }
            return cleaned_infos
        else:
            raise ValueError(f"Please add the env to this process_infos function (even if just identity).")

    def _process_obs(self, obs):
        """ Functionality for env specific observation processing """
        if self.env_type_string and "textworld" in self.env_type_string:
            obs = clean_cookingworld_obs(obs[0])
            ctx = get_necessary_context(obs)
            self.necessary_context = ctx if ctx else self.necessary_context
            return (obs,)
        elif self.env_type_string and "alfworld" in self.env_type_string:
            return (clean_alfworld_obs(obs[0]),)
        elif self.env_type_string and "scienceworld" in self.env_type_string:
            return obs
        elif self.env_type_string and "twx" in self.env_type_string:
            obs = clean_cookingworld_obs(obs[0])
            ctx = get_necessary_context(obs)
            self.necessary_context = ctx if ctx else self.necessary_context
            return (obs,)
        else:
            raise ValueError(f"Please add the env to this process_obs function (even if just identity).")

    # For testing ray processes
    def ping(self):
        return f"Hi from worker with seed {self.my_seed}"



class GeneralEnvs(gym.Env):
    def __init__(self, general_config_path, seed=0, env_num=1, group_n=1, is_train=True, main_config = None, env_kwargs={}):
        """  Purpose of this is to be an env wrapper that manages the underlying workers. """
        super().__init__()
        
        # Start by initializing Ray (if not initialized)
        if not ray.is_initialized():
            ray.init()
        
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')

        config = load_config_file(general_config_path)
        self.env_type = config['env']['type']
        self.main_config = main_config

        # base_env is a 'GeneralTWEnv'
        base_env = get_environment(self.env_type)(config, train_eval='train' if is_train else 'test', main_config = main_config)

        # 'base_env.game_files' gives us our seeds we'll manage:
        #     - For twx this is a list of ints (seeds)
        #     - For alfworld this is a list of files (pddl)
        self.max_seed_idx = len(base_env.game_files)
        self.cur_seed_idx = 0
        self.seeds = base_env.game_files

        self.multi_modal = False
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = GeneralWorker.remote(config, base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(len(self.workers))]

    def step(self, actions):
        assert len(actions) == len(self.workers), \
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
            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)
            self.prev_admissible_commands[i] = info['env_provided']['admissible_commands']
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
        for idx, worker in enumerate(self.workers):
            if int(self.cur_seed_idx) == self.max_seed_idx:
                for worker in self.workers[idx:]:
                    ray.kill(worker)
                self.workers = self.workers[:idx]
                break  # kill remaining ray workers (finished seeds) and break
            
            future = worker.reset.remote(self.seeds[int(self.cur_seed_idx)])
            self.cur_seed_idx = self.cur_seed_idx + 1/self.group_n
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        # print("Len of results from reset:", len(results))
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['env_provided']['admissible_commands']
            info_list.append(info)

        # self.ping_workers()
        return text_obs_list, info_list

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    # =====================
    # Ray helpers / debug
    # =====================
    def ping_workers(self):
        msgs = ray.get([w.ping.remote() for w in self.workers])
        print(msgs)

    def close(self):
        for worker in self.workers:
            ray.kill(worker)


def build_general_envs(general_config_path, seed, env_num, group_n, is_train=True, main_config = None, env_kwargs={}):
    return GeneralEnvs(general_config_path, seed, env_num, group_n, is_train, main_config=main_config, env_kwargs=env_kwargs)