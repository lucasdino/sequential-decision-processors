# Textworld express only supports single instance envs. So we wrap all of the envs in a 'fake' batch env.
import textworld_express as twx
import gymnasium as gym
import numpy as np
from .download_game_files import get_seeds_twx, TEXTWORLD_EXPRESS_TASKS

# Base twx class, taken from TALES.
class TextWorldExpressEnv(gym.Env):

    def __init__(
        self, game_name, game_params, split="train", max_steps = 100, admissible_commands=False, *args, **kwargs
    ):
        self.game_name = game_name
        self.game_params = game_params
        self.admissible_commands = admissible_commands
        self.env = twx.TextWorldExpressEnv(envStepLimit=max_steps)
        self.split = split
        self.seeds = get_seeds_twx(split=split, env=self.env)
        self.seed = self.seeds[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = self.seeds[seed % len(self.seeds)]

        obs, info = self.env.reset(
            seed=self.seed,
            gameFold=self.split,
            gameName=self.game_name,
            gameParams=self.game_params,
            generateGoldPath=True,
        )

        # Add task description to the first observation.
        obs = info["taskDescription"] + "\n\n" + obs

        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = False
        info["lost"] = False
        info["moves"] = 0
        info["score"] = int(info["score"] * 100)
        info["admissible_commands"] = info["validActions"]
        info["extra.walkthrough"] = self.env.getGoldActionSequence()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = info["tasksuccess"]
        info["lost"] = info["taskfailure"]
        info["moves"] = info["numMoves"]
        info["score"] = int(info["score"] * 100)
        info["admissible_commands"] = info["validActions"]
        return obs, reward, done, info

    def close(self):
        self.env.close()

class TWXBatchGym(gym.Env):
    # This basically just generates a bunch of TextWorldExpressEnv instances and steps through them in sequence.
    def __init__(
        self, tasks, split = "train", max_steps = 100, *args, **kwargs
    ):
        
        self.seed = tasks
        self.tasks = TEXTWORLD_EXPRESS_TASKS
        self.envs = []
        print("Seeds:", self.seed)
        for task in self.tasks:
            print("Task:", task)
            env = TextWorldExpressEnv(task[1], task[2], split=split, max_steps=max_steps)
            print(f'Creating {split} TWX env for game: {task[0]} with params: {task[2]}')
            self.envs.append(env)
            

    def reset(self, *, seed=None, options=None):
        for i, env in enumerate(self.envs):
            env.reset(seed=self.seed[i], options=options)

    def step(self, action):
        # Gather all of the obs, reward, done and info into arrays.
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []
        for env in self.envs:
            obs, reward, done, info = env.step(action)
            all_obs.append(obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_infos.append(info)
        return obs, reward, done, info

    def close(self):
        for env in self.envs:
            env.close()