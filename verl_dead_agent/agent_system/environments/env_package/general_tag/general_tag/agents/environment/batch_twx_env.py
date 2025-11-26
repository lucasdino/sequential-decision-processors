# Textworld express only supports single instance envs. So we wrap all of the envs in a 'fake' batch env.
import textworld_express as twx
import gymnasium as gym
import numpy as np
from .download_game_files import get_seeds_twx, TEXTWORLD_EXPRESS_TASKS


COOKINGWORLD_VERBS = ["chop _", "cook _ in _", "dice _", "drop _", "examine _", "go _", "open _", "slice _ ", "take _", "prepare meal", "eat meal", "look around", "inventory"]

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

    def reset(self, seed, *, options=None):
        obs, info = self.env.reset(
            seed=seed,
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
        info["verbs"] = COOKINGWORLD_VERBS
        return (obs,), info

    def step(self, action):
        act = "look around" if action[0] == "look" else action[0]        
        obs, reward, done, info = self.env.step(action[0])
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = info["tasksuccess"]
        info["lost"] = info["taskfailure"]
        info["moves"] = info["numMoves"]
        info["score"] = int(info["score"] * 100)
        info["admissible_commands"] = info["validActions"]
        info["verbs"] = COOKINGWORLD_VERBS
        return (obs,), (reward,), (done,), info

    def close(self):
        self.env.close()

class TWXBatchGym(gym.Env):
    # This basically just generates a bunch of TextWorldExpressEnv instances and steps through them in sequence.
    def __init__(
        self, tasks, split = "train", max_steps = 100, *args, **kwargs
    ):
        self.seeds = tasks
        self.tasks = TEXTWORLD_EXPRESS_TASKS
        self.task = TEXTWORLD_EXPRESS_TASKS[0]   # Only allowing first elem to be our task
        assert self.task[1] == "cookingworld"    # Otherwise need to adjust the 'verbs' above
        self.env = TextWorldExpressEnv(self.task[1], self.task[2], split=split, max_steps=max_steps)

    def reset(self, seed, *, options=None):
        return self.env.reset(seed=seed, options=options)
        
    def step(self, action):
        return self.env.step(action)
        
    def close(self):
        for env in self.envs:
            env.close()