import os
import json
import random

from tqdm import tqdm
from termcolor import colored

# Based off Alfworld implementation since it is textworld underneath. Stripping down as much stuff as possible to get things running.

import textworld
import textworld.agents
import textworld.gym
from .download_game_files import get_seeds_twx
from .get_envs import get_alfworld_games, get_cooking_game
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
from textworld.envs.wrappers import Filter
from .batch_twx_env import TWXBatchGym

TASK_TYPES = {}


class GeneralTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, train_eval="train", main_config=None):
        print("Initializing GeneralTWEnv...")
        self.config = config
        self.main_config = main_config
        self.train_eval = train_eval
        print("Train eval value:", train_eval)
        print("Main config:", self.main_config)
        
        if 'tales' in self.main_config['env']['env_name']:
            framework_name = self.main_config['env']['env_name'].split('_')[-1]
            if framework_name == 'alfworld':
                # Do this programatically the same way so for the sake of consistency
                train = sorted(get_alfworld_games(max_num_per_task=1, skip = []))
                test = sorted(get_alfworld_games(max_num_per_task=1, skip = train))
                if train_eval == 'train':
                    self.game_files = train
                else:
                    self.game_files = test
                print(colored(f"Using ALFWorld framework with {len(self.game_files)} games.", 'green'))
                if self.train_eval == 'train':
                    print("Training on games:")
                    for game in self.game_files:
                        print(f"Game: {game}")
                else:
                    print("Testing on games:")
                    for game in self.game_files:
                        print(f"Game: {game}")
            elif framework_name == 'textworld':
                train = sorted(get_cooking_game(split='train'))
                test = sorted(get_cooking_game(split='test'))
                print("printing train_eval to sanity check", train_eval)
                if train_eval == 'train':
                    self.game_files = train
                    for gfile in self.game_files:
                        print(f"Training on game: {gfile}")
                else:
                    self.game_files = test
                    for gfile in self.game_files:
                        print(f"Testing on game: {gfile}")
                print(colored(f"Using CookingWorld framework with {len(self.game_files)} games.", 'green'))
            elif framework_name == 'twx':
                # Need to do this a bit differently for textworld express since they dont follow the typical batch_env setup
                print("Using TextWorld Express framework.")
                if train_eval == "train":
                    print("Textworld Express training")
                    self.game_files = get_seeds_twx('train')
                else:
                    print("Textworld Express training")
                    self.game_files = get_seeds_twx('valid')
            else:
                self.collect_game_files(self.config['framework'])
        else:
            raise ValueError("No framework specified in config. Please specify a framework.")
        self.use_expert = False
        print(f"use_expert = {self.use_expert}")

    def init_n_env(self, game_files):
        request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, extras=["gamefile"], moves=True)

        max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]

        env_id = textworld.gym.register_games(game_files, request_infos,
                                              batch_size=1,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)

        return env

    def init_env(self):
        # Register a new Gym environment.
        if 'tales' in self.main_config['env']['env_name']:
            framework_name = self.main_config['env']['env_name'].split('_')[-1]
            if framework_name == 'alfworld':
                wrappers = [AlfredDemangler()]
                request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, extras=["gamefile"])

                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
                
                env_id = textworld.gym.register_games(self.game_files, request_infos,
                                                    batch_size=1,
                                                    asynchronous=True,
                                                    max_episode_steps=max_nb_steps_per_episode,
                                                    wrappers=wrappers)
                # Launch Gym environment.
                env = textworld.gym.make(env_id)
            elif framework_name == 'textworld':
                # Disable moves bc it seems to be erroring out for some reason.
                request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, extras=["gamefile"], moves=False)


                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
                
                env_id = textworld.gym.register_games(self.game_files, request_infos,
                                                    batch_size=1,
                                                    asynchronous=True,
                                                    max_episode_steps=max_nb_steps_per_episode)
                # Launch Gym environment.
                env = textworld.gym.make(env_id)
            elif framework_name == 'twx':
                # textworld express only supports single instance envs. So we wrap all of the envs in a 'fake' batch env.
                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]

                env = TWXBatchGym(self.game_files, self.train_eval, max_nb_steps_per_episode)

            return env

