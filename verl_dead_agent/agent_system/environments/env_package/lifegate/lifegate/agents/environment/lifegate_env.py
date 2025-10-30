import os
import json
import random

from tqdm import tqdm
from termcolor import colored

# Based off Alfworld implementation since it is texctworld underneath. Stripping down as much stuff as possible to get things running.

import textworld
import textworld.agents
import textworld.gym

TASK_TYPES = {}


class LifegateTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, train_eval="train", main_config=None):
        print("Initializing LifegateTWEnv...")
        self.config = config
        self.main_config = main_config
        self.train_eval = train_eval

        self.collect_game_files()
        self.use_expert = False
        print(f"use_expert = {self.use_expert}")

    def collect_game_files(self, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        # Get the directory where all of the lifegate game files are stored
        target_dir = self.main_config['env']['env_dir']

        # Automatically detect if a train/test split is present.
        has_split = all(os.path.isdir(os.path.join(target_dir, folder)) for folder in ['train', 'test'])

        if has_split:
            if self.train_eval == "train":
                target_dir = os.path.join(target_dir, 'train')
            elif self.train_eval == "test":
                target_dir = os.path.join(target_dir, 'test')
            else:
                # Just putting this here for completeness.
                target_dir = target_dir

        # Now iterate through all the files in the directory and extract the lifegate ulx files.
        for file in os.listdir(target_dir):
            print(file)
            if file.endswith('.z8') and "lifegate" in file.lower():
                self.game_files.append(os.path.join(target_dir, file))

        print(f"Overall we have {len(self.game_files)} games in split={self.train_eval}")
        for game in self.game_files:
            if self.train_eval == "train":
                print(f"Training game: {game}")
            elif self.train_eval == "test":
                print(f"Testing game: {game}")
        self.num_games = len(self.game_files)

        # Add each game to tasks so we can track performance
        task_counter = 1
        for game_name in self.game_files:
            TASK_TYPES[task_counter] = game_name
            task_counter += 1


        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)
            print("Evaluating with %d games" % (len(self.game_files)))

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
        request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, extras=["gamefile"], moves=True)

        max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        
        env_id = textworld.gym.register_games(self.game_files, request_infos,
                                              batch_size=1,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)

        return env

