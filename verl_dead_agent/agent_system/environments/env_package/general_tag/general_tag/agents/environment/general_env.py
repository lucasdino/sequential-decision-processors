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
from .batch_alfworld_env import AlfworldResetEnv
from .batch_twx_env import TWXBatchGym

TASK_TYPES = {}


class GeneralTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, train_eval="train", main_config=None):
        # print("Initializing GeneralTWEnv...")
        self.config = config
        self.main_config = main_config
        self.train_eval = train_eval
        # print("Train eval value:", train_eval)
        # print("Main config:", self.main_config)
        
        if 'tales' in self.main_config['env']['env_name']:
            framework_name = self.main_config['env']['env_name'].split('_')[-1]
            use_valid_seen = self.main_config['env']['valid_seen']
            load_env_seeds = self.main_config['env']['load_env_seeds']
            rand_seed = self.main_config['env']['seed']
            rng = random.Random(rand_seed)
            
            SAVE_ENV_SEEDS = False   # Toggle in case we want to save env seeds
            
            if framework_name == 'alfworld':
                # Load in our various types of games we could be using
                train = sorted(get_alfworld_games(max_num_per_task=200, skip = ["valid"]))
                valid_seen = sorted(get_alfworld_games(max_num_per_task=25, skip = ["train", "valid_unseen"]))
                valid_unseen = sorted(get_alfworld_games(max_num_per_task=25, skip = ["train", "valid_seen"]))
                rng.shuffle(train); rng.shuffle(valid_seen); rng.shuffle(valid_unseen)
                
                # Optionally can save
                if SAVE_ENV_SEEDS:
                    max_to_save = 128
                    self._save_validation_games("alfworld_valid_seen.txt", valid_seen[:max_to_save])
                    self._save_validation_games("alfworld_valid_unseen.txt", valid_unseen[:max_to_save])

                # Optionally can load as well
                if load_env_seeds:
                    valid_seen = self._load_validation_games("alfworld_valid_seen.txt")
                    valid_unseen = self._load_validation_games("alfworld_valid_unseen.txt")

                # Set / return our game files
                if train_eval == 'train':
                    self.game_files = train
                else:
                    self.game_files = valid_seen if use_valid_seen else valid_unseen
                print(colored(f"Using ALFWorld framework with {len(self.game_files)} games.", 'green'))
            
            # elif framework_name == 'textworld':
            #     train = sorted(get_cooking_game(split='train'))
            #     test = sorted(get_cooking_game(split='test'))
            #     print("printing train_eval to sanity check", train_eval)
            #     if train_eval == 'train':
            #         self.game_files = train
            #         for gfile in self.game_files:
            #             print(f"Training on game: {gfile}")
            #     else:
            #         self.game_files = test
            #         for gfile in self.game_files:
            #             print(f"Testing on game: {gfile}")
            #     print(colored(f"Using CookingWorld framework with {len(self.game_files)} games.", 'green'))
            
            elif framework_name == 'twx':
                train_seeds = get_seeds_twx('train')
                valid_seeds = get_seeds_twx('valid')
                random.seed(rand_seed)
                rng.shuffle(train_seeds); rng.shuffle(valid_seeds)

                # Optionally can save
                if SAVE_ENV_SEEDS:
                    max_to_save = 128
                    self._save_validation_games("twx_valid.txt", valid_seeds[:max_to_save])

                # Optionally can load in our seeds as well
                if load_env_seeds:
                    valid_seeds = self._load_validation_games("twx_valid.txt")
                    valid_seeds = [int(v) for v in valid_seeds]

                self.game_files = train_seeds if train_eval == "train" else valid_seeds
                print(colored(f"Using TWX framework with {len(self.game_files)} games.", 'green'))
            
            else:
                self.collect_game_files(self.config['framework'])
        else:
            raise ValueError("No framework specified in config. Please specify a framework.")
        self.use_expert = False
        # print(f"use_expert = {self.use_expert}")

    def init_n_env(self, game_files):
        request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, verbs=True, extras=["gamefile"], moves=True)
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
                request_infos = textworld.EnvInfos(
                    won=True, lost=True, admissible_commands=True, verbs=True,
                    extras=["gamefile", "walkthrough"]
                )
                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]

                env = AlfworldResetEnv(
                    self.game_files,
                    request_infos,
                    max_nb_steps_per_episode,
                    wrappers=wrappers,
                    batch_size=1,
                    asynchronous=True,
                )

                # wrappers = [AlfredDemangler()]
                # request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, verbs=True, extras=["gamefile", "walkthrough"])

                # max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
                
                # env_id = textworld.gym.register_games(self.game_files, request_infos,
                #                                     batch_size=1,
                #                                     asynchronous=True,
                #                                     max_episode_steps=max_nb_steps_per_episode,
                #                                     wrappers=wrappers)
                # # Launch Gym environment.
                # env = textworld.gym.make(env_id)
            elif framework_name == 'textworld':
                # Disable moves bc it seems to be erroring out for some reason.
                request_infos = textworld.EnvInfos(won=True, lost=True, admissible_commands=True, verbs=True, extras=["gamefile", "walkthrough"], moves=False)


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


    # ========================================
    # Save and load valid seeds helper
    # ========================================
    def _save_validation_games(self, filename, game_files):
        validation_dir = os.path.join(os.path.dirname(__file__), "validation_seeds")
        os.makedirs(validation_dir, exist_ok=True)
        file_path = os.path.join(validation_dir, filename)
        
        # Ensure all seeds are unique
        gfiles_set = set(game_files)
        assert len(gfiles_set) == len(game_files)

        try:
            with open(file_path, "w") as f:
                for game in game_files:
                    f.write(f"{game}\n")
            print(colored(f"Saved {len(game_files)} validation game files to {file_path}.", 'cyan'))
        except OSError as exc:
            print(colored(f"Failed to save validation games to {file_path}: {exc}", 'red'))

    def _load_validation_games(self, filename):
        validation_dir = os.path.join(os.path.dirname(__file__), "validation_seeds")
        file_path = os.path.join(validation_dir, filename)

        if not os.path.exists(file_path):
            print(colored(f"Validation seed file not found: {file_path}", 'yellow'))
            return []

        try:
            with open(file_path, "r") as f:
                seeds = [line.strip() for line in f if line.strip()]
            print(colored(f"Loaded {len(seeds)} validation seeds from {file_path}.", 'cyan'))
            return seeds
        except OSError as exc:
            print(colored(f"Failed to load validation seeds from {file_path}: {exc}", 'red'))
            return []