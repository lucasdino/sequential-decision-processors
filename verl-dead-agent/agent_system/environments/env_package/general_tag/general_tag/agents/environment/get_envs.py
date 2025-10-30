

import glob
import os
from .download_game_files import prepare_alfworld_data, prepare_twcooking_data
from os.path import join as pjoin
from .download_game_files import TASK_TYPES, TALES_CACHE_ALFWORLD_VALID_SEEN, TALES_CACHE_ALFWORLD_VALID_UNSEEN, TALES_CACHE_TWCOOKING_TEST, TALES_CACHE_TWCOOKING_TRAIN


def get_cooking_game(split="train", difficulties=[1, 2, 3, 4, 5], one_game_per_difficulty=True):
    prepare_twcooking_data()  # make sure the data is ready
    all_game_files = []
    if split == "train":
        cooking_dir_base = TALES_CACHE_TWCOOKING_TRAIN
    elif split == "test":
        cooking_dir_base = TALES_CACHE_TWCOOKING_TEST
    else:
        raise ValueError("split must be either 'train' or 'test'")
    
    for difficulty in difficulties:
        cooking_dir = pjoin(cooking_dir_base, f"difficulty_level_{difficulty}")
        print(cooking_dir)
        game_files = glob.glob(pjoin(cooking_dir, "*.z8"))
        if one_game_per_difficulty and game_files:
            game_files = [game_files[0]]
        all_game_files += game_files
    return all_game_files


def get_alfworld_games(max_num_per_task = 1, skip = []):
    prepare_alfworld_data()  # make sure the data is ready
    all_game_files = []
    for task in TASK_TYPES:
        game_files_seen = sorted(glob.glob(pjoin(TALES_CACHE_ALFWORLD_VALID_SEEN, f"{task}*", "**", "*.tw-pddl")))
        game_files_unseen = sorted(glob.glob(pjoin(TALES_CACHE_ALFWORLD_VALID_UNSEEN, f"{task}*", "**", "*.tw-pddl")))
        if skip:
            game_files_seen = [f for f in game_files_seen if not any(s in f for s in skip)]
            game_files_unseen = [f for f in game_files_unseen if not any(s in f for s in skip)]
        all_game_files.extend(game_files_seen[:max_num_per_task])
        all_game_files.extend(game_files_unseen[:max_num_per_task])

    return all_game_files