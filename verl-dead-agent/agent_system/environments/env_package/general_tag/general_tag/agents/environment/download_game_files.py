import glob
import os
import zipfile
import shutil
import tempfile
import numpy as np
import requests
from os.path import join as pjoin
from tqdm import tqdm
from .utils import mkdirs, download

# Replicating how TALES does this

############### TEXTWORLD ################

DEFAULT_TALES_CACHE_HOME = os.path.expanduser("~/.cache/tales")
TALES_CACHE_HOME = os.getenv("TALES_CACHE_HOME", DEFAULT_TALES_CACHE_HOME)
os.environ["TALES_CACHE_HOME"] = (
    TALES_CACHE_HOME  # Set the environment variable, in case it wasn't.
)
os.makedirs(TALES_CACHE_HOME, exist_ok=True)

def prepare_twcooking_data(force=False):
    os.makedirs(TALES_CACHE_TWCOOKING, exist_ok=True)
    if os.path.exists(TALES_CACHE_TWCOOKING_TEST) and not force:
        return

    zip_file = pjoin(TALES_CACHE_TWCOOKING, "rl.0.2.zip")
    if not os.path.exists(zip_file) or force:
        download(
            TW_COOKING_URL,
            dst=TALES_CACHE_TWCOOKING,
            desc="Downloading TWCooking",
            force=force,
        )

    # Extract the content of the folder test from the downloaded file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Only extract the test folder
        for member in zip_ref.namelist():
            if "test" in member:
                zip_ref.extract(member, TALES_CACHE_TWCOOKING)
            elif "train" in member:
                zip_ref.extract(member, TALES_CACHE_TWCOOKING)

TW_COOKING_URL = (
    "https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/rl.0.2.zip"
)
TALES_CACHE_TEXTWORLD = pjoin(TALES_CACHE_HOME, "textworld")
TALES_CACHE_TWCOOKING = pjoin(TALES_CACHE_TEXTWORLD, "tw-cooking")
TALES_CACHE_TWCOOKING_TEST = pjoin(TALES_CACHE_TWCOOKING, "test")
TALES_CACHE_TWCOOKING_TRAIN = pjoin(TALES_CACHE_TWCOOKING, "train_1")

############### TEXTWORLD_EXPRESS ################

import textworld_express as twx

TEXTWORLD_EXPRESS_TASKS = [
    (
        "CookingWorld",
        "cookingworld",
        "numLocations=1, numIngredients=2, numDistractorItems=5, includeDoors=0, limitInventorySize=0",
    ),
    (
        "TextWorldCommonsense",
        "twc",
        "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0",
    ),
    (
        "CoinCollector",
        "coin",
        "numLocations=1, numDistractorItems=5, limitInventorySize=0",
    ),
    ("Arithmetic", "arithmetic", ""),
    (
        "MapReader",
        "mapreader",
        "numLocations=2, maxDistanceApart=1, maxDistractorItemsPerLocation=2, includeDoors=0, limitInventorySize=0",
    ),
    ("Sorting", "sorting", ""),
    ("SimonSays10", "simonsays", "gameLength=10, numDistractors=4, memorization=0"),
    ("SimonSays50", "simonsays", "gameLength=50, numDistractors=4, memorization=0"),
    ("SimonSays100", "simonsays", "gameLength=100, numDistractors=4, memorization=0"),
    (
        "SimonSaysWithMemory10",
        "simonsays",
        "gameLength=10, numDistractors=4, memorization=1, verbose=0",
    ),
    (
        "SimonSaysWithMemory50",
        "simonsays",
        "gameLength=50, numDistractors=4, memorization=1, verbose=0",
    ),
    (
        "SimonSaysWithMemory100",
        "simonsays",
        "gameLength=100, numDistractors=4, memorization=1, verbose=0",
    ),
    (
        "SimonSaysWithMemory10Verbose",
        "simonsays",
        "gameLength=10, numDistractors=4, memorization=1, verbose=1",
    ),
    (
        "SimonSaysWithMemory50Verbose",
        "simonsays",
        "gameLength=50, numDistractors=4, memorization=1, verbose=1",
    ),
    (
        "SimonSaysWithMemory100Verbose",
        "simonsays",
        "gameLength=100, numDistractors=4, memorization=1, verbose=1",
    ),
    ("PeckingOrder", "peckingorder", ""),
]


def get_seeds_twx(split, env=None):
    env = env or twx.TextWorldExpressEnv()
    if split == "train":
        return env.getValidSeedsTrain()
    elif split == "valid":
        return env.getValidSeedsDev()
    elif split == "test":
        return env.getValidSeedsTest()
    else:
        raise NotImplementedError("Only plan to support train, dev, and test splits.")

################ ALFWORLD ################

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

ALFWORLD_DATA_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
TALES_CACHE_ALFWORLD = pjoin(TALES_CACHE_HOME, "alfworld")
TALES_CACHE_ALFWORLD_DATA_ZIP = pjoin(TALES_CACHE_ALFWORLD, "json_2.1.3_tw-pddl.zip")
TALES_CACHE_ALFWORLD_VALID_SEEN = pjoin(
    TALES_CACHE_ALFWORLD, "json_2.1.1", "valid_seen"
)
TALES_CACHE_ALFWORLD_VALID_UNSEEN = pjoin(
    TALES_CACHE_ALFWORLD, "json_2.1.1", "valid_unseen"
)


def prepare_alfworld_data(force=False):
    os.makedirs(TALES_CACHE_ALFWORLD, exist_ok=True)
    data_exists = os.path.exists(TALES_CACHE_ALFWORLD_VALID_SEEN) and os.path.exists(
        TALES_CACHE_ALFWORLD_VALID_UNSEEN
    )
    if data_exists and not force:
        return

    if not os.path.exists(TALES_CACHE_ALFWORLD_DATA_ZIP) or force:
        download(
            ALFWORLD_DATA_URL,
            dst=TALES_CACHE_ALFWORLD,
            desc="Downloading ALFWorld data",
            force=force,
        )

    # Extract the content of the folder test from the downloaded file
    with zipfile.ZipFile(TALES_CACHE_ALFWORLD_DATA_ZIP, "r") as zip_ref:
        # Only extract the test folder
        for member in zip_ref.namelist():
            if "valid_seen" in member or "valid_unseen" in member:
                zip_ref.extract(member, TALES_CACHE_ALFWORLD)


