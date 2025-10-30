import glob
import os
import zipfile
from os.path import join as pjoin
from .utils import mkdirs, download

DEFAULT_TALES_CACHE_HOME = os.path.expanduser("~/.cache/tales")
TALES_CACHE_HOME = os.getenv("TALES_CACHE_HOME", DEFAULT_TALES_CACHE_HOME)
os.environ["TALES_CACHE_HOME"] = (
    TALES_CACHE_HOME  # Set the environment variable, in case it wasn't.
)
os.makedirs(TALES_CACHE_HOME, exist_ok=True)

TW_COOKING_URL = (
    "https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/rl.0.2.zip"
)
TALES_CACHE_TEXTWORLD = pjoin(TALES_CACHE_HOME, "textworld")
TALES_CACHE_TWCOOKING = pjoin(TALES_CACHE_TEXTWORLD, "tw-cooking")
TALES_CACHE_TWCOOKING_TEST = pjoin(TALES_CACHE_TWCOOKING, "test")
TALES_CACHE_TWCOOKING_TRAIN = pjoin(TALES_CACHE_TWCOOKING, "train_1")


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


def get_cooking_game(split="train", difficulties=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], one_game_per_difficulty=True):
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