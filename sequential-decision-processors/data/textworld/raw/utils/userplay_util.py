import re, json
import shutil, subprocess

import textworld.gym

from pathlib import Path


# ######################################################
# #########     Textworld Cooking Helpers      #########
# ######################################################

# -------
# Observation Extraction Helpers
# -------
def _trim_ascii_cooking(text: str) -> str:
    """
    Simple function that trims the Text World ASCII art from the initial observation
    """
    m = re.search(r"\${6,}(?![\s\S]*\${6,})([\s\S]*)\Z", text)
    if not m:
        return text
    out = m.group(1)
    out = re.sub(r"^[ \t\r\n]+", "", out)
    return out

def clean_obs_cooking(text: str) -> str:
    """
    Simple function to clean / trim the observation
    """
    text = _trim_ascii_cooking(text)
    m = re.search(r"^(.*)>(?!.*>)", text, re.DOTALL)
    if not m:
        return text.strip()
    return m.group(1).rstrip(" \t\r\n")

# -------
# Game creation helpers
# -------
def instantiate_textworld_cooking_game(
    filename="custom_cooking",
    folder="tw_games",
    recipe=5, take=3, go=1,
    open_=True, cook=True, cut=True, drop=True,
    split="train", recipe_seed=None, seed=None, fmt="z8",
):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)

    # nuke existing files with same basename (any extension)
    for p in list(folder.glob(filename)) + list(folder.glob(filename + ".*")):
        (p.unlink() if (p.is_file() or p.is_symlink()) else shutil.rmtree(p))

    out = folder / f"{filename}.{fmt}"
    cmd = [
        "tw-make", "tw-cooking",
        "--recipe", str(recipe), "--take", str(take), "--go", str(go),
        "--split", split, "--format", fmt, "--output", str(out),
    ]
    for flag, ok in [("--open", open_), ("--cook", cook), ("--cut", cut), ("--drop", drop)]:
        if ok: cmd.append(flag)
    if recipe_seed is not None: cmd += ["--recipe-seed", str(recipe_seed)]
    if seed is not None: cmd += ["--seed", str(seed)]

    subprocess.run(cmd, check=True)
    return str(out)

def load_textworld_cooking_game(
    filename="custom_cooking",
    folder="tw_games",
    request_infos=None,
    max_episode_steps=50
):
    env_id = textworld.gym.register_game(f"{folder}/{filename}.z8", request_infos=request_infos, max_episode_steps=max_episode_steps)
    env = textworld.gym.make(env_id)
    json_info = json.load(open(f"{folder}/{filename}.json"))
    env_info = {**json_info['metadata'], **{'objective': json_info['objective']}}
    return env, env_info