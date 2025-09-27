import re, json
import subprocess, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

import textworld.gym
from textworld import EnvInfos

from .dataclass import GameState
from .util import silent_io

# ######################################################
# #########     Textworld Cooking Wrapper      #########
# ######################################################
class Textworld_Cooking_Wrapper_Env():
    ENV_FILENAME = "custom_cooking"
    # ENV_FOLDER = "sequential-decision-processors/data/textworld/tw_games"
    ENV_FOLDER = "tw_games"

    INFO_REQUESTS = EnvInfos(
        objective=True,
        verbs=True,
        location=True,
        description=True,
        inventory=True,
        policy_commands=True,
        max_score=True,
        moves=True,
        last_command=True
    )

    def __init__(self):
        """
        Wrapper used to get standard input/output for the cooking task.
        """
        self.game_state = GameState()
    
    # =======================================
    # Env handling
    # =======================================
    def reset_env(self):
        with silent_io():
            env_response, env_state = self.env.reset()
        self._update_metadata(env_state=env_state, initial_state=True)
        

    def step_env(self, action=None):
        with silent_io():
            if action:
                env_response, score, done, env_state = self.env.step(action)
            else:
                env_response, score, done, env_state = self.env.step(self.game_state.cur_gold_path.popleft())
            
        # Update metadata
        self.game_state.done = done
        self.game_state.cur_score = score
        self._update_metadata(env_state=env_state, env_response=env_response, initial_state=False)
        return done


    def get_env_state(self, full_state=True) -> str:
        """ External function for getting a structured print of our current environment. """
        if full_state:
            return self.game_state.get_full_state()
        else:
            return self.game_state.get_intermediate_state()

    # =======================================
    # Game creation / loading
    # =======================================
    def generate_new_game(
        self,
        filename=ENV_FILENAME,
        folder=ENV_FOLDER,
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

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)


    def load_env(
        self,
        filename=ENV_FILENAME,
        folder=ENV_FOLDER,
        request_infos=INFO_REQUESTS,
        max_episode_steps=75
    ):
        self.env_id = textworld.gym.register_game(f"{folder}/{filename}.z8", request_infos=request_infos, max_episode_steps=max_episode_steps)
        self.env = textworld.gym.make(self.env_id)
        self.game_state.full_gold_path = json.load(open(f"{folder}/{filename}.json"))['metadata']['walkthrough']
        self.reset_env()


    # =======================================
    # Game metadata state management
    # =======================================
    def _extract_location(self, text: str) -> Optional[str]:
        """ Find '-= location =-' pattern and return the location name, else None. """
        match = re.search(r"-=\s*(.*?)\s*=-", text)
        return match.group(1).strip() if match else None


    def _update_metadata(self, env_state: Dict[str, Any], env_response=None, initial_state=False) -> None:
        """ Helper function to update our 'self.game_state' metadata. """
        self.game_state.objective = env_state['objective']
        self.game_state.game_verbs = env_state['verbs']
        self.game_state.observation = self._clean_obs(env_state['description'])
        self.game_state.inventory = env_state['inventory']
        self.game_state.moves = env_state['moves']
        self.game_state.last_command = env_state['last_command']

        # In initial state store full gold path
        if initial_state:
            self.game_state.cur_gold_path = deque(self.game_state.full_gold_path)
            self.game_state.max_score = env_state['max_score']
        else:
            self.game_state.env_response = self._clean_env_response(env_response)

        # Need to manually extract location due to bug in env
        location = self._extract_location(env_state['description'])
        self.game_state.location = location if location else self.game_state.location

        # Test for necessary context and update it if it exists
        necessary_context_trigger = 'You open the copy of "Cooking: A Modern Approach (3rd Ed.)" and start reading:'
        if not initial_state and necessary_context_trigger in env_response:
            s = env_response.split(necessary_context_trigger, 1)[1].replace("\r", "")
            ctx = self._clean_env_response(s).lstrip("\n")
            if ctx:
                self.game_state.necessary_context = self._clean_env_response(ctx)

    # =======================================
    # Other Helpers
    # =======================================
    def _trim_ascii(self, text: str) -> str:
        """
        Simple function that trims the Text World ASCII art from the initial observation
        """
        m = re.search(r"\${6,}(?![\s\S]*\${6,})([\s\S]*)\Z", text)
        if not m:
            return text
        out = m.group(1)
        out = re.sub(r"^[ \t\r\n]+", "", out)
        return out


    def _clean_obs(self, text: str) -> str:
        """
        Simple function to clean / trim the observation
        """
        text = self._trim_ascii(text)
        m = re.search(r"^(.*)>(?!.*>)", text, re.DOTALL)
        cleaned = text if not m else m.group(1)
        cleaned = re.sub(r"-=\s*.*?\s*=-", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip("\n")
        return cleaned
    

    def _clean_env_response(self, text: str) -> str:
        """Trim env response at last '>' or score notice, then strip trailing newlines."""
        text = self._trim_ascii(text)
        for marker in (">", "Your score has just gone up by one point."):
            i = text.rfind(marker)
            if i != -1:
                text = text[:i]
        stripped_text = text.rstrip("\n").lstrip("\n")
        stripped_text = stripped_text + "\n\nThe end." if self.game_state.done else stripped_text
        return stripped_text