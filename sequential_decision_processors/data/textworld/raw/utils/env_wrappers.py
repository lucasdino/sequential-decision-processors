import re, json, random, shutil, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

import textworld.gym
from textworld import EnvInfos
from scienceworld import ScienceWorldEnv

from .dataclass import GameState, Scienceworld_Task
from .util import silent_io
from .exceptions import NoFurtherGenerations

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

    DEFAULT_GAME_GEN_ARGS = {
        "recipe": 5,
        "take": 3,
        "go": 1,
        "open_": True,
        "cook": True,
        "cut": True,
        "drop": True
    }

    def __init__(self):
        """
        Wrapper used to get standard input/output for the cooking task.
        """
    
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
        randomize_gen_args: bool = True,
        split: str = "train",
        recipe_seed: int | None = None,
        seed: int | None = None,
        fmt: str = "z8",
        max_attempts: int = 30,
        reroll_fixed_seeds: bool = False,
    ):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        last_err = None

        for attempt in range(1, max_attempts + 1):
            gen = (self._random_game_args() if randomize_gen_args
                else dict(self.DEFAULT_GAME_GEN_ARGS))
            cur_seed   = (random.randint(0, 2**31 - 1)
                        if (seed is None or (reroll_fixed_seeds and attempt > 1)) else seed)
            cur_rseed  = (random.randint(0, 2**31 - 1)
                        if (recipe_seed is None or (reroll_fixed_seeds and attempt > 1)) else recipe_seed)

            # nuke previous artifacts
            for p in list(folder.glob(filename)) + list(folder.glob(filename + ".*")):
                (p.unlink() if (p.is_file() or p.is_symlink()) else shutil.rmtree(p))

            args_dict = {
                "filename": filename,
                "folder": str(folder),
                "fmt": fmt,
                "split": split,
                "seed": cur_seed,
                "recipe_seed": cur_rseed,
                **gen,  # recipe, take, go, open_, cook, cut, drop
            }

            ok, cmd, stderr = self._run_tw_make(args_dict)
            if ok:
                take, open_, drop = gen["take"], gen["open_"], gen["drop"]
                ctx  = (f"You can only carry {take} items at a time. " if drop else "There is no inventory limit. ")
                ctx += ("You must open containers / doors first. " if open_ else "You do not need to open containers / doors. ")
                self.game_state = GameState(generation_args=args_dict, necessary_context=ctx)
                return
            else:
                print(f"Failed on generation: {gen}")

            last_err = f"Attempt {attempt}/{max_attempts} failed.\nCMD: {' '.join(cmd)}\nSTDERR:\n{stderr}"

        raise RuntimeError(f"tw-make failed after {max_attempts} attempts.\n\n{last_err}")

    def load_env(
        self,
        filename=ENV_FILENAME,
        folder=ENV_FOLDER,
        request_infos=INFO_REQUESTS,
        max_episode_steps=75
    ):
        self.game_state = GameState()
        self.env_id = textworld.gym.register_game(f"{folder}/{filename}.z8", request_infos=request_infos, max_episode_steps=max_episode_steps)
        self.env = textworld.gym.make(self.env_id)
        self.game_state.full_gold_path = self._clean_goldpath(json.load(open(f"{folder}/{filename}.json"))['metadata']['walkthrough'])
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
        self.game_state.observation = self._clean_obs(env_state['description'])
        self.game_state.inventory = env_state['inventory']
        self.game_state.moves = env_state['moves']
        self.game_state.last_command = env_state['last_command'].capitalize() if env_state['last_command'] else None

        # In initial state store full gold path
        if initial_state:
            self.game_state.objective = env_state['objective']
            self.game_state.game_verbs = env_state['verbs']
            self.game_state.max_score = env_state['max_score']
            self.game_state.cur_gold_path = deque(self.game_state.full_gold_path)
            self.game_state.done = False
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
                if self.game_state.necessary_context:
                    self.game_state.necessary_context += f"\n{self._clean_env_response(ctx)}"
                else:
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
    

    def _clean_goldpath(self, goldpath):
        new_goldpath = []
        for g in goldpath:
            if g.lower() == "inventory":
                continue
            new_goldpath.append(g)
        return new_goldpath


    def _run_tw_make(self, args: dict):
        out = Path(args["folder"]) / f"{args['filename']}.{args['fmt']}"
        cmd = [
            "tw-make", "tw-cooking",
            "--recipe", str(args["recipe"]),
            "--take",   str(args["take"]),
            "--go",     str(args["go"]),
            "--split",  args["split"],
            "--format", args["fmt"],
            "--output", str(out),
        ]
        for flag, ok in [("--open", args["open_"]), ("--cook", args["cook"]),
                        ("--cut", args["cut"]), ("--drop", args["drop"])]:
            if ok: cmd.append(flag)
        if args["recipe_seed"] is not None: cmd += ["--recipe-seed", str(args["recipe_seed"])]
        if args["seed"] is not None:        cmd += ["--seed", str(args["seed"])]

        res = subprocess.run(cmd, capture_output=True, text=True)
        return (res.returncode == 0), cmd, res.stderr


    def _random_game_args(self, p_true: float = 0.8) -> dict:
        """
        Random gen-args:
        recipe ∈ [2,5] (uniform); take ∈ [0, recipe] (uniform);
        go ∈ {1,6,9,12} (uniform); booleans True w.p. p_true.
        """
        recipe = random.randint(2, 5)
        take   = random.randint(1, recipe)
        go     = random.choice([1, 6, 9, 12])
        bern   = lambda: random.random() < p_true

        return {
            "recipe": recipe, "take": take, "go": go,
            "open_": bern(), "cook": bern(), "cut": bern(), "drop": not bern(),
        }
    


# ######################################################
# #########     Scienceworld Wrapper      #########
# ######################################################
class Scienceworld_Wrapper_Env():
    ENV_FILENAME = "scienceworld"
    ENV_FOLDER = "tw_games"

    def __init__(self, env_step_limit=100):
        """
        Wrapper used to get standard input/output for the cooking task.
        """
        self.env = ScienceWorldEnv("", "", envStepLimit=env_step_limit)
        self._instantiate_variations()
    
    # =======================================
    # Env handling
    # =======================================
    def reset_env(self):
        with silent_io():
            env_response, env_state = self.env.reset()
        self.game_state.full_gold_path = self.env.get_gold_action_sequence()
        self._update_metadata(env_state=env_state, initial_state=True)
        

    def step_env(self, action=None):
        with silent_io():
            if action:
                env_response, reward, done, env_state = self.env.step(action)
            else:
                action = self.game_state.cur_gold_path.popleft()
                env_response, reward, done, env_state = self.env.step(action)
            
        # Update metadata
        self.game_state.done = done
        self.game_state.cur_score += reward
        self._update_metadata(env_state=env_state, env_response=env_response, action=action, initial_state=False)
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
    def generate_new_game(self, **kwargs):
        # We don't need this function for Scienceworld but keeping for consistency with other wrappers 
        pass


    def load_env(self, randomize_gen_args: bool = True, task_args = None):
        """ Function to load an environment. If you set 'randomize_gen_args' to False, you must provide 'task_args'. 'task_args' should be a tuple: ('task_name': str, 'variation_number': int, 'simplifications': str). """
        self.game_state = GameState()
        try:
            gen_args = self.task_generator.sample_task() if randomize_gen_args else task_args
            if gen_args is None:
                raise ValueError("If you do not randomize generation args, you must provide 'task_args'.")
            task_name, variation, simplifications = gen_args
        except Exception as e:
            if isinstance(e, NoFurtherGenerations):
                print(f"No further generations possible. Exiting gracefully.\n\n{e}")
            else:
                raise e

        # Load our env
        self.env.load(
            taskName=task_name, variationIdx=variation, simplificationStr=simplifications, generateGoldPath=True
        )
        self.reset_env()


    # =======================================
    # Game metadata state management
    # =======================================
    def _update_metadata(self, env_state: Dict[str, Any], env_response=None, action=None, initial_state=False) -> None:
        """ Helper function to update our 'self.game_state' metadata. """
        self.game_state.observation = env_state['look']
        self.game_state.inventory = env_state['inv']
        self.game_state.moves = env_state['moves']
        self.game_state.last_command = action

        if initial_state:
            self.game_state.objective = self._clean_objective(env_state['taskDesc'])
            self.game_state.cur_score = 0
            self.game_state.max_score = 100
            self.game_state.game_verbs = self._get_actions()
            self.game_state.cur_gold_path = deque(self.game_state.full_gold_path)
            self.game_state.done = False
            self.game_state.necessary_context = f"Game Args - Simplifications: {env_state['simplificationStr']}" if env_state['simplificationStr'] else None
        else:
            self.game_state.env_response = env_response

        # Need to manually extract location due to bug in env
        location = self._extract_location(env_state['look'])
        self.game_state.location = location if location else self.game_state.location


    # =======================================
    # Other Helpers
    # =======================================
    def _clean_objective(self, text: str) -> str:
        return text[text.find(":")+2:]    

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Look for 'This room is called the ____.' and return the extracted name (stripped),
        or None if not present.
        """
        m = re.search(r"\bThis room is called the\s+(.+?)\.", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        name = m.group(1).strip().strip('"\'').capitalize()
        return name or None


    def _get_actions(self) -> List[str]:
        """ Docs aren't super clear but this will give us all the possible actions we can use for that task (regardless of step). """
        final_actions = set()
        for action in self.env.get_possible_actions():
            count = action.count("OBJ")
            if count == 0:
                final_actions.add(action.strip())
            elif count == 1:
                part = action.split("OBJ", 1)[0].strip()
                final_actions.add(part)
            elif count == 2:
                part = action.split("OBJ", 2)[0:2]  # split into first two chunks
                final_actions.add("OBJ".join(part).strip())
            else:
                raise ValueError(f"Unexpected >2 'OBJ' in word: {action}")
        final_actions = [a for a in final_actions if not "reset" in a]
        return sorted(final_actions)    # Return as sorted list


    # =======================================
    # Scienceworld Helpers
    # =======================================
    def _instantiate_variations(self):
        self.task_generator = Scienceworld_Random_Generator(self.env, var_range=[0.0, 0.5])
    


# Helper to generate random environment variations
class Scienceworld_Random_Generator():
    def __init__(self, env: ScienceWorldEnv, var_range=(0.0, 0.5)):
        self.tasks = set()
        self.spent_tasks = set()
        for t in env.get_task_names():
            max_vars = env.get_max_variations(t)
            self.tasks.add(Scienceworld_Task(t, max_vars, var_range))
        
    def sample_task(self):
        if len(self.tasks) == 0:
            raise NoFurtherGenerations()
        
        task = random.choice(tuple(self.tasks))
        variant, still_valid = task.sample_var()
        
        if not still_valid:
            self.tasks.remove(task)
            self.spent_tasks.add(task)

        # Now return our necessary task data for generation
        return task.task_name, variant, self._get_simplifications()
    
    def _get_simplifications(self):
        return ""
