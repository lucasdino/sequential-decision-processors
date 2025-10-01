import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set



@dataclass
class GameState:
    generation_args: Dict = field(default_factory=dict)
    objective: str = ""
    game_verbs: List[str] = field(default_factory=list)
    location: str = ""
    observation: str = ""
    inventory: List[str] = field(default_factory=list)
    cur_gold_path: List[str] = field(default_factory=list)
    full_gold_path: List[str] = field(default_factory=list)
    last_command: str = ""
    env_response: str = ""
    moves: int = 0
    cur_score: int = 0
    max_score: int = 0
    done: bool = False
    necessary_context: str = None


    def get_full_state(self) -> str:
        full_state = (
            f"Move {self.moves} | Score = {self.cur_score}/{self.max_score}\n"
            f"Actionable Verbs: {self.game_verbs}\n"
            f"Objective: {self.objective}\n"
        )
        if self.location:
            full_state += f"Location: {self.location}\n"
        full_state += (
            f"Inventory: {self.inventory}\n"
            f"Current Observation: {self.observation}\n"
        )
        if self.necessary_context:
            full_state += f"Necessary Context: {self.necessary_context}\n"
        return [("Full Observation", full_state)]


    def get_intermediate_state(self) -> str:
        command = ("User", self.last_command)
        environment = ("Environment", self.env_response + f" Score: {self.cur_score}/{self.max_score}")
        return [command, environment]



@dataclass
class SamplingArgs:
    name: str
    max_samples_per_env: int = 5    # Max number of samples to generate for each env
    rollout_length:      int = 5    # Steps to rollout for, defined similar to 'plies' in chess ('user, env, user, env, user' would be '5'). Always starts with 'user'
    total_samples:       int = 50   # Total samples desired to generate
    max_step_overlap:    int = 0    # When sampling multiple samples from the same env, this determines the max 'ply overlap' that can exist. So if this is set to '1' you can see the same action as the first action in one sample and last action in another.


# Defining this as a standard class rather than a dataclass given required functionality
class Scienceworld_Task:
    def __init__(self, task_name: str, max_vars: int, var_range: Tuple[float, float]):
        self.task_name = task_name
        self.max_vars = max_vars
        self.var_range = var_range        
        self._instantiate_var_range()

    def _instantiate_var_range(self):
        assert self.var_range[0] <= self.var_range[1]
        assert self.var_range[0] >= 0
        assert self.var_range[1] <= 1
        
        lo = int(self.var_range[0] * self.max_vars)
        hi = int(self.var_range[1] * self.max_vars)
        self.seen_vars, self.unseen_vars = set(), set()
        for i in range(lo, hi + 1):
            self.unseen_vars.add(i)

    def sample_var(self):
        if not self.unseen_vars:
            return None, False
        val = random.choice(tuple(self.unseen_vars))
        self.unseen_vars.remove(val)
        self.seen_vars.add(val)
        return val, self.remaining_vars()

    def remaining_vars(self) -> bool:
        return len(self.unseen_vars) > 0