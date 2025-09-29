from dataclasses import dataclass, field
from typing import List, Dict



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
        full_state = f"Move {self.moves} | Score = {self.cur_score}/{self.max_score}\nActionable Verbs: {self.game_verbs}\nObjective: {self.objective}\nLocation: {self.location}\nInventory: {self.inventory}\nCurrent Observation: {self.observation}\n"
        full_state = full_state + f"Necessary Context: {self.necessary_context}\n" if self.necessary_context else full_state
        return [("Full Observation", full_state)]

    def get_intermediate_state(self) -> str:
        command = ("User", self.last_command)
        environment = ("Environment", self.env_response + f" [Move {self.moves} | Score = {self.cur_score}/{self.max_score}]")
        return [command, environment]    

@dataclass
class SamplingArgs:
    name: str
    max_samples_per_env: int = 5    # Max number of samples to generate for each env
    rollout_length:      int = 5    # Steps to rollout for, defined similar to 'plies' in chess ('user, env, user, env, user' would be '5'). Always starts with 'user'
    total_samples:       int = 50   # Total samples desired to generate
    max_step_overlap:    int = 0    # When sampling multiple samples from the same env, this determines the max 'ply overlap' that can exist. So if this is set to '1' you can see the same action as the first action in one sample and last action in another.