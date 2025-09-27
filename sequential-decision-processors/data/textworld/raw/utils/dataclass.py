from dataclasses import dataclass, field
from typing import List



@dataclass
class GameState:
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
        full_state = f"< Full Game State >\nMove {self.moves} | Score = {self.cur_score}/{self.max_score}\nActionable Verbs: {self.game_verbs}\nObjective: {self.objective}\nLocation: {self.location}\nInventory: {self.inventory}\nCurrent Observation: {self.observation}\n"
        full_state = full_state + f"Necessary Context: {self.necessary_context}\n" if self.necessary_context else full_state
        return full_state

    def get_intermediate_state(self) -> str:
        return f"Move {self.moves} | Score = {self.cur_score}/{self.max_score}\n> {self.last_command}\n{self.env_response}\n"