import os, sys
from typing import Callable, Optional, Tuple, Dict, Any, List
from contextlib import contextmanager


# =======================================
# Helper to limit unwanted print statements
# =======================================
@contextmanager
def silent_io():
    devnull = open(os.devnull, "w")
    _out, _err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = devnull
        yield
    finally:
        sys.stdout, sys.stderr = _out, _err
        devnull.close()


# =======================================
# Sampling Manager
# =======================================
class Textworld_Sampling_Manager():
    DEFAULT_SAMPLING_ARGS = {
        "max_samples": 5,
        "rollout_length": 5
    }
    MAX_ALLOWABLE_OVERLAP = 1

    def __init__(self, wrapper, sampling_args=DEFAULT_SAMPLING_ARGS):
        """ Sampling manager that sits around our wrapper and generates samples. """
        self.wrapper = wrapper
        self.sampling_args = sampling_args

    
    def _get_sampling_moves(self) -> None:
        """ Generates the indices for sampling. """
        gold_path_length = len(self.wrapper.game_state.full_gold_path)

        # Now see how many possible samples we can get with minimal overlap
        