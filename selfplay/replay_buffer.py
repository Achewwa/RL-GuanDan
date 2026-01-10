from dataclasses import dataclass
from typing import List, Any


@dataclass
class Transition:
    # --- identifiers (NEW) ---
    episode_id: int
    player_id: int
    
    state: Any  # encoded state vector
    candidate_action_feats: Any  # all candidate action encodings for this step
    action_index: int
    reward: float
    done: bool
    log_prob: float  # log-softmax of the chosen action at sampling time


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Transition] = []

    def add_transition(self, transition: Transition) -> None:
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(transition)

    def add_trajectory(self, transitions: List[Transition]) -> None:
        for transition in transitions:
            self.add_transition(transition)

    def __len__(self) -> int:
        return len(self.storage)

    def to_training_batch(self, device: str = 'cpu') -> List[Transition]:
        # Return raw transitions so callers can handle variable-sized candidate sets.
        return list(self.storage)

    def clear(self) -> None:
        self.storage.clear()
