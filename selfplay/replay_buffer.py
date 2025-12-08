from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch


@dataclass
class Transition:
    state: Any
    action_feat: Any
    action_index: int
    reward: float
    done: bool
    log_prob: float


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

    def to_training_batch(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        states = np.stack([np.asarray(t.state, dtype=np.float32) for t in self.storage], axis=0)
        action_feats = np.stack([np.asarray(t.action_feat, dtype=np.float32) for t in self.storage], axis=0)
        actions_idx = np.array([t.action_index for t in self.storage], dtype=np.int64)
        rewards = np.array([t.reward for t in self.storage], dtype=np.float32)
        dones = np.array([1.0 if t.done else 0.0 for t in self.storage], dtype=np.float32)
        old_log_probs = np.array([t.log_prob for t in self.storage], dtype=np.float32)

        return {
            'states': torch.tensor(states, dtype=torch.float32, device=device),
            'action_feats': torch.tensor(action_feats, dtype=torch.float32, device=device),
            'actions_idx': torch.tensor(actions_idx, dtype=torch.long, device=device),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=device),
            'dones': torch.tensor(dones, dtype=torch.float32, device=device),
            'old_log_probs': torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self.storage.clear()
