import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.action_fc1 = nn.Linear(action_dim, hidden_dim)
        self.action_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim * 2, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: Tensor, action_batch: Tensor) -> Tuple[Tensor, Tensor]:
        state_embed = F.relu(self.state_fc1(state_batch))
        state_embed = F.relu(self.state_fc2(state_embed))

        action_embed = F.relu(self.action_fc1(action_batch))
        action_embed = F.relu(self.action_fc2(action_embed))

        joint_features = torch.cat((state_embed, action_embed), dim=-1)
        logits = self.policy_head(joint_features)
        values = self.value_head(state_embed)
        return logits, values