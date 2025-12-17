import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from agents.action_generator import ActionGenerator
from agents.feature_encoder import FeatureEncoder
from models.policy_value_net import PolicyValueNet


class RLGuanDanAgent:
    '''Policy wrapper combining feature encoding, action generation, and a policy-value network.'''

    def __init__(
        self,
        policy_value_net: PolicyValueNet,
        feature_encoder: FeatureEncoder,
        action_generator: ActionGenerator,
        device: str = 'cpu'
    ) -> None:
        self.policy_value_net = policy_value_net
        self.feature_encoder = feature_encoder
        self.action_generator = action_generator
        self.device = device

    def _compute_action_distribution(self, obs_for_player: dict, explore: bool = True) -> Tuple[List[dict], torch.Tensor, torch.Tensor, list, list]:
        candidate_actions = self.action_generator.generate_legal_actions(obs_for_player)

        has_non_pass = any(action.get('claim') for action in candidate_actions)
        if not has_non_pass:
            player_id = obs_for_player.get('id', 0)
            candidate_actions = [{'player': player_id, 'action': [], 'claim': []}]

        state_vec = self.feature_encoder.encode_state(obs_for_player)
        action_vecs = [self.feature_encoder.encode_action(obs_for_player, action) for action in candidate_actions]

        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        state_batch = state_tensor.unsqueeze(0).repeat(len(action_vecs), 1)

        action_vecs = np.asarray(action_vecs, dtype=np.float32)
        action_tensor = torch.tensor(action_vecs, dtype=torch.float32, device=self.device)

        logits, _ = self.policy_value_net(state_batch, action_tensor)
        logits_flat = logits.squeeze(-1)

        probabilities = F.softmax(logits_flat, dim=0)
        log_probabilities = F.log_softmax(logits_flat, dim=0)

        return candidate_actions, probabilities, log_probabilities, state_vec, action_vecs

    def select_action(self, obs_for_player: dict, explore: bool = True) -> dict:
        candidate_actions, probabilities, _, _, _ = self._compute_action_distribution(obs_for_player, explore=explore)

        if explore:
            action_idx = torch.multinomial(probabilities, 1).item()
        else:
            action_idx = torch.argmax(probabilities).item()

        return candidate_actions[action_idx]

    def select_action_with_logprob(self, obs_for_player: dict, explore: bool = True) -> Tuple[dict, int, torch.Tensor]:
        candidate_actions, probabilities, log_probabilities, _, _ = self._compute_action_distribution(obs_for_player, explore=explore)

        if explore:
            action_idx = torch.multinomial(probabilities, 1).item()
        else:
            action_idx = torch.argmax(probabilities).item()

        selected_action = candidate_actions[action_idx]
        selected_log_prob = log_probabilities[action_idx].detach()

        return selected_action, action_idx, selected_log_prob
