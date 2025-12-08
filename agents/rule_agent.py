from typing import List, Tuple

from agents.action_generator import ActionGenerator
from env import GuanDanEnv
from utils import Utils


class RuleBasedAgent:
    '''Deterministic, lightweight baseline agent for Guandan.'''

    def __init__(self, env: GuanDanEnv):
        self.env = env
        self.action_generator = ActionGenerator(env)
        self.utils = Utils()
        self.rank_order = self.utils.cardscale + ['o', 'O']
        self.rank_to_idx = {rank: idx for idx, rank in enumerate(self.rank_order)}

    def select_action(self, obs_for_player: dict) -> dict:
        candidate_actions = self.action_generator.generate_legal_actions(obs_for_player)
        pass_action = next((act for act in candidate_actions if not act.get('claim')), candidate_actions[0])

        non_pass_actions = [act for act in candidate_actions if act.get('claim')]
        if not non_pass_actions:
            return pass_action

        deck_size = len(obs_for_player.get('deck', []))
        last_move = obs_for_player.get('last_move', {}) or {}
        last_claim = last_move.get('claim', [])
        starting_trick = not bool(last_claim)

        non_bomb_actions = [act for act in non_pass_actions if not self._is_bomb_or_rocket(act)]
        bomb_actions = [act for act in non_pass_actions if self._is_bomb_or_rocket(act)]

        # Heuristic: when leading a trick, try a small multi-card set first to probe opponents.
        if starting_trick:
            multi_card = [act for act in non_bomb_actions if len(act.get('claim', [])) >= 2]
            if multi_card:
                return self._best_action(multi_card)
            if non_bomb_actions:
                return self._best_action(non_bomb_actions)
            return self._best_action(bomb_actions) if bomb_actions else pass_action

        # When responding, prefer the cheapest non-bomb action that beats the last move.
        if non_bomb_actions:
            return self._best_action(non_bomb_actions)

        # Avoid bombs unless few cards remain or no other options can beat the last move.
        if bomb_actions and (deck_size <= 3 or not non_bomb_actions):
            return self._best_action(bomb_actions)

        return pass_action

    def _is_bomb_or_rocket(self, action: dict) -> bool:
        claim = action.get('claim', [])
        hand_type, _ = self.env._check_poker_type(claim)
        return hand_type in ['bomb', 'rocket']

    def _best_action(self, actions: List[dict]) -> dict:
        '''Deterministically pick the lexicographically smallest action by size then rank.'''
        return min(actions, key=self._action_rank_key)

    def _action_rank_key(self, action: dict) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
        claim = action.get('claim', [])
        if not claim:
            return (float('inf'), float('inf'), (), ())
        ranks = [self.utils.Num2Poker(card)[1] for card in claim]
        rank_indices = [self.rank_to_idx.get(rank, len(self.rank_to_idx)) for rank in ranks]
        min_rank = min(rank_indices) if rank_indices else len(self.rank_to_idx)
        # Include sorted raw card ids as a final tie-breaker for full determinism.
        return (len(claim), min_rank, tuple(sorted(rank_indices)), tuple(sorted(claim)))
