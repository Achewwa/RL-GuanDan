from typing import List, Tuple
from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    '''Deterministic, lightweight baseline agent for Guandan.'''
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
        if bomb_actions and (deck_size <= 10 or not non_bomb_actions):
            return self._best_action(bomb_actions)

        return pass_action

    def _is_bomb_or_rocket(self, action: dict) -> bool:
        claim = action.get('claim', [])
        hand_type, _ = self.env._check_poker_type(claim)
        return hand_type in ['bomb', 'rocket']

    def _best_action(self, actions: List[dict]) -> dict:
        
        def key_fun(a):
            # 统计 a 被多少个动作压过
            beaten_count = 0
            for b in actions:
                if b is not a and self._is_strictly_bigger(b, a):
                    beaten_count += 1

            return beaten_count

        return min(actions, key=key_fun)

    def _is_strictly_bigger(self, a: dict, b: dict) -> bool:
        claim_a = a.get('claim', [])
        claim_b = b.get('claim', [])

        type_a, pts_a = self.env._check_poker_type(claim_a)
        type_b, pts_b = self.env._check_poker_type(claim_b)

        # _check_bigger(type_b, pts_b, type_a, pts_a) 判定 a > b ?
        return self.env._check_bigger(type_b, pts_b, type_a, pts_a) is True
