from typing import Dict, List

from env import GuanDanEnv
from utils import Utils


class ActionGenerator:
    '''Heuristic action enumerator for GuanDan self-play.

    The goal is to surface a manageable set of candidate moves without
    brute-forcing every card subset. We lean on the environment's internal
    validators to keep only legal, competitive actions.
    '''

    def __init__(self, env: GuanDanEnv):
        self.env = env
        self.utils = Utils()
        self.cardscale = self.utils.cardscale

    def generate_legal_actions(self, obs_for_player: dict) -> List[dict]:
        '''Generate a small list of candidate responses for the given player.'''
        player_id = obs_for_player.get('id')
        deck = sorted(obs_for_player.get('deck', []))
        last_move = obs_for_player.get('last_move', {})
        last_claim = last_move.get('claim', []) if last_move else []

        has_last = bool(last_claim)
        last_type = None
        last_points = None
        if has_last:
            last_type, last_points = self.env._check_poker_type(last_claim)

        actions: List[dict] = []
        seen: set = set()

        def add_candidate(cards: List[int]):
            '''Validate and keep a candidate action if it beats last_move (when needed).'''
            cand = sorted(cards)
            key = tuple(cand)
            if key in seen:
                return
            seen.add(key)

            hand_type, hand_points = self.env._check_poker_type(cand)
            if hand_type == 'invalid':
                return
            if has_last:
                bigger = self.env._check_bigger(last_type, last_points, hand_type, hand_points)
                if bigger is not True:
                    return

            actions.append({'player': player_id, 'action': cand, 'claim': cand})

        # Unless the first to act, always expose a pass action (environment will decide legality if used first-hand).
        if has_last:
            actions.append({'player': player_id, 'action': [], 'claim': []})

        rank_groups = self._group_by_rank(deck)

        # Single cards: straightforward options.
        for card in deck:
            add_candidate([card])

        # Pairs, triples, and four-of-a-kind bombs from rank groupings.
        for cards in rank_groups.values():
            if len(cards) >= 2:
                add_candidate(cards[:2])
            if len(cards) >= 3:
                add_candidate(cards[:3])
            if len(cards) >= 4:
                add_candidate(cards[:4])
                # If more than four copies exist (two decks), offer the full pile as a heavier bomb.
                if len(cards) > 4:
                    add_candidate(cards)

        # Rocket: two small and two big jokers.
        if 'o' in rank_groups and 'O' in rank_groups:
            if len(rank_groups['o']) >= 2 and len(rank_groups['O']) >= 2:
                rocket = rank_groups['o'][:2] + rank_groups['O'][:2]
                add_candidate(rocket)

        # Short straights (length 5 to 10) using one card per rank.
        sequential_ranks = [r for r in rank_groups.keys() if r not in ['o', 'O']]
        sequential_ranks.sort(key=lambda r: self.cardscale.index(r))
        for length in range(5, 11):
            for idx in range(len(sequential_ranks) - length + 1):
                window = sequential_ranks[idx: idx + length]
                if not self._is_consecutive(window):
                    continue
                straight_cards = [rank_groups[rank][0] for rank in window]
                add_candidate(straight_cards)

        return actions

    def _group_by_rank(self, deck: List[int]) -> Dict[str, List[int]]:
        '''Group card indices by their rank character.'''
        groups: Dict[str, List[int]] = {}
        for card in deck:
            poker = self.utils.Num2Poker(card)
            rank = poker[1]
            groups.setdefault(rank, []).append(card)
        for rank in groups:
            groups[rank].sort()
        return groups

    def _is_consecutive(self, ranks: List[str]) -> bool:
        '''Check if rank characters form a consecutive run in cardscale.'''
        if not ranks:
            return False
        indices = [self.cardscale.index(r) for r in ranks]
        for pos in range(1, len(indices)):
            if indices[pos] - indices[pos - 1] != 1:
                return False
        return True
