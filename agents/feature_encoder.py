import numpy as np

from env import GuanDanEnv
from utils import Utils


class FeatureEncoder:
    '''Simple, deterministic encodings for GuanDan states and actions.'''

    def __init__(self):
        # Card rank ordering used across encodings; jokers appended.
        self.rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'o', 'O']
        self.rank_to_idx = {rank: idx for idx, rank in enumerate(self.rank_order)}
        self.card_types = [
            'pass',
            'single',
            'pair',
            'three',
            'straight',
            'set',
            'three_straight',
            'triple_pairs',
            'bomb',
            'rocket',
            'straight_flush',
            'invalid'
        ]
        self.type_to_idx = {ctype: idx for idx, ctype in enumerate(self.card_types)}
        self.utils = Utils()
        self.env = GuanDanEnv()
        self.current_level = None

    def encode_state(self, obs_for_player: dict) -> np.ndarray:
        '''Encode a single-player observation dict into a 1D feature vector.'''
        player_id = obs_for_player.get('id', 0)
        level = obs_for_player.get('level')
        self.current_level = level
        deck = obs_for_player.get('deck', [])
        history = obs_for_player.get('history', [])
        last_move = obs_for_player.get('last_move', {})
        last_claim = last_move.get('claim', []) if last_move else []

        level_one_hot = self._one_hot_rank(level)
        deck_counts = self._rank_counts(deck)
        played_counts = self._cards_played_per_player(history)
        bombs_flag = np.array([1.0 if self._any_bombs_or_rockets(history) else 0.0], dtype=float)

        if last_claim:
            last_type, last_points = self.env._check_poker_type(last_claim)
        else:
            last_type, last_points = 'pass', ()
        last_type_one_hot = self._one_hot_type(last_type)
        last_rank_one_hot = self._one_hot_rank(last_points[0] if last_points else None)

        player_one_hot = np.zeros(4, dtype=float)
        if 0 <= player_id < 4:
            player_one_hot[player_id] = 1.0

        components = [
            level_one_hot,
            deck_counts,
            played_counts,
            bombs_flag,
            last_type_one_hot,
            last_rank_one_hot,
            player_one_hot
        ]
        return np.concatenate(components).astype(float)

    def encode_action(self, action: dict) -> np.ndarray:
        '''Encode an action dict into a 1D feature vector.'''
        claim = action.get('claim', [])
        level = self.current_level or '2'
        act_type, act_points = self.env._check_poker_type(claim)
        act_type_one_hot = self._one_hot_type(act_type)
        act_rank_one_hot = self._one_hot_rank(act_points[0] if act_points else None)

        num_cards = np.array([len(claim)], dtype=float)
        ranks_in_action = [self.utils.Num2Poker(c)[1] for c in claim]
        distinct_rank_count = np.array([len(set(ranks_in_action))], dtype=float)
        bomb_flag = np.array([1.0 if act_type in ['bomb', 'rocket'] else 0.0], dtype=float)
        level_flag = np.array([1.0 if level in ranks_in_action else 0.0], dtype=float)

        components = [
            act_type_one_hot,
            act_rank_one_hot,
            num_cards,
            distinct_rank_count,
            bomb_flag,
            level_flag
        ]
        return np.concatenate(components).astype(float)

    def _rank_counts(self, deck) -> np.ndarray:
        counts = np.zeros(len(self.rank_order), dtype=float)
        for card in deck:
            rank = self.utils.Num2Poker(card)[1]
            idx = self.rank_to_idx.get(rank)
            if idx is not None:
                counts[idx] += 1.0
        return counts

    def _cards_played_per_player(self, history) -> np.ndarray:
        counts = np.zeros(4, dtype=float)
        for move in history:
            player = move.get('player', -1)
            action_cards = move.get('action', [])
            if 0 <= player < 4:
                counts[player] += float(len(action_cards))
        return counts

    def _any_bombs_or_rockets(self, history) -> bool:
        for move in history:
            claim = move.get('claim', [])
            if not claim:
                continue
            mtype, _ = self.env._check_poker_type(claim)
            if mtype in ['bomb', 'rocket']:
                return True
        return False

    def _one_hot_rank(self, rank) -> np.ndarray:
        vec = np.zeros(len(self.rank_order), dtype=float)
        idx = self.rank_to_idx.get(rank)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _one_hot_type(self, card_type: str) -> np.ndarray:
        vec = np.zeros(len(self.card_types), dtype=float)
        idx = self.type_to_idx.get(card_type)
        if idx is not None:
            vec[idx] = 1.0
        return vec
