import numpy as np

from env import GuanDanEnv
from agents.action_generator import ActionGenerator
from utils import Utils


class FeatureEncoder:
    '''Simple, deterministic encodings for GuanDan states and actions.'''

    def __init__(self, env: GuanDanEnv = None, action_generator: ActionGenerator = None):
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
        self.cand_type_order = [
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
            'straight_flush'
        ]
        self.cand_type_to_idx = {ctype: idx for idx, ctype in enumerate(self.cand_type_order)}
        self.utils = Utils()
        self.env = env or GuanDanEnv()
        self.action_generator = action_generator or ActionGenerator(self.env)
        self.current_level = None
        self._last_obs_cache = None

    def encode_state(self, obs_for_player: dict) -> np.ndarray:
        '''Encode a single-player observation dict into a 1D feature vector.'''
        return self.encode_state_v2(obs_for_player)

    def encode_action(self, obs_for_player: dict, action: dict = None) -> np.ndarray:
        '''Encode an action dict into a 1D feature vector using v2 features.'''
        if action is None:
            action = obs_for_player
            obs_for_player = self._last_obs_cache
            if obs_for_player is None:
                raise ValueError('encode_action requires obs_for_player when no cached observation is available.')
        return self.encode_action_v2(obs_for_player, action)

    def encode_state_v2(self, obs_for_player: dict) -> np.ndarray:
        player_id = obs_for_player.get('id', -1)
        level = obs_for_player.get('level') or self.current_level or '2'
        self.current_level = level
        self._last_obs_cache = obs_for_player
        deck = obs_for_player.get('deck', [])
        history = obs_for_player.get('history', [])
        last_move = obs_for_player.get('last_move', {}) or {}
        last_claim = last_move.get('claim', []) if last_move else []
        last_player = last_move.get('player', -1)

        player_one_hot = self._one_hot_player(player_id)
        level_one_hot = self._one_hot_rank(level)
        level_scalar = np.array([self.rank_to_idx.get(level, self.rank_to_idx['2']) / 14.0], dtype=np.float32)

        rank_counts = self._rank_counts(deck)
        hand_rank_counts_norm = rank_counts / 8.0  # Two decks -> at most 8 cards per rank.
        wild_count = self._wild_count(deck, level)
        wild_count_norm = np.array([wild_count / 2.0], dtype=np.float32)  # Two decks -> at most 2 wilds.
        hand_summary_norm = self._hand_summary(rank_counts, len(deck))

        cand_type_counts_norm = self._candidate_type_counts(obs_for_player)

        last_type, last_points = self.env._check_poker_type(last_claim) if last_claim else ('pass', ())
        last_type_one_hot = self._one_hot_type(last_type)
        last_rank = last_points[0] if last_points else None
        last_rank_one_hot = self._one_hot_rank(last_rank)
        last_len_norm = np.array([len(last_claim) / 10.0], dtype=np.float32)  # Typical trick sizes are small; /10 caps it.
        starting_trick_flag = np.array([1.0 if not last_claim else 0.0], dtype=np.float32)
        last_player_rel = self._relative_player_one_hot(player_id, last_player)
        last_is_bomb_flag = np.array([1.0 if last_type in ['bomb', 'rocket', 'straight_flush'] else 0.0], dtype=np.float32)
        last_bomb_len_norm = np.array([self._bomb_len_norm(last_type, last_points)], dtype=np.float32)

        depletion_counts_norm = self._depletion_counts(history)

        components = [
            player_one_hot,
            level_one_hot,
            level_scalar,
            hand_rank_counts_norm,
            wild_count_norm,
            hand_summary_norm,
            cand_type_counts_norm,
            last_type_one_hot,
            last_rank_one_hot,
            last_len_norm,
            starting_trick_flag,
            last_player_rel,
            last_is_bomb_flag,
            last_bomb_len_norm,
            depletion_counts_norm
        ]
        state_vec = np.concatenate(components).astype(np.float32)
        assert state_vec.shape == (103,)
        return state_vec

    def encode_action_v2(self, obs_for_player: dict, action: dict) -> np.ndarray:
        self._last_obs_cache = obs_for_player
        level = obs_for_player.get('level') or self.current_level or '2'
        self.current_level = level
        deck = obs_for_player.get('deck', [])
        claim = action.get('claim', [])
        action_cards = action.get('action', [])

        act_type, act_points = self.env._check_poker_type(claim) if claim else ('pass', ())
        act_type_one_hot = self._one_hot_type(act_type)
        act_rank = act_points[0] if act_points else None
        act_rank_one_hot = self._one_hot_rank(act_rank)
        act_rank_scalar = np.array([self.rank_to_idx.get(act_rank, 0) / 14.0], dtype=np.float32)
        num_cards_norm = np.array([len(claim) / 10.0], dtype=np.float32)  # /10 keeps counts in a small, comparable range.

        claim_ranks = [self.utils.Num2Poker(c)[1] for c in claim]
        distinct_rank_count_norm = np.array([len(set(claim_ranks)) / 15.0], dtype=np.float32)

        uses_wild_flag = np.array([1.0 if any(self.utils.Num2Poker(c) == 'h' + level for c in action_cards) else 0.0], dtype=np.float32)
        claim_diff_flag = np.array([1.0 if action_cards != claim else 0.0], dtype=np.float32)
        bomb_len_norm = np.array([self._bomb_len_norm(act_type, act_points)], dtype=np.float32)
        level_in_claim_flag = np.array([1.0 if level in claim_ranks else 0.0], dtype=np.float32)
        is_pass_flag = np.array([1.0 if not claim else 0.0], dtype=np.float32)

        deck_after = self._remove_cards(deck, action_cards)
        remaining_rank_counts = self._rank_counts(deck_after)
        remaining_cards_norm = np.array([len(deck_after) / 27.0], dtype=np.float32)  # 54 cards / 4 players -> ~27 max hand size.
        remaining_distinct_ranks_norm = np.array([np.count_nonzero(remaining_rank_counts) / 15.0], dtype=np.float32)
        remaining_wild_count_norm = np.array([self._wild_count(deck_after, level) / 2.0], dtype=np.float32)
        remaining_max_same_rank_norm = np.array([remaining_rank_counts.max() / 8.0 if remaining_rank_counts.size else 0.0], dtype=np.float32)
        remaining_num_ranks_ge4_norm = np.array([np.count_nonzero(remaining_rank_counts >= 4) / 15.0], dtype=np.float32)
        remaining_num_ranks_ge2_norm = np.array([np.count_nonzero(remaining_rank_counts >= 2) / 15.0], dtype=np.float32)

        components = [
            act_type_one_hot,
            act_rank_one_hot,
            act_rank_scalar,
            num_cards_norm,
            distinct_rank_count_norm,
            uses_wild_flag,
            claim_diff_flag,
            bomb_len_norm,
            level_in_claim_flag,
            is_pass_flag,
            remaining_cards_norm,
            remaining_distinct_ranks_norm,
            remaining_wild_count_norm,
            remaining_max_same_rank_norm,
            remaining_num_ranks_ge4_norm,
            remaining_num_ranks_ge2_norm
        ]
        action_vec = np.concatenate(components).astype(np.float32)
        assert action_vec.shape == (41,)
        return action_vec

    def _rank_counts(self, deck) -> np.ndarray:
        counts = np.zeros(len(self.rank_order), dtype=np.float32)
        for card in deck:
            rank = self.utils.Num2Poker(card)[1]
            idx = self.rank_to_idx.get(rank)
            if idx is not None:
                counts[idx] += 1.0
        return counts

    def _wild_count(self, deck, level: str) -> float:
        return float(sum(1 for c in deck if self.utils.Num2Poker(c) == 'h' + level))

    def _hand_summary(self, rank_counts: np.ndarray, deck_len: int) -> np.ndarray:
        distinct_ranks = np.count_nonzero(rank_counts)
        max_same_rank = rank_counts.max() if rank_counts.size else 0.0
        num_ranks_ge2 = np.count_nonzero(rank_counts >= 2)
        num_ranks_ge3 = np.count_nonzero(rank_counts >= 3)
        num_ranks_ge4 = np.count_nonzero(rank_counts >= 4)
        return np.array([
            deck_len / 27.0,  # 54 cards split by 4 players -> ~27 max.
            distinct_ranks / 15.0,
            max_same_rank / 8.0,
            num_ranks_ge2 / 15.0,
            num_ranks_ge3 / 15.0,
            num_ranks_ge4 / 15.0
        ], dtype=np.float32)

    def _candidate_type_counts(self, obs_for_player: dict) -> np.ndarray:
        temp_obs = dict(obs_for_player)
        temp_obs['deck'] = list(obs_for_player.get('deck', []))
        temp_obs['history'] = list(obs_for_player.get('history', []))
        temp_obs['last_move'] = {'player': -1, 'action': [], 'claim': []}
        candidate_actions = self.action_generator.generate_legal_actions(temp_obs)

        counts = np.zeros(len(self.cand_type_order), dtype=np.float32)
        for cand in candidate_actions:
            claim = cand.get('claim', [])
            cand_type, _ = self.env._check_poker_type(claim) if claim else ('pass', ())
            idx = self.cand_type_to_idx.get(cand_type)
            if idx is not None:
                counts[idx] += 1.0
        return counts / 20.0  # Cap-like normalization: typical candidate lists are << 20.

    def _relative_player_one_hot(self, me: int, other: int) -> np.ndarray:
        vec = np.zeros(4, dtype=np.float32)
        if 0 <= me < 4 and 0 <= other < 4:
            rel = (other - me) % 4
            vec[rel] = 1.0
        return vec

    def _bomb_len_norm(self, move_type: str, points) -> float:
        if move_type == 'bomb' and points and isinstance(points, tuple):
            return float(points[0]) / 10.0
        if move_type == 'rocket':
            return 4.0 / 10.0
        if move_type == 'straight_flush':
            return 5.0 / 10.0
        return 0.0

    def _depletion_counts(self, history) -> np.ndarray:
        counts = np.zeros(len(self.rank_order), dtype=np.float32)
        for move in history:
            for card in move.get('action', []):
                rank = self.utils.Num2Poker(card)[1]
                idx = self.rank_to_idx.get(rank)
                if idx is not None:
                    counts[idx] += 1.0
        return counts / 8.0  # Two decks -> up to 8 per rank.

    def _remove_cards(self, deck, cards_to_remove):
        remaining = list(deck)
        for card in cards_to_remove:
            if card in remaining:
                remaining.remove(card)
        return remaining

    def _one_hot_rank(self, rank) -> np.ndarray:
        vec = np.zeros(len(self.rank_order), dtype=np.float32)
        idx = self.rank_to_idx.get(rank)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _one_hot_type(self, card_type: str) -> np.ndarray:
        vec = np.zeros(len(self.card_types), dtype=np.float32)
        idx = self.type_to_idx.get(card_type)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _one_hot_player(self, player_id: int) -> np.ndarray:
        vec = np.zeros(4, dtype=np.float32)
        if 0 <= player_id < 4:
            vec[player_id] = 1.0
        return vec
