# agents/feature_encoder.py

import numpy as np

from env import GuanDanEnv
from utils import Utils


class FeatureEncoder:
    '''
    Feature encodings for GuanDan states and actions.

    This version is aligned with env.py / utils.py you provided:
      - Uses env._check_poker_type() exact types (no "normal" ambiguity)
      - Adds explicit point one-hot + length scalar for last_move and action
      - Removes old [A3]
      - Replaces [A4] with:
          (i) make_single_onehot over A..K
          (ii) break_bomb_onehot over A..K
          (iii) break_straight_flush_onehot over A..K
        where "break_straight_flush" is computed correctly under the heart-level covering rule.

    Notes about the heart-level covering (wildcard) rule:
      - In env._is_legal_claim(), any action card equal to 'h'+level can cover any non-joker card in claim.
      - Claim cannot include jokers ('jo','jO') at all.
      - Our straight-flush availability test treats the number of covering cards as flexible substitutes for missing
        suit+rank cards in a candidate straight_flush, matching env legality constraints.
    '''

    def __init__(self, env: GuanDanEnv):
        self.env = env
        self.utils = Utils()

        # Rank order for counts: A..K plus jokers ranks (o, O) to match env.point_order extension.
        # utils.Num2Poker returns 'jo' / 'jO' and we map their rank to 'o' / 'O' via poker[1].
        self.rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'o', 'O']
        self.rank_to_idx = {r: i for i, r in enumerate(self.rank_order)}

        # A-K (13) for one-hot features requested in [A4]
        self.ak_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.ak_to_idx = {r: i for i, r in enumerate(self.ak_order)}

        # Exact card types (as requested)
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
        self.type_to_idx = {t: i for i, t in enumerate(self.card_types)}

        # Point one-hot vocabulary: use env.point_order (includes level moved near top + 'o','O' appended).
        # Rocket has points ('jo') in env._check_poker_type; we encode rocket's point as all-zeros
        # because type already identifies it and rocket's internal point does not participate in comparisons.
        self.point_vocab = None
        self.point_to_idx = None

        self.current_level = None

        # Total copies in TWO decks:
        # - Normal ranks: 8 copies (4 suits * 2 decks)
        # - Jokers: 2 copies each (1 per deck)
        self._total_per_rank = {r: (2 if r in ['o', 'O'] else 8) for r in self.rank_order}

    # =========================
    # Public APIs
    # =========================
    def encode_state(self, obs_for_player: dict) -> np.ndarray:
        '''
        Encode a single-player observation dict into a 1D feature vector.

        Feature layout:
        [S0] my_rank_counts_norm (15)
             - ranks: A,2,3,4,5,6,7,8,9,0,J,Q,K,o,O
             - normalized by total copies in 2 decks: /8 for normal ranks, /2 for jokers
        [S1] is_lead (1)
             - 1 if last_move.claim is empty (new trick / reset), else 0
        [S2] follow_is_teammate (1)
             - if not lead: whether last_move.player is teammate; if lead: 0
        [S3] hand_count_norm: self/teammate/up/down (4)
             - self exact, others inferred as 27 - cards_played_so_far
        [S4] hand_is_low (<=10): self/teammate/up/down (4)
        [S5] last_move_detail (12 + P + 1)
             - last_type_onehot over self.card_types (12)
             - last_point_onehot over env.point_order (P=15 typically, includes o,O; rocket -> all zeros)
             - last_len_scalar = len(last_claim)/10
        [S6] remaining_rank_counts_norm_for_others (15)
             - total_in_two_decks - my_current_hand - played_history, normalized

        Total dim = 15 + 1 + 1 + 4 + 4 + (12 + P + 1) + 15
        '''
        player_id = int(obs_for_player.get('id', 0))
        level = obs_for_player.get('level', '2')
        self.current_level = level
        self._ensure_point_vocab(level)

        deck = obs_for_player.get('deck', []) or []
        history = obs_for_player.get('history', []) or []
        last_move = obs_for_player.get('last_move', {}) or {}
        last_claim = last_move.get('claim', []) if last_move else []
        last_player = int(last_move.get('player', -1)) if last_move else -1

        # --- [S0] my_rank_counts_norm (15) ---
        my_rank_counts_norm = self._rank_counts_norm(deck)

        # --- [S1] is_lead (1) ---
        is_lead = 1.0 if (not last_claim) else 0.0
        is_lead_vec = np.array([is_lead], dtype=float)

        # --- [S2] follow_is_teammate (1) ---
        follow_is_teammate = 0.0
        if is_lead < 0.5 and last_player >= 0:
            follow_is_teammate = 1.0 if ((player_id - last_player) % 2 == 0) else 0.0
        follow_is_teammate_vec = np.array([follow_is_teammate], dtype=float)

        # --- [S3][S4] hand_count_norm + hand_is_low ---
        played_counts = self._cards_played_count_per_player(history)  # shape (4,)
        inferred_hand_counts = np.zeros(4, dtype=float)
        for pid in range(4):
            if pid == player_id:
                inferred_hand_counts[pid] = float(len(deck))
            else:
                inferred_hand_counts[pid] = float(max(0.0, 27.0 - played_counts[pid]))

        teammate = (player_id + 2) % 4
        up = (player_id - 1) % 4
        down = (player_id + 1) % 4

        hand_count_norm = np.array([
            inferred_hand_counts[player_id] / 27.0,
            inferred_hand_counts[teammate] / 27.0,
            inferred_hand_counts[up] / 27.0,
            inferred_hand_counts[down] / 27.0
        ], dtype=float)

        hand_is_low = np.array([
            1.0 if inferred_hand_counts[player_id] <= 10.0 else 0.0,
            1.0 if inferred_hand_counts[teammate] <= 10.0 else 0.0,
            1.0 if inferred_hand_counts[up] <= 10.0 else 0.0,
            1.0 if inferred_hand_counts[down] <= 10.0 else 0.0
        ], dtype=float)

        # --- [S5] last_move_detail (type onehot + point onehot + len scalar) ---
        last_type_onehot, last_point_onehot, last_len_scalar = self._encode_move_type_point_len(
            claim=last_claim,
            level=level
        )

        # --- [S6] remaining_rank_counts_norm_for_others (15) ---
        played_rank_counts = self._rank_counts_from_history(history)
        my_rank_counts_raw = self._rank_counts_raw(deck)
        remaining_rank_counts_norm = np.zeros(len(self.rank_order), dtype=float)
        for r in self.rank_order:
            idx = self.rank_to_idx[r]
            total = float(self._total_per_rank[r])
            rem = total - float(my_rank_counts_raw[idx]) - float(played_rank_counts[idx])
            rem = max(0.0, rem)
            remaining_rank_counts_norm[idx] = rem / total

        components = [
            my_rank_counts_norm,          # [S0] 15
            is_lead_vec,                  # [S1] 1
            follow_is_teammate_vec,       # [S2] 1
            hand_count_norm,              # [S3] 4
            hand_is_low,                  # [S4] 4
            last_type_onehot,             # [S5] 12
            last_point_onehot,            # [S5] P
            np.array([last_len_scalar], dtype=float),  # [S5] 1
            remaining_rank_counts_norm    # [S6] 15
        ]
        return np.concatenate(components).astype(float)

    def encode_action(self, obs_for_player: dict, action: dict) -> np.ndarray:
        '''
        Encode an action dict into a 1D feature vector.

        Feature layout:
        [A0] is_pass (1)
             - 1 if claim empty
        [A1] uses_heart_level (1)
             - 1 if the action (actual played cards) includes 'h'+level
        [A2] action_detail (12 + P + 1)
             - act_type_onehot over self.card_types (12)  (pass is a valid type)
             - act_point_onehot over env.point_order (P)  (rocket -> all zeros)
             - act_len_scalar = len(claim)/10
        [A4] structure_break_features (13 + 13 + 13)
             - make_single_onehot over A..K:
                 =1 for rank r if count_before(r) >= 2 and count_after(r) == 1
             - break_bomb_onehot over A..K:
                 =1 for rank r if count_before(r) >= 4 and count_after(r) < 4
             - break_straight_flush_onehot over A..K:
                 Let SF_pre(r)=1 if there exists ANY feasible straight_flush (considering covering cards)
                 whose rank-set includes r, using the pre-action hand.
                 SF_post(r)=1 similarly for the post-action hand.
                 break_SF(r) = 1 if SF_pre(r)=1 and SF_post(r)=0.

        Total dim = 1 + 1 + (12 + P + 1) + 39
        '''
        level = obs_for_player.get('level', self.current_level or '2')
        self.current_level = level
        self._ensure_point_vocab(level)

        deck = obs_for_player.get('deck', []) or []

        claim = action.get('claim', []) or []
        action_cards = action.get('action', claim) or []

        # --- [A0] is_pass ---
        is_pass = 1.0 if (len(claim) == 0) else 0.0
        is_pass_vec = np.array([is_pass], dtype=float)

        # --- [A1] uses_heart_level ---
        heart_level_str = 'h' + str(level)
        uses_heart_level = 0.0
        for c in action_cards:
            if self.utils.Num2Poker(int(c)) == heart_level_str:
                uses_heart_level = 1.0
                break
        uses_heart_level_vec = np.array([uses_heart_level], dtype=float)

        # --- [A2] act_type/point/len ---
        act_type_onehot, act_point_onehot, act_len_scalar = self._encode_move_type_point_len(
            claim=claim,
            level=level
        )

        # --- [A4] make_single / break_bomb / break_straight_flush (all over A..K) ---
        make_single_onehot, break_bomb_onehot = self._compute_break_singles_and_bombs(
            deck=deck,
            action_cards=action_cards,
            level=level
        )
        break_straight_flush_onehot = self._compute_break_straight_flush_onehot(
            deck=deck,
            action_cards=action_cards,
            level=level
        )

        components = [
            is_pass_vec,                   # [A0] 1
            uses_heart_level_vec,          # [A1] 1
            act_type_onehot,               # [A2] 12
            act_point_onehot,              # [A2] P
            np.array([act_len_scalar], dtype=float),   # [A2] 1
            make_single_onehot,            # [A4] 13
            break_bomb_onehot,             # [A4] 13
            break_straight_flush_onehot    # [A4] 13
        ]
        return np.concatenate(components).astype(float)

    # =========================
    # Internal: vocab init
    # =========================
    def _ensure_point_vocab(self, level):
        '''
        Ensure self.point_vocab == env.point_order for the current level.
        env.point_order is level-dependent (level moved to near top; o/O appended).
        '''
        # env.point_order is already updated by env.reset/_set_level for its own level,
        # but obs_for_player carries the level; we rely on env.point_order as current truth.
        # If you ever call FeatureEncoder without env.reset aligning level, you can reconstruct
        # the order here; for now, keep consistent with env.
        if self.point_vocab is None or self.current_level != level:
            self.point_vocab = list(self.env.point_order)
            self.point_to_idx = {p: i for i, p in enumerate(self.point_vocab)}

    # =========================
    # Internal: (type, point, len) encoding for a move
    # =========================
    def _encode_move_type_point_len(self, *, claim, level):
        '''
        Given a claim (list[int] card ids), encode:
        - type onehot over self.card_types (12)
        - point onehot over env.point_order (P)
        - length scalar in [0,1] using len(claim)/10

        "point" policy (no kicker for set, per your requirement):
        - pass: point all zeros
        - rocket: point -> 'o' (force to small-joker slot)
        - bomb: use points[1] (rank)
        - all other valid types: use points[0] (main rank / starting rank)
        '''
        self._ensure_point_vocab(level)

        # type
        t_onehot = np.zeros(len(self.card_types), dtype=float)
        p_onehot = np.zeros(len(self.point_vocab), dtype=float)
        l_scalar = float(len(claim)) / 10.0

        pokertype, points = self.env._check_poker_type(claim)

        idx_t = self.type_to_idx.get(pokertype, self.type_to_idx['invalid'])
        t_onehot[idx_t] = 1.0

        # point
        main_point = None

        if pokertype == 'pass' or pokertype == 'invalid':
            main_point = None

        elif pokertype == 'rocket':
            # Force rocket's point encoding to 'o' slot (small joker rank)
            main_point = 'o'

        elif pokertype == 'bomb':
            # points: (length, rank)
            if points and len(points) >= 2:
                main_point = points[1]

        else:
            # points: (rank,) or (first_rank,) or (three_rank, pair_rank) for set
            if points and len(points) >= 1:
                main_point = points[0]

        if main_point is not None and main_point in self.point_to_idx:
            p_onehot[self.point_to_idx[main_point]] = 1.0

        return t_onehot, p_onehot, l_scalar

    # =========================
    # Helpers: rank counting (A..K,o,O)
    # =========================
    def _rank_counts_raw(self, cards) -> np.ndarray:
        counts = np.zeros(len(self.rank_order), dtype=float)
        for card in cards:
            poker = self.utils.Num2Poker(int(card))
            rank = poker[1]  # 'A'..'K' or 'o'/'O'
            idx = self.rank_to_idx.get(rank)
            if idx is not None:
                counts[idx] += 1.0
        return counts

    def _rank_counts_norm(self, cards) -> np.ndarray:
        raw = self._rank_counts_raw(cards)
        norm = np.zeros_like(raw)
        for r in self.rank_order:
            idx = self.rank_to_idx[r]
            denom = float(self._total_per_rank[r])
            norm[idx] = raw[idx] / denom
        return norm.astype(float)

    def _rank_counts_from_history(self, history) -> np.ndarray:
        counts = np.zeros(len(self.rank_order), dtype=float)
        for move in history:
            action_cards = move.get('action', []) or []
            for c in action_cards:
                poker = self.utils.Num2Poker(int(c))
                rank = poker[1]
                idx = self.rank_to_idx.get(rank)
                if idx is not None:
                    counts[idx] += 1.0
        return counts

    def _cards_played_count_per_player(self, history) -> np.ndarray:
        counts = np.zeros(4, dtype=float)
        for move in history:
            pid = int(move.get('player', -1))
            action_cards = move.get('action', []) or []
            if 0 <= pid < 4:
                counts[pid] += float(len(action_cards))
        return counts

    # =========================
    # [A4] make_single + break_bomb
    # =========================
    def _compute_break_singles_and_bombs(self, *, deck, action_cards, level):
        '''
        Compute:
          - make_single_onehot[A..K]
          - break_bomb_onehot[A..K]

        Both are computed on rank-multiplicity in the player's hand before and after playing action_cards.
        Heart-level covering card still has its own rank (the level), so removing it can change multiplicities
        and is counted normally here (which matches the "拆牌" intuition).
        '''
        before_counts = self._count_ak_ranks_in_hand(deck)
        after_deck = self._remove_cards_from_deck(deck, action_cards)
        after_counts = self._count_ak_ranks_in_hand(after_deck)

        make_single = np.zeros(len(self.ak_order), dtype=float)
        break_bomb = np.zeros(len(self.ak_order), dtype=float)

        for r in self.ak_order:
            b = before_counts.get(r, 0)
            a = after_counts.get(r, 0)

            # make_single: >=2 -> 1
            if b >= 2 and a == 1:
                make_single[self.ak_to_idx[r]] = 1.0

            # break_bomb: >=4 -> <4
            if b >= 4 and a < 4:
                break_bomb[self.ak_to_idx[r]] = 1.0

        return make_single, break_bomb

    def _count_ak_ranks_in_hand(self, cards):
        '''
        Count ranks A..K in a hand, ignoring jokers.
        Returns dict {rank: count}.
        '''
        d = {r: 0 for r in self.ak_order}
        for c in cards:
            pok = self.utils.Num2Poker(int(c))
            rank = pok[1]
            if rank in d:
                d[rank] += 1
        return d

    def _remove_cards_from_deck(self, deck, action_cards):
        '''
        Remove action_cards from deck (both are lists of unique ids in normal play).
        Uses a multiset-safe approach in case of duplicates in inputs.
        '''
        remain = list(deck)
        for c in action_cards:
            if c in remain:
                remain.remove(c)
        return remain

    # =========================
    # [A4] break_straight_flush_onehot (A..K), correct under covering rule
    # =========================
    def _compute_break_straight_flush_onehot(self, *, deck, action_cards, level):
        '''
        break_straight_flush_onehot over A..K:

        1) Compute SF_pre[r] from the pre-action hand:
           SF_pre[r]=1 if there exists ANY feasible straight_flush whose rank-set includes r.

        2) Compute SF_post[r] from the post-action hand (deck minus action_cards).

        3) break[r] = 1 if SF_pre[r]=1 and SF_post[r]=0.

        Feasibility model (matches env._is_legal_claim + env._check_poker_type constraints):
          - Straight_flush requires 5 cards of the same suit whose ranks form a legal straight:
              * normal consecutive 5 in env.cardscale order, including A2345
              * special A0JQK (env recognizes points ['A','0','J','Q','K'] and returns first '0')
          - Claim cannot include jokers.
          - Covering cards are exactly 'h'+level in the ACTION, and can represent any non-joker claim card.
            Therefore, in feasibility we treat:
              * c = number of covering cards in hand
              * for a candidate (suit, ranks[5]), each missing suit+rank can be supplied by one covering card
            This correctly accounts for the heart-level wildcard ability.

        Importantly, we do NOT approximate via type rules; we compute availability by suit+rank inventory with wildcard.
        '''
        sf_pre = self._straight_flush_possible_by_rank(deck, level)
        after_deck = self._remove_cards_from_deck(deck, action_cards)
        sf_post = self._straight_flush_possible_by_rank(after_deck, level)

        out = np.zeros(len(self.ak_order), dtype=float)
        for r in self.ak_order:
            if sf_pre.get(r, False) and (not sf_post.get(r, False)):
                out[self.ak_to_idx[r]] = 1.0
        return out

    def _straight_flush_possible_by_rank(self, deck, level):
        '''
        Return dict {rank(A..K): bool} indicating whether there exists at least one feasible straight_flush
        whose 5-rank set includes that rank, given this deck and level.

        This is computed by enumerating all candidate (suit, 5-rank-sequence) patterns recognized by env,
        and checking feasibility using suit inventory + covering cards count.
        '''
        # Count covering cards
        covering = 'h' + str(level)
        cover_cnt = 0

        # Build suit->rank->count inventory excluding jokers and excluding covering cards
        inv = {s: {r: 0 for r in self.ak_order} for s in self.env.suitset}

        for c in deck:
            pok = self.utils.Num2Poker(int(c))
            if pok == 'jo' or pok == 'jO':
                continue
            if pok == covering:
                cover_cnt += 1
                continue
            suit = pok[0]
            rank = pok[1]
            if suit in inv and rank in inv[suit]:
                inv[suit][rank] += 1

        # Enumerate all candidate straight sequences as env would accept
        # Normal consecutive sequences (length 5) within env.cardscale
        # env.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        sequences = []
        cs = list(self.env.cardscale)

        # consecutive windows
        for start_idx in range(0, len(cs) - 5 + 1):
            sequences.append(cs[start_idx:start_idx + 5])

        # special A0JQK case (env checks points == ['A','0','J','Q','K'] and returns first '0')
        # This is NOT a contiguous window in cs order; add explicitly.
        sequences.append(['A', '0', 'J', 'Q', 'K'])

        possible = {r: False for r in self.ak_order}

        # For each suit and each sequence, check feasibility: need 5 suit-cards, missing filled by cover_cnt
        for suit in self.env.suitset:
            suit_inv = inv[suit]
            for seq in sequences:
                missing = 0
                for r in seq:
                    if suit_inv.get(r, 0) <= 0:
                        missing += 1
                if missing <= cover_cnt:
                    # This straight_flush is feasible (choose missing cards via covering)
                    for r in seq:
                        if r in possible:
                            possible[r] = True

        return possible
