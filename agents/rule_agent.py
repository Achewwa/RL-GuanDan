from typing import Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils import Utils


class RuleBasedAgent(BaseAgent):
    '''
    Heuristic baseline inspired by common Guandan rule strategies:
    - Separate lead vs follow behavior.
    - Use inferred hand sizes to detect endgame danger/help situations.
    - Do not casually break bombs (>=4 same-rank in hand).
    '''

    # Thresholds
    _ENEMY_DANGER_TH = 2   # enemy has <=2 cards -> very dangerous
    _ALLY_CLOSE_TH = 2    # ally has <=2 cards -> try to assist
    _MY_CLOSE_TH = 8      # if I have few cards, can be more aggressive to finish

    # Penalties (tune if needed)
    _BREAK_BOMB_PEN = 250.0     # strong: discourage breaking bomb for normal types
    _ROCKET_PEN = 2000.0
    _STRAIGHT_FLUSH_PEN = 600.0
    _BOMB_BASE_PEN = 500.0      # base penalty for bombs; still allows bombs in danger cases

    def select_action(self, obs_for_player: dict) -> dict:
        candidate_actions = self.action_generator.generate_legal_actions(obs_for_player)
        if not candidate_actions:
            pid = int(obs_for_player.get('id', 0))
            return {'player': pid, 'action': [], 'claim': []}

        pass_action = next((act for act in candidate_actions if not act.get('claim')), candidate_actions[0])
        non_pass_actions = [act for act in candidate_actions if act.get('claim')]
        if not non_pass_actions:
            return pass_action

        pid = int(obs_for_player.get('id', 0))
        deck = obs_for_player.get('deck', []) or []
        last_move = obs_for_player.get('last_move', {}) or {}
        last_claim = last_move.get('claim', []) or []
        last_player = int(last_move.get('player', -1))

        starting_trick = (len(last_claim) == 0)

        # Inferred hand sizes
        hand_sizes = self._infer_hand_sizes(obs_for_player)
        ally = (pid + 2) % 4
        up = (pid - 1) % 4
        down = (pid + 1) % 4

        my_hand = hand_sizes[pid]
        ally_hand = hand_sizes[ally]
        enemy_min_hand = min(hand_sizes[up], hand_sizes[down])

        enemy_danger = (enemy_min_hand <= self._ENEMY_DANGER_TH)
        ally_close = (ally_hand <= self._ALLY_CLOSE_TH)
        my_close = (my_hand <= self._MY_CLOSE_TH)

        # Split bombs vs non-bombs
        non_bomb_actions = [a for a in non_pass_actions if not self._is_bomb_like(a)]
        bomb_actions = [a for a in non_pass_actions if self._is_bomb_like(a)]

        # ---- FOLLOW (responding) ----
        if not starting_trick:
            last_is_ally = (last_player >= 0 and ((pid - last_player) % 2 == 0))

            # If ally is controlling and enemy not dangerous, conserve (pass).
            if last_is_ally and not enemy_danger:
                # Exception: if ally is close to finishing, it can be good to keep control too.
                # Still pass in most cases, because we want ally to keep the lead.
                return pass_action

            # If we can beat with non-bomb, choose cheapest safe action (avoid breaking bombs).
            if non_bomb_actions:
                return self._pick_best(
                    obs_for_player,
                    non_bomb_actions,
                    context='follow',
                    enemy_danger=enemy_danger,
                    ally_close=ally_close,
                    my_close=my_close
                )

            # Otherwise, decide if bombing is necessary.
            if bomb_actions:
                # Bomb if enemy is dangerous or we are close to finishing.
                if enemy_danger or my_close:
                    return self._pick_best(
                        obs_for_player,
                        bomb_actions,
                        context='bomb',
                        enemy_danger=enemy_danger,
                        ally_close=ally_close,
                        my_close=my_close
                    )
                return pass_action

            return pass_action

        # ---- LEAD (starting a trick) ----
        # If ally is close to finishing, lead something that helps ally take control:
        # typically a small single/pair, but still avoid breaking bombs.
        if ally_close and non_bomb_actions:
            sp = [a for a in non_bomb_actions if self._type_of(a) in ['single', 'pair']]
            if sp:
                return self._pick_best(
                    obs_for_player,
                    sp,
                    context='lead_ally',
                    enemy_danger=enemy_danger,
                    ally_close=ally_close,
                    my_close=my_close
                )

        # Normal lead: shed cards (prefer multi-card) but do not break bombs.
        if non_bomb_actions:
            return self._pick_best(
                obs_for_player,
                non_bomb_actions,
                context='lead',
                enemy_danger=enemy_danger,
                ally_close=ally_close,
                my_close=my_close
            )

        # If only bombs exist, we may still lead a bomb if close to finishing; otherwise pass.
        if bomb_actions and my_close:
            return self._pick_best(
                obs_for_player,
                bomb_actions,
                context='bomb',
                enemy_danger=enemy_danger,
                ally_close=ally_close,
                my_close=my_close
            )

        return pass_action

    # -------------------------
    # Core scoring / selection
    # -------------------------
    def _pick_best(
        self,
        obs_for_player: dict,
        actions: List[dict],
        *,
        context: str,
        enemy_danger: bool,
        ally_close: bool,
        my_close: bool
    ) -> dict:
        deck = obs_for_player.get('deck', []) or []
        hand_rank_counts = self._rank_counts_from_cards(deck)

        def score(a: dict) -> Tuple[float, float, float, float]:
            claim = a.get('claim', []) or []
            t, pts = self.env._check_poker_type(claim)
            n_cards = len(claim)

            # (S0) Bomb-like base penalties
            bomb_pen = 0.0
            if t == 'rocket':
                bomb_pen = self._ROCKET_PEN
            elif t == 'straight_flush':
                bomb_pen = self._STRAIGHT_FLUSH_PEN
            elif t == 'bomb':
                # Encourage smaller bombs if bombing is needed.
                # pts typically (length, rank) in your env.
                try:
                    bomb_len = float(pts[0])
                except Exception:
                    bomb_len = float(n_cards)
                bomb_pen = self._BOMB_BASE_PEN + 30.0 * bomb_len

            # (S1) Breaking-bomb penalty (the key requirement)
            break_bomb_pen = 0.0
            if t not in ['bomb', 'straight_flush', 'rocket']:
                used_rank_counts = self._rank_counts_from_cards(claim)
                for r, hand_cnt in hand_rank_counts.items():
                    if hand_cnt >= 4:
                        used_cnt = used_rank_counts.get(r, 0)
                        if 0 < used_cnt < hand_cnt:
                            # This action uses part of a bomb rank -> breaks bomb.
                            # Strongly discourage unless absolutely needed.
                            break_bomb_pen += self._BREAK_BOMB_PEN

            # (S2) Card shedding preference
            # Lead: prefer more cards; Follow: prefer fewer cards (cheapest beat)
            if context in ['lead', 'lead_ally']:
                shed_term = -float(n_cards)
            elif context == 'bomb':
                shed_term = -0.2 * float(n_cards)  # mild preference to shed if bombing anyway
            else:
                shed_term = float(n_cards)

            # (S3) Strength/rank cost (cheaper ranks preferred when not in urgent danger)
            rank_cost = self._rank_cost(t, pts)

            # In urgent enemy danger, reduce rank cost weight (we care more about taking control).
            rank_w = 1.0
            if enemy_danger:
                rank_w = 0.4

            # (S4) If ally is close, avoid taking away control by playing too strong on lead
            ally_control_pen = 0.0
            if context == 'lead_ally':
                ally_control_pen = 0.6 * rank_cost

            # (S5) Bomb usage is more acceptable if enemy dangerous or I am close
            if context != 'bomb' and (enemy_danger or my_close):
                bomb_pen *= 0.6

            high_rank_pen = 0.0
            if context in ['lead', 'lead_ally']:
                ranks_used = self._ranks_from_cards(claim)
                level = str(obs_for_player.get('level', '2'))

                # Strongly protect level card and jokers; mildly protect A/K.
                if level in ranks_used:
                    high_rank_pen += 80.0
                if 'O' in ranks_used or 'o' in ranks_used:
                    high_rank_pen += 200.0
                if 'A' in ranks_used:
                    high_rank_pen += 30.0
                if 'K' in ranks_used:
                    high_rank_pen += 10.0

                # Extra penalty if it's a scale-type (straight / triple_pairs / three_straight),
                # because cardscale treats A as "small", which is misleading in resource terms.
                if hasattr(self.env, 'scaletypes') and t in self.env.scaletypes:
                    high_rank_pen *= 1.5

            total = bomb_pen + break_bomb_pen + rank_w * rank_cost + shed_term + ally_control_pen + high_rank_pen

            # Tie-breakers: fewer distinct ranks -> keep structure; then lower rank_cost; then more shed on lead
            distinct_ranks = float(len(set(self._ranks_from_cards(claim)))) if claim else 0.0
            return (total, distinct_ranks, rank_cost, -float(n_cards))

        return min(actions, key=score)

    # -------------------------
    # Utility methods
    # -------------------------
    def _type_of(self, action: dict) -> str:
        claim = action.get('claim', []) or []
        t, _ = self.env._check_poker_type(claim)
        return t

    def _is_bomb_like(self, action: dict) -> bool:
        t = self._type_of(action)
        return t in ['bomb', 'straight_flush', 'rocket']

    def _infer_hand_sizes(self, obs_for_player: dict) -> Dict[int, int]:
        '''
        Infer hand sizes for all 4 players.
        - self: exact len(deck)
        - others: 27 - total played cards in history by that player
        '''
        pid = int(obs_for_player.get('id', 0))
        deck = obs_for_player.get('deck', []) or []
        history = obs_for_player.get('history', []) or []

        played = {0: 0, 1: 0, 2: 0, 3: 0}
        for mv in history:
            p = int(mv.get('player', -1))
            act = mv.get('action', []) or []
            if 0 <= p < 4:
                played[p] += len(act)

        sizes = {0: 27 - played[0], 1: 27 - played[1], 2: 27 - played[2], 3: 27 - played[3]}
        sizes[pid] = len(deck)

        for k in sizes:
            if sizes[k] < 0:
                sizes[k] = 0
            if sizes[k] > 27:
                sizes[k] = 27
        return sizes

    def _rank_counts_from_cards(self, card_ids: List[int]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cid in card_ids:
            rank = self.utils.Num2Poker(int(cid))[1]
            counts[rank] = counts.get(rank, 0) + 1
        return counts

    def _ranks_from_cards(self, card_ids: List[int]) -> List[str]:
        return [self.utils.Num2Poker(int(cid))[1] for cid in card_ids]

    def _rank_cost(self, t: str, pts: tuple) -> float:
        '''
        Convert action strength to a comparable "cost" where smaller is cheaper.

        - For scaletypes: use env.cardscale index (A is smallest)
        - For other normal types: use env.point_order index
        - Bomb: combine length + rank (longer is stronger)
        - Rocket: huge
        '''
        if t == 'rocket':
            return 1e9

        if t == 'bomb':
            try:
                bomb_len = float(pts[0])
                r = pts[1]
                return 100.0 * bomb_len + float(self.env.point_order.index(r))
            except Exception:
                return 1e6

        if t == 'straight_flush':
            try:
                r = pts[0]
                return 2000.0 + float(self.env.cardscale.index(r))
            except Exception:
                return 3000.0

        if hasattr(self.env, 'scaletypes') and t in self.env.scaletypes:
            # A is smallest in cardscale
            r = pts[0] if pts else 'A'
            return float(self.env.cardscale.index(r))

        # normal types: point_order (level ordering)
        if pts:
            r = pts[0]
            return float(self.env.point_order.index(r))
        return 0.0
