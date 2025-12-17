from typing import Dict, List

from env import GuanDanEnv
from utils import Utils

whole_deck = range(54)

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
    
    @property
    def point_order(self):
        # 始终从当前环境读取 point_order，保证跟级牌保持同步
        return self.env.point_order

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
        
        # Unless the first player to act, always expose a pass action (environment will decide legality if used first-hand).
        if has_last:
            actions.append({'player': player_id, 'action': [], 'claim': []})

        rank_groups = self._group_by_rank(deck)
        
        # Check whether wild cards exist.
        # In an action, only use one wild card.
        wild_card = 'h' + self.env.level
        has_wild_card = any(self.utils.Num2Poker(c) == wild_card for c in deck)
        wild_card_id = self.utils.Poker2Num(wild_card, deck) if has_wild_card else None
        
        all_bombs = []

        straight_flush_mask = [False] * 108

        # Straight_flush:
        # mask the composition of straight_flushes
        for suit in self.env.suitset:
            for index in range(0, 10):
                if index == 9:
                    window = ['A', '0', 'J', 'Q', 'K']
                else:
                    window = [self.cardscale[i] for i in range(index, index + 5)]

                action = []
                missing_ranks = []

                for rank in window:
                    poker_str = suit + rank
                    card_id = self.utils.Poker2Num(poker_str, deck)
                    if card_id in deck:
                        action.append(card_id)
                    else:
                        missing_ranks.append(rank)

                
                if len(action) == 5:
                    all_bombs.append({
                        'player': player_id,
                        'action': action,
                        'claim': action,
                    })
                    for c in action:
                        straight_flush_mask[c] = True

                elif (
                    len(action) == 4
                    and has_wild_card
                    and not (suit == 'h' and self.env.level in window)
                ):
                    for c in action:
                        straight_flush_mask[c] = True

                    claim = []
                    for rank in window:
                        card_in_hand = self.utils.Poker2Num(suit + rank, deck)
                        if card_in_hand in deck:
                            claim.append(card_in_hand)
                        else:
                            claim.append(self.utils.Poker2Num(suit + rank, whole_deck))

                    action_with_wild = action + [wild_card_id]
                    all_bombs.append({
                        'player': player_id,
                        'action': action_with_wild,
                        'claim': claim,
                    })

                    straight_flush_mask[wild_card_id] = True
        
        # Bomb:
        for number, pokers in rank_groups.items():

            # For bombs, prefer using cards that are NOT part of any straight-flush candidate or wild cards.
            # i.e., avoid consuming cards marked by straight_flush_mask unless unavoidable.
            safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
            key_cards = [c for c in pokers if c not in safe_cards]

            # Count
            safe_cnt = len(safe_cards)
            tot_cnt = len(pokers)

            for bomb_len in range(4, tot_cnt + 1):

                # If we can form a bomb without touching straight-flush-critical cards, do that first.
                if safe_cnt >= bomb_len:
                    action = safe_cards[:bomb_len]
                    all_bombs.append({'player': player_id, 'action': action, 'claim': action})

                # Otherwise, we must include straight-flush-critical cards to form the bomb.
                else:
                    action = safe_cards + key_cards[:bomb_len - safe_cnt]
                    all_bombs.append({'player': player_id, 'action': action, 'claim': action})
            
            # use wild card only to extend the longest bomb.
            if has_wild_card and tot_cnt >= 3 and number != wild_card[1]:
                action = pokers + [wild_card_id]
                claim = pokers + [pokers[0]]
                all_bombs.append({'player': player_id, 'action': action, 'claim': claim})

        # Rocket: two small and two big jokers.
        if 'o' in rank_groups and 'O' in rank_groups:
            if len(rank_groups['o']) >= 2 and len(rank_groups['O']) >= 2:
                action = rank_groups['o'][:2] + rank_groups['O'][:2]
                all_bombs.append({'player': player_id, 'action': action, 'claim': action})

        # Add all the bombs that bigger than last_move to the current action list.
        for bomb in all_bombs:
            current_type, current_points = self.env._check_poker_type(bomb['claim'])
            if not has_last:
                actions.append(bomb)
            elif self.env._check_bigger(last_type, last_points, current_type, current_points):
                actions.append(bomb)
        
        # If last_action is a bomb, no need to check for normal-type actions.
        if has_last and last_type not in self.env.Normaltypes:
            return actions

        # Straight:
        if not has_last or last_type == 'straight':
            first_idx = self.cardscale.index(last_points[0]) + 1 if has_last else 0
            window = []
            for index in range(first_idx, 10):
                if index == 9:
                    window = ['A', '0', 'J', 'Q', 'K']
                else:
                    window = [self.cardscale[i] for i in range(index, index + 5)]
                action = []
                missing_numbers = []
                for number in window:
                    if number not in rank_groups:
                        missing_numbers.append(number)
                    else:
                        pokers = rank_groups[number]
                        safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                        if len(safe_cards) >= 1:
                            action.append(safe_cards[0])
                        else:
                            action.append(pokers[0])
                if len(action) == 5:
                    actions.append({'player': player_id, 'action': action, 'claim': action})
                elif len(action) == 4 and has_wild_card and wild_card_id not in action:
                    claim = action + [self.utils.Poker2Num('h' + missing_numbers[0], whole_deck)]
                    action = action + [wild_card_id]
                    actions.append({'player': player_id, 'action': action, 'claim': claim})

        # Three_straight:
        if not has_last or last_type == 'three_straight':
            first_idx = self.cardscale.index(last_points[0]) + 1 if has_last else 0
            window = []
            for index in range(first_idx, 13):
                if index == 12:
                    window = ['A', 'K']
                else:
                    window = [self.cardscale[i] for i in range(index, index + 2)]
                action = []
                missing_numbers = []
                for number in window:
                    if number not in rank_groups:
                        missing_numbers.append(number)
                    else:
                        pokers = rank_groups[number]
                        safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                        key_cards = [c for c in pokers if c not in safe_cards]
                        safe_len = len(safe_cards)
                        tot_len = len(pokers)
                        if safe_len >= 3:
                            action += safe_cards[:3]
                        elif tot_len >= 3:
                            action += safe_cards + key_cards[:3 - safe_len]
                        else:
                            action += pokers
                            missing_numbers.append(number)

                if len(action) == 6:
                    actions.append({'player': player_id, 'action': action, 'claim': action})
                elif len(action) == 5 and has_wild_card and wild_card_id not in action:
                    claim = action + [self.utils.Poker2Num('h' + missing_numbers[0], whole_deck)]
                    action = action + [wild_card_id]
                    actions.append({'player': player_id, 'action': action, 'claim': claim})
        
        # Triple_pairs:
        if not has_last or last_type == 'triple_pairs':
            first_idx = self.cardscale.index(last_points[0]) + 1 if has_last else 0
            window = []
            for index in range(first_idx, 12):
                if index == 11:
                    window = ['A', 'Q', 'K']
                else:
                    window = [self.cardscale[i] for i in range(index, index + 3)]
                action = []
                missing_numbers = []
                for number in window:
                    if number not in rank_groups:
                        missing_numbers.append(number)
                    else:
                        pokers = rank_groups[number]
                        safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                        key_cards = [c for c in pokers if c not in safe_cards]
                        safe_len = len(safe_cards)
                        tot_len = len(pokers)
                        if safe_len >= 2:
                            action += safe_cards[:2]
                        elif tot_len >= 2:
                            action += safe_cards + key_cards[:2 - safe_len]
                        else:
                            action += pokers
                            missing_numbers.append(number)

                if len(action) == 6:
                    actions.append({'player': player_id, 'action': action, 'claim': action})
                elif len(action) == 5 and has_wild_card and wild_card_id not in action:
                    claim = action + [self.utils.Poker2Num('h' + missing_numbers[0], whole_deck)]
                    action = action + [wild_card_id]
                    actions.append({'player': player_id, 'action': action, 'claim': claim})

        # Three:
        if not has_last or last_type == 'three':
            first_idx = self.point_order.index(last_points[0]) + 1 if has_last else 0
            for index in range(first_idx, 13):
                action = []
                number = self.point_order[index]
                if number in rank_groups:
                    pokers = rank_groups[number]
                    safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                    key_cards = [c for c in pokers if c not in safe_cards]
                    safe_len = len(safe_cards)
                    tot_len = len(pokers)
                    
                    if safe_len >= 3:
                        action += safe_cards[:3]
                    elif tot_len >= 3:
                        action += safe_cards + key_cards[:3 - safe_len]
                    else:
                        action += pokers
                    
                    if len(action) == 3:
                        actions.append({'player': player_id, 'action': action, 'claim': action})
                    elif len(action) == 2 and has_wild_card and wild_card_id not in action:
                        claim = action + [action[0]]
                        action += [wild_card_id]
                        actions.append({'player': player_id, 'action': action, 'claim': claim})


        # Pair
        if not has_last or last_type == 'pair':
            first_idx = self.point_order.index(last_points[0]) + 1 if has_last else 0
            for index in range(first_idx, 15):
                action = []
                number = self.point_order[index]
                if number in rank_groups:
                    pokers = rank_groups[number]
                    safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                    key_cards = [c for c in pokers if c not in safe_cards]
                    safe_len = len(safe_cards)
                    tot_len = len(pokers)
                    
                    if safe_len >= 2:
                        action += safe_cards[:2]
                    elif tot_len >= 2:
                        action += safe_cards + key_cards[:2 - safe_len]
                    else:
                        action += pokers
                    
                    if len(action) == 2:
                        actions.append({'player': player_id, 'action': action, 'claim': action})
                    elif len(action) == 1 and has_wild_card and wild_card_id not in action and index < 13:
                        claim = action + action
                        action += [wild_card_id]
                        actions.append({'player': player_id, 'action': action, 'claim': claim})

        # Single
        if not has_last or last_type == 'single':
            first_idx = self.point_order.index(last_points[0]) + 1 if has_last else 0

            for index in range(first_idx, 15):
                action = []
                number = self.point_order[index]
                if number in rank_groups:
                    pokers = rank_groups[number]
                    safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                    if len(safe_cards) >= 1:
                        action = [safe_cards[0]]
                        actions.append({'player': player_id, 'action': action, 'claim': action})

        # Set:
        if not has_last or last_type == 'set':
            first_idx = self.point_order.index(last_points[0]) + 1 if has_last else 0
            for index in range(first_idx, 13):
                action = []
                number = self.point_order[index]
                if number in rank_groups:
                    pokers = rank_groups[number]
                    safe_cards = [c for c in pokers if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                    key_cards = [c for c in pokers if c not in safe_cards]
                    safe_len = len(safe_cards)
                    tot_len = len(pokers)
                    if tot_len >= 3:
                        if safe_len >= 3:
                            action += safe_cards[:3]
                        else:
                            action += safe_cards + key_cards[:3 - safe_len]
                        
                        # Find the smallest pure pair, prefer using cards that are NOT part of any straight-flush candidate or wild cards. 
                        pairs = []
                        for pair_index, pair_number in enumerate(self.point_order):
                            if pair_number in rank_groups and pair_number != number:
                                pokers2 = rank_groups[pair_number]
                                safe_cards2 = [c for c in pokers2 if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                                if len(safe_cards2) == 2:
                                    pairs += safe_cards2
                                    break
                        
                        # If there is no pure pair, try using the wild card.
                        flag = False
                        if len(pairs) == 0 and has_wild_card and number != wild_card[1]:
                            for pair_index, pair_number in enumerate(self.point_order):
                                if pair_number in rank_groups and pair_number != number and pair_number != wild_card[1]:
                                    pokers2 = rank_groups[pair_number]
                                    safe_cards2 = [c for c in pokers2 if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                                    if len(safe_cards2) == 1:
                                        pairs += safe_cards2
                                        flag = True
                                        break
                        if len(pairs) != 0:
                            if not flag:
                                action += pairs
                                actions.append({'player': player_id, 'action': action, 'claim': action})
                            else:
                                claim = action + pairs + pairs
                                action += pairs + [wild_card_id]
                                actions.append({'player': player_id, 'action': action, 'claim': claim})

                    elif tot_len == 2 and has_wild_card and number != wild_card[1]:
                        # Try using the wild card to form the three.
                        # Find the pure pairs.
                        action = pokers[:]
                        pairs = []
                        for pair_index, pair_number in enumerate(self.point_order):
                            if pair_number in rank_groups and pair_number != number and pair_number != wild_card[1]:
                                pokers2 = rank_groups[pair_number]
                                safe_cards2 = [c for c in pokers2 if not straight_flush_mask[c] or self.utils.Num2Poker(c) == wild_card]
                                if len(safe_cards2) == 2:
                                    pairs += safe_cards2
                                    break
                        if len(pairs) == 2:
                            claim = [action[0]] + action + pairs
                            action += [wild_card_id] + pairs
                            actions.append({'player': player_id, 'action': action, 'claim': claim})

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

