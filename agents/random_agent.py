import random
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    '''Stochastic baseline agent for Guandan: uniformly random over legal actions.'''

    def select_action(self, obs_for_player: dict) -> dict:
        # 生成当前玩家的所有合法出牌（包含可能的 pass）
        candidate_actions = self.action_generator.generate_legal_actions(obs_for_player)

        # 极端保险：如果没有生成任何动作，就构造一个 pass
        if not candidate_actions:
            player_id = obs_for_player.get('id', 0)
            return {
                'player': player_id,
                'action': [],
                'claim': []
            }

        # 从候选动作中等概率随机选择一个
        action = random.choice(candidate_actions)

        # 确保带上 player 字段，和环境其他 agent 的约定一致
        if 'player' not in action:
            action = dict(action)
            action['player'] = obs_for_player.get('id', 0)

        return action
