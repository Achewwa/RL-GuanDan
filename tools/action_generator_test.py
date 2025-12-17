import os
import random
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from agents.action_generator import ActionGenerator
from env import GuanDanEnv
from utils import Utils



def format_cards(utils: Utils, cards):
    """把一组牌的编号转成类似 'h3 d3 s4 ...' 的可读形式。"""
    if not cards:
        return 'PASS'
    
    return ' '.join(utils.Num2Poker(c) for c in sorted(cards, key=lambda x: x % 54))


def main(num_episodes: int = 20, fixed_level: str = 'Q'):
    env = GuanDanEnv()
    utils = Utils()

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'action_test.txt')

    with open(log_path, 'w', encoding='utf-8') as f:
        for ep in range(num_episodes):
            # 可以用固定级牌，也可以改成随机：
            # level = random.choice(['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A'])
            level = fixed_level

            obs = env.reset({'level': level})
            action_generator = ActionGenerator(env)

            # obs 是 {player_id: obs_for_player, ...} 这种结构，取第一个玩家
            first_player = list(obs.keys())[0]
            obs_player = obs[first_player]
            deck = obs_player.get('deck', [])

            f.write(f'=== Episode {ep} | level={level} | player={first_player} ===\n')
            f.write('Hand: ' + format_cards(utils, deck) + '\n')

            actions = action_generator.generate_legal_actions(obs_player)
            f.write(f'Total candidate actions: {len(actions)}\n')

            for idx, act in enumerate(actions):
                print(act)
                action = act.get('action', [])
                claim = act.get('claim', [])

                # 规则层检查
                legal = env._is_legal_claim(action, claim)
                
                flag = True
                for poker_no in action: 
                    if poker_no not in deck:
                        flag = False
                        break

                if action:
                    act_type, act_points = env._check_poker_type(action)
                else:
                    act_type, act_points = 'pass', ()

                if claim:
                    claim_type, claim_points = env._check_poker_type(claim)
                else:
                    claim_type, claim_points = 'pass', ()

                # 可视化这一手
                f.write(
                    f'  [{idx:03d}] '
                    f'action: {format_cards(utils, action):<30} '
                    f'claim: {format_cards(utils, claim):<30} '
                    f'legal={legal} '
                    f'a_type={act_type:<12} '
                    f'c_type={claim_type:<12}'
                    f'in_deck={flag}\n'
                )

                # 额外的错误提示（方便你定位）
                if not legal:
                    f.write('        >>> ERROR: not a legal claim according to env._is_legal_claim\n')

                if claim and claim_type == 'invalid':
                    f.write('        >>> ERROR: claim type is invalid\n')

                # 检查 action 是否只用了自己手里的牌
                if any(c not in deck for c in action):
                    f.write('        >>> ERROR: action contains cards not in player deck\n')

            f.write('\n')

    print(f'Action generator test log written to {log_path}')


if __name__ == '__main__':
    main()
