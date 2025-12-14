import os
import sys
import json
import random
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from env import GuanDanEnv
from utils import Utils
from agents.action_generator import ActionGenerator
from agents.feature_encoder import FeatureEncoder
from agents.rl_agent import RLGuanDanAgent
from models.policy_value_net import PolicyValueNet


def sort_ids_by_mod54(card_ids):
    return sorted(card_ids, key=lambda cid: cid % 54)


def build_policy_from_checkpoint(device: str):
    env_probe = GuanDanEnv()

    probe_level = '2'
    obs = env_probe.reset({'level': probe_level})
    action_generator = ActionGenerator(env_probe)
    feature_encoder = FeatureEncoder(env_probe)
    probe_player = list(obs.keys())[0]
    obs_player = obs[probe_player]

    state_vec = feature_encoder.encode_state(obs_player)
    candidate_actions = action_generator.generate_legal_actions(obs_player)
    if not candidate_actions:
        candidate_actions = [{'player': probe_player, 'action': [], 'claim': []}]
    action_vec = feature_encoder.encode_action(candidate_actions[0])

    state_dim = len(state_vec)
    action_dim = len(action_vec)

    policy_net = PolicyValueNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)

    path = 'models/ppo_checkpoint.pt'
    checkpoint_path = path if os.path.exists(path) else None
    if checkpoint_path is None:
        raise FileNotFoundError('Could not find PPO checkpoint at model/ppo_checkpoint.pt or models/ppo_checkpoint.pt')

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    return policy_net, feature_encoder, action_generator


def create_ppo_agents(env, policy_net, feature_encoder, action_generator, device: str):
    action_generator.env = env
    if hasattr(feature_encoder, 'env'):
        feature_encoder.env = env
    return [RLGuanDanAgent(policy_net, feature_encoder, action_generator, device=device) for _ in range(4)]


def _cards_to_strings(cards, utils: Utils):
    sorted_ids = sort_ids_by_mod54(cards)
    if hasattr(utils, 'id2strlist'):
        return utils.id2strlist(sorted_ids)
    readable = []
    for cid in sorted_ids:
        if hasattr(utils, 'id2str'):
            readable.append(utils.id2str(cid))
        else:
            readable.append(utils.Num2Poker(cid))
    return readable


def _format_claim(claim_cards, utils: Utils):
    claim_strings = _cards_to_strings(claim_cards, utils)
    return ' '.join(claim_strings) if claim_strings else 'pass'


def run_ppo_games(num_games: int, output_path: str, device: str):
    env = GuanDanEnv()
    utils = Utils()

    policy_net, feature_encoder, action_generator = build_policy_from_checkpoint(device)
    agents = create_ppo_agents(env, policy_net, feature_encoder, action_generator, device)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    levels = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']

    with open(output_path, 'w') as log_file:
        for game_idx in range(1, num_games + 1):
            level = random.choice(levels)
            obs = env.reset({'level': level})
            step_idx = 0
            log_file.write(f'=== Game {game_idx} | Level {level} ===\n')
            while True:
                if not obs:
                    break

                current_player = list(obs.keys())[0]
                obs_player = obs[current_player]

                deck = obs_player.get('deck', [])
                hand_strings = _cards_to_strings(deck, utils)
                last_move = obs_player.get('last_move', {})
                last_claim = last_move.get('claim', []) if last_move else []
                last_move_str = _format_claim(last_claim, utils)

                action = agents[current_player].select_action(obs_player, explore=False)
                claim_cards = action.get('claim', [])
                claim_str = _format_claim(claim_cards, utils)

                obs = env.step(action)
                step_idx += 1

                remaining = len(env.player_decks[current_player]) if hasattr(env, 'player_decks') else max(len(deck) - len(claim_cards), 0)

                log_file.write(f'Game {game_idx} | Step {step_idx} | Player {current_player}\n')
                log_file.write(f'  Hand before: {" ".join(hand_strings) if hand_strings else "empty"}\n')
                log_file.write(f'  Plays: {claim_str}\n')
                log_file.write(f'  Remaining cards: {remaining}\n')
                log_file.write(f'  Last table move: {last_move_str}\n\n')

                if env.done:
                    rewards = [env.reward.get(pid, 0) for pid in range(4)]
                    log_file.write(f'Final rewards: {json.dumps(rewards)}\n')
                    log_file.write(f'Game state: {env.game_state_info}\n\n')
                    log_file.flush()
                    break

            print(f'Finished game {game_idx}/{num_games}')
    print(f'Logs written to {output_path}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('logs', exist_ok=True)
    output_path = os.path.join('logs', 'ppo_play_logs.txt')
    run_ppo_games(num_games=5, output_path=output_path, device=device)
