import os
import sys
import json
import random
import argparse
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
from agents.rule_agent import RuleBasedAgent
from agents.random_agent import RandomAgent


def sort_ids_by_mod54(card_ids):
    return sorted(card_ids, key=lambda cid: cid % 54)


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


class _LocalRandomAgent:
    '''
    Fallback random agent if agents.random_agent.RandomAgent is not present.
    '''
    def __init__(self, env: GuanDanEnv, action_generator: ActionGenerator):
        self.env = env
        self.action_generator = action_generator

    def select_action(self, obs_for_player: dict) -> dict:
        candidate_actions = self.action_generator.generate_legal_actions(obs_for_player)
        if not candidate_actions:
            pid = int(obs_for_player.get('id', 0))
            return {'player': pid, 'action': [], 'claim': []}
        return random.choice(candidate_actions)


class _RLAdapter:
    '''
    Unify interface:
      - RL agent uses select_action(obs, explore=False)
      - Rule/Random use select_action(obs)
    '''
    def __init__(self, rl_agent: RLGuanDanAgent):
        self.rl_agent = rl_agent

    def select_action(self, obs_for_player: dict) -> dict:
        return self.rl_agent.select_action(obs_for_player, explore=False)


def build_policy_from_checkpoint(env_for_probe: GuanDanEnv, checkpoint_path: str, device: str):
    '''
    Build policy net with correct state/action dimensions by probing env once.
    '''
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    # Bind encoder & generator to probe env (IMPORTANT: your FeatureEncoder expects env)
    feature_encoder = FeatureEncoder(env_for_probe)
    action_generator = ActionGenerator(env_for_probe)

    # Probe observation
    obs = env_for_probe.reset({'level': '2'})
    probe_player = list(obs.keys())[0]
    obs_player = obs[probe_player]

    state_vec = feature_encoder.encode_state(obs_player)

    candidate_actions = action_generator.generate_legal_actions(obs_player)
    if not candidate_actions:
        candidate_actions = [{'player': probe_player, 'action': [], 'claim': []}]

    # encode_action requires obs + action in your latest encoder
    action_vec = feature_encoder.encode_action(obs_player, candidate_actions[0])

    state_dim = int(len(state_vec))
    action_dim = int(len(action_vec))

    policy_net = PolicyValueNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    return policy_net


def create_agent_by_type(
    agent_type: str,
    env: GuanDanEnv,
    policy_net,
    feature_encoder: FeatureEncoder,
    action_generator: ActionGenerator,
    device: str
):
    '''
    Create a seat agent according to agent_type: rl / rule / random.
    '''
    if agent_type == 'rl':
        rl_agent = RLGuanDanAgent(policy_net, feature_encoder, action_generator, device=device)
        return _RLAdapter(rl_agent)

    if agent_type == 'rule':
        # Most common signature in your project: RuleBasedAgent(env, action_generator)
        # If your RuleBasedAgent signature differs, adjust ONLY this line.
        return RuleBasedAgent(env)

    if agent_type == 'random':
        if RandomAgent is not None:
            # If your RandomAgent signature differs, adjust ONLY this line.
            return RandomAgent(env)
        return _LocalRandomAgent(env)

    raise ValueError(f'Unknown agent type: {agent_type}')


def run_games(
    num_games: int,
    output_path: str,
    device: str,
    level: str,
    seat_types: list,
    checkpoint_path: str
):
    '''
    Run games with mixed agents and log action traces.
    seat_types: list of 4 strings, each in {rl, rule, random}
    '''
    env = GuanDanEnv()
    utils = Utils()

    # Action generator + feature encoder must be bound to the SAME env instance used for stepping.
    action_generator = ActionGenerator(env)
    feature_encoder = FeatureEncoder(env)

    # If any seat is rl, we need a policy network
    policy_net = None
    if any(t == 'rl' for t in seat_types):
        # Use a separate probe env to infer dimensions (or reuse env; either is fine)
        env_probe = GuanDanEnv()
        policy_net = build_policy_from_checkpoint(env_probe, checkpoint_path, device)

    # Create 4 seat agents
    agents = []
    for seat_id in range(4):
        agents.append(
            create_agent_by_type(
                seat_types[seat_id],
                env,
                policy_net,
                feature_encoder,
                action_generator,
                device
            )
        )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as log_file:
        for game_idx in range(1, num_games + 1):
            obs = env.reset({'level': level})
            step_idx = 0
            log_file.write(f'=== Game {game_idx} | Level {level} | Seats {seat_types} ===\n')

            while True:
                if not obs:
                    break

                current_player = list(obs.keys())[0]
                obs_player = obs[current_player]

                deck = obs_player.get('deck', [])
                hand_strings = _cards_to_strings(deck, utils)

                last_move = obs_player.get('last_move', {}) or {}
                last_claim = last_move.get('claim', []) or []
                last_move_str = _format_claim(last_claim, utils)

                # Select action by seat type
                action = agents[current_player].select_action(obs_player)

                claim_cards = action.get('claim', []) or []
                claim_str = _format_claim(claim_cards, utils)

                obs = env.step(action)
                step_idx += 1

                # Remaining cards: prefer env.player_decks if exists
                if hasattr(env, 'player_decks'):
                    remaining = len(env.player_decks[current_player])
                else:
                    remaining = max(len(deck) - len(claim_cards), 0)

                log_file.write(f'Game {game_idx} | Step {step_idx} | Player {current_player} ({seat_types[current_player]})\n')
                log_file.write(f'  Hand before: {" ".join(hand_strings) if hand_strings else "empty"}\n')
                log_file.write(f'  Plays: {claim_str}\n')
                log_file.write(f'  Remaining cards: {remaining}\n')
                log_file.write(f'  Last table move: {last_move_str}\n\n')

                if getattr(env, 'done', False):
                    rewards = [env.reward.get(pid, 0) for pid in range(4)] if hasattr(env, 'reward') else [0, 0, 0, 0]
                    log_file.write(f'Final rewards: {json.dumps(rewards)}\n')
                    if hasattr(env, 'game_state_info'):
                        log_file.write(f'Game state: {env.game_state_info}\n')
                    if hasattr(env, 'cleared'):
                        log_file.write(f'Cleared order: {json.dumps(list(env.cleared))}\n')
                    if hasattr(env, 'round'):
                        log_file.write(f'Total turns (env.round): {env.round}\n')
                    log_file.write('\n')
                    log_file.flush()
                    break

            print(f'Finished game {game_idx}/{num_games}')

    print(f'Logs written to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Test PPO checkpoint vs Rule/Random agents with seat configuration.')
    parser.add_argument('--num-games', type=int, default=1, help='Number of games to run.')
    parser.add_argument('--level', type=str, default='2', help='Game level, default 2.')

    parser.add_argument('--p0', type=str, default='rl', choices=['rl', 'rule', 'random'], help='Seat 0 agent type.')
    parser.add_argument('--p1', type=str, default='rl', choices=['rl', 'rule', 'random'], help='Seat 1 agent type.')
    parser.add_argument('--p2', type=str, default='rl', choices=['rl', 'rule', 'random'], help='Seat 2 agent type.')
    parser.add_argument('--p3', type=str, default='rl', choices=['rl', 'rule', 'random'], help='Seat 3 agent type.')

    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path (required if any seat uses rl).')

    parser.add_argument('--output', type=str, default='logs/ppo_play_logs.txt', help='Output log file path.')
    parser.add_argument('--device', type=str, default='', help='Device: cuda/cpu. Default: auto.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = args.device.strip()
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seat_types = [args.p0, args.p1, args.p2, args.p3]

    # If any RL seat is used, checkpoint must be provided.
    if any(t == 'rl' for t in seat_types) and not args.checkpoint:
        raise ValueError('At least one seat is rl, but --checkpoint is not provided.')

    run_games(
        num_games=int(args.num_games),
        output_path=str(args.output),
        device=str(device),
        level=str(args.level),
        seat_types=seat_types,
        checkpoint_path=str(args.checkpoint)
    )
