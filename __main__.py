import json
import os
import sys
import types
import importlib
import torch

# Ensure local modules are importable both via short names and the
# original training package paths expected by rl_agent.py.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.action_generator import ActionGenerator
from agents.feature_encoder import FeatureEncoder
from models.policy_value_net import PolicyValueNet
from agents.rl_agent import RLGuanDanAgent
from env import GuanDanEnv
from utils import Utils


STATE = {
    'id': None,
    'level': '2',
    'deck': [],
    'history': [],
    'last_move': []
}

utils = Utils()
env = GuanDanEnv()
encoder = FeatureEncoder()
action_gen = ActionGenerator(env)

# Model dimensions derived from encoders.
STATE_DIM = 66
ACTION_DIM = 31

policy_net = PolicyValueNet(STATE_DIM, ACTION_DIM)
MODEL_PATH = '/data/GuanDan_001.pt'
try:
    loaded = torch.load(MODEL_PATH, map_location='cpu')
    if isinstance(loaded, dict):
        policy_net.load_state_dict(loaded)
except Exception:
    # Missing or incompatible model should not crash bot; fall back to random weights.
    pass
policy_net.eval()

agent = RLGuanDanAgent(
    policy_value_net=policy_net,
    feature_encoder=encoder,
    action_generator=action_gen,
    device='cpu'
)

def remove_cards_from_hand(cards):
    for c in cards:
        if c in STATE['deck']:
            STATE['deck'].remove(c)

def strength_index(card):
    rank = utils.Num2Poker(card)[1]
    try:
        return env.point_order.index(rank)
    except ValueError:
        return len(env.point_order)


def handle_deal(req):
    STATE['id'] = req.get('your_id', 0)
    STATE['hand'] = sorted(req.get('deliver', []))
    STATE['history'] = []
    global_info = req.get('global', {})

    STATE['level'] = global_info.get('level')
    env.reset({'level': STATE['level']})
    return []

def handle_play(req):
    pass_on = req.get('pass_on', -1)
    if pass_on == -1:
        STATE['last_move'] = {'player': -1, 'action': [], 'claim': []}

    short_history = req.get('history', None)
    if short_history:
        for record in short_history:
            if len(record) > 0:
                player_id = record.get('player', 0)
                res = record.get('response', [])
                if len(res) == 2:
                    action, claim = res
                    rec_converted = {
                        'player': player_id,
                        'action': action,
                        'claim': claim
                    }
                    STATE['history'].append(rec_converted)

                    if player_id == pass_on:
                        STATE['last_move'] = rec_converted

    obs = {
        'id': STATE['id'],
        'level': STATE['level'],
        'deck': sorted(STATE['deck']),
        'history': STATE['history'],
        'last_move': STATE['last_move']
    }

    selected = agent.select_action(obs, explore=False)
    action = selected.get('action', []) or []
    claim = selected.get('claim', []) or action
    remove_cards_from_hand(action)

    return [action, claim]


def process_request(raw_line):
    try:
        req = json.loads(raw_line)
    except Exception:
        return []

    stage = req.get('stage')
    if stage == 'deal':
        return handle_deal(req)
    if stage == 'tribute':
        pass
    if stage == 'return':
        pass
    if stage == 'play':
        return handle_play(req)
    return []


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = process_request(line)
        try:
            print(json.dumps(response, separators=(',', ':')))
        except Exception:
            print('[]')
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
