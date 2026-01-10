# botzone_main.py
# Long-running Botzone runner (deal/play only).
# Fixes the key issue you identified: play.history is only a short window.
# We maintain a full history with robust overlap-stitching and compute last_move
# via a proper "table state machine" (non-pass + pass clearing) that respects
# active players (excluding done players).

import json
import os
import sys
import torch

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


# =========================
# Global singletons (long-running)
# =========================

utils = Utils()
env = GuanDanEnv()
encoder = FeatureEncoder(env)
action_gen = ActionGenerator(env)

# Botzone model path
MODEL_PATH = '/data/6869_1.pt'

# Persistent agent/net (loaded once)
policy_net = None
agent = None

# Global state across requests
STATE = {
    'id': None,
    'level': '2',
    'deck': [],

    # Full chronological history from deal to now:
    # each item: {'player': int, 'action': [int], 'claim': [int]}
    'history_full': [],

    # Players finished (from req['done'])
    'done': set(),

    # Table state (the actual last non-pass play to follow, if any)
    'table_last_move': {'player': -1, 'action': [], 'claim': []},
    'table_last_player': -1,     # player id of last non-pass
    'table_pass_set': set(),     # players who passed since last non-pass
}


# =========================
# Utilities
# =========================

def _derive_model_dims() -> tuple:
    """
    Derive (state_dim, action_dim) from encoder/action generator once.
    This avoids hard-coded dims that can silently mismatch after feature changes.
    """
    obs = env.reset({'level': '2'})
    first_player = int(list(obs.keys())[0])
    obs_player = obs[first_player]

    state_vec = encoder.encode_state(obs_player)
    cand = action_gen.generate_legal_actions(obs_player)
    if not cand:
        cand = [{'player': first_player, 'action': [], 'claim': []}]
    action_vec = encoder.encode_action(obs_player, cand[0])

    return len(state_vec), len(action_vec)


def _load_policy_once():
    """
    Create policy_net / agent once. If load fails, we still keep a net,
    but we DO NOT silently swallow shape mismatch without any output.
    """
    global policy_net, agent

    state_dim, action_dim = _derive_model_dims()
    policy_net = PolicyValueNet(state_dim, action_dim, hidden_dim=128)

    try:
        loaded = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(loaded, dict):
            policy_net.load_state_dict(loaded)
        else:
            # Some users save full module; handle cautiously
            policy_net = loaded
    except Exception as e:
        # Print to stderr so you can see it in Botzone logs (do not crash).
        print(f'[WARN] Failed to load model from {MODEL_PATH}: {repr(e)}', file=sys.stderr)

    policy_net.eval()

    agent = RLGuanDanAgent(
        policy_value_net=policy_net,
        feature_encoder=encoder,
        action_generator=action_gen,
        device='cpu'
    )


def _remove_cards_from_deck(cards):
    # Remove played cards from our local deck.
    # action cards are actual removed cards; claim is declarative.
    for c in cards:
        if c in STATE['deck']:
            STATE['deck'].remove(c)


def _convert_botzone_history_to_moves(history_list):
    """
    Convert Botzone history list to our internal move dicts.
    Each record: {'player': pid, 'response': [action, claim]}
    Return list of dicts in chronological order:
      {'player': pid, 'action': [...], 'claim': [...]}
    """
    moves = []
    if not history_list:
        return moves

    for record in history_list:
        if not isinstance(record, dict) or not record:
            continue
        pid = int(record.get('player', 0))
        res = record.get('response', [])
        if not (isinstance(res, list) and len(res) == 2):
            continue
        action, claim = res
        action = action or []
        claim = claim or []
        moves.append({'player': pid, 'action': action, 'claim': claim})
    return moves


def _moves_to_tuples(moves):
    """
    For overlap-matching.
    """
    out = []
    for m in moves:
        out.append((int(m['player']), tuple(m.get('action', []) or []), tuple(m.get('claim', []) or [])))
    return out


def _append_history_with_overlap(full_history, delta_moves):
    """
    Stitch delta_moves onto full_history by max suffix-prefix overlap.

    Botzone play.history is a short window. It can overlap with what we already stored.
    We compute the largest k such that:
        full_history[-k:] == delta_moves[:k]
    then append delta_moves[k:].
    Return: (num_appended, appended_moves_list)
    """
    if not delta_moves:
        return 0, []

    full_t = _moves_to_tuples(full_history)
    delta_t = _moves_to_tuples(delta_moves)

    max_k = min(len(full_t), len(delta_t))
    k = 0
    # find max overlap
    for kk in range(max_k, 0, -1):
        if full_t[-kk:] == delta_t[:kk]:
            k = kk
            break

    to_add = delta_moves[k:]
    full_history.extend(to_add)
    return len(to_add), to_add


def _active_players():
    """
    Players still in the hand (not finished), based on STATE['done'].
    """
    return [p for p in [0, 1, 2, 3] if p not in STATE['done']]


def _table_reset():
    STATE['table_last_move'] = {'player': -1, 'action': [], 'claim': []}
    STATE['table_last_player'] = -1
    STATE['table_pass_set'] = set()


def _update_table_state_with_move(move):
    """
    Update table_last_move according to GuanDan trick logic:
      - Non-pass sets the table_last_move and clears pass_set.
      - Pass adds to pass_set.
      - When all OTHER active players have passed since last non-pass, clear table (new lead).
    This is the critical fix to construct correct obs['last_move'].
    """
    pid = int(move['player'])
    action = move.get('action', []) or []
    claim = move.get('claim', []) or []

    # If action is non-empty => new table card
    if len(action) > 0:
        STATE['table_last_move'] = {'player': pid, 'action': action, 'claim': claim}
        STATE['table_last_player'] = pid
        STATE['table_pass_set'] = set()
        return

    # Pass: only meaningful if there is a table card to follow
    if STATE['table_last_player'] < 0:
        return

    # Record pass
    STATE['table_pass_set'].add(pid)

    # If all other active players (except last_player) have passed, reset for new lead
    active = set(_active_players())
    lastp = int(STATE['table_last_player'])

    others = active - {lastp}
    if others and others.issubset(STATE['table_pass_set']):
        _table_reset()


# =========================
# Handlers
# =========================

def handle_deal(req):
    # Reset per-game state
    STATE['id'] = int(req.get('your_id', 0))
    STATE['deck'] = sorted(req.get('deliver', []) or [])
    STATE['history_full'] = []
    STATE['done'] = set()
    _table_reset()

    global_info = req.get('global', {}) or {}
    STATE['level'] = global_info.get('level', '2')

    # Keep env consistent with level (you said fixed=2, but we still honor input).
    env.reset({'level': STATE['level']})

    return []


def handle_play(req):
    # Update done players (may be empty)
    done_list = req.get('done', []) or []
    # done is a list of player ids already out
    STATE['done'] = set(int(x) for x in done_list) if isinstance(done_list, list) else set()

    # Convert delta history
    bot_hist = req.get('history', []) or []
    delta_moves = _convert_botzone_history_to_moves(bot_hist)

    # Stitch into full history (robust overlap)
    _, appended = _append_history_with_overlap(STATE['history_full'], delta_moves)

    # Update table state by processing only newly appended moves (in chronological order)
    for m in appended:
        _update_table_state_with_move(m)

    # Build observation for our agent
    obs = {
        'id': STATE['id'],
        'level': STATE['level'],
        'deck': sorted(STATE['deck']),
        'history': STATE['history_full'],
        'last_move': STATE['table_last_move']
    }

    # Select action
    selected = agent.select_action(obs, explore=False)

    action = selected.get('action', []) or []
    claim = selected.get('claim', None)

    if claim is None or claim == []:
        claim = action

    _remove_cards_from_deck(action)

    return [action, claim]


# =========================
# Botzone request loop
# =========================

def process_request(raw_line):
    try:
        req = json.loads(raw_line)
    except Exception:
        return []

    if isinstance(req, dict) and 'requests' in req:
        req = req['requests'][0]

    stage = req.get('stage')
    if stage == 'deal':
        return handle_deal(req)
    if stage == 'play':
        return handle_play(req)
    return []


def main():
    # Load model and build agent once (long-running mode)
    _load_policy_once()

    while True:
        line = input()
        if not line:
            continue
        response = process_request(line)
        print(json.dumps({'response': response}))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
