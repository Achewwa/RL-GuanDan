import os
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.action_generator import ActionGenerator
from agents.feature_encoder import FeatureEncoder
from agents.rl_agent import RLGuanDanAgent
from agents.base_agent import BaseAgent
from agents.rule_agent import RuleBasedAgent
from agents.random_agent import RandomAgent
from env import GuanDanEnv
from models.policy_value_net import PolicyValueNet
from selfplay.replay_buffer import ReplayBuffer, Transition

TRAIN_LEVELS = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
train_all_levels = True
fixed_level = '2'


def make_shared_components(device: str) -> Tuple[GuanDanEnv, FeatureEncoder, ActionGenerator, PolicyValueNet, int, int]:
    env = GuanDanEnv()

    level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
    obs = env.reset({'level': level})

    action_generator = ActionGenerator(env)
    feature_encoder = FeatureEncoder(env)

    first_player = list(obs.keys())[0]
    obs_player = obs[first_player]
    state_vec = feature_encoder.encode_state(obs_player)

    candidate_actions = action_generator.generate_legal_actions(obs_player)
    if not candidate_actions:
        candidate_actions = [{'player': first_player, 'action': [], 'claim': []}]

    # IMPORTANT BUGFIX: encode_action(obs_for_player, action), not reversed
    action_vec = feature_encoder.encode_action(obs_player, candidate_actions[0])

    state_dim = len(state_vec)
    action_dim = len(action_vec)

    policy_value_net = PolicyValueNet(state_dim, action_dim, hidden_dim=128).to(device)
    return env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim


def create_rl_agents(env: GuanDanEnv, net: PolicyValueNet, feature_encoder: FeatureEncoder, action_generator: ActionGenerator, device: str) -> List[RLGuanDanAgent]:
    return [RLGuanDanAgent(net, feature_encoder, action_generator, device=device) for _ in range(4)]


# =========================
# Mixed opponent utilities
# =========================

def choose_opponent_mode(update_idx: int, warm_up: int, has_past_ckpt: bool) -> str:
    '''
    Return one of: 'rule', 'self', 'past', 'random'.

    Schedule:
      - update_idx < warm_up:
          80% rule, 15% self, 5% random
      - else:
          10% rule, 60% self, 25% past, 5% random
        If no past checkpoints available, reallocate that mass into self-play.
    '''
    if update_idx < warm_up:
        r = random.random()
        if r < 0.4:
            return 'rule'
        if r < 0.95:
            return 'self'
        return 'random'

    # after warmup
    if not has_past_ckpt:
        # 10% rule, 85% self, 5% random
        r = random.random()
        if r < 0.10:
            return 'rule'
        if r < 0.95:
            return 'self'
        return 'random'

    r = random.random()
    if r < 0.10:
        return 'rule'
    if r < 0.70:
        return 'self'
    if r < 0.95:
        return 'past'
    return 'random'


def load_checkpoint_net(
    ckpt_path: str,
    *,
    device: str,
    state_dim: int,
    action_dim: int
) -> PolicyValueNet:
    '''
    Build a PolicyValueNet and load weights from ckpt_path.
    Used to create "past RL checkpoint" opponents.

    Note: hidden_dim must match training.
    '''
    net = PolicyValueNet(state_dim, action_dim, hidden_dim=128).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(sd)
    net.eval()
    return net


def build_episode_agents(
    *,
    env: GuanDanEnv,
    feature_encoder: FeatureEncoder,
    action_generator: ActionGenerator,
    current_net: PolicyValueNet,
    state_dim: int,
    action_dim: int,
    device: str,
    mode: str,
    checkpoint_pool: List[str]
) -> Tuple[Dict[int, RLGuanDanAgent], Dict[int, BaseAgent]]:
    '''
    Build agent assignments for one episode.

    Requirement:
      - Two teams, each controlled by two same-type agents.
      - Training team is always current RL (two seats).
      - Opponent team depends on mode:
          'rule'   -> RuleBasedAgent
          'random' -> RandomAgent
          'self'   -> current RL as opponent too (same net)
          'past'   -> frozen RL from a random past checkpoint

    Also randomize "which side starts":
      - env fixed player0 starts.
      - Randomly choose training team seats:
          (0,2) or (1,3)
        So player0 sometimes is training, sometimes opponent.

    Returns:
      train_team: {pid -> RLGuanDanAgent} for the two training players
      opp_team:   {pid -> BaseAgent} for the two opponent players
    '''
    # Randomize which team includes player0 (thereby randomizing which side starts)
    train_on_even = (random.random() < 0.5)
    train_pids = [0, 2] if train_on_even else [1, 3]
    opp_pids = [1, 3] if train_on_even else [0, 2]

    # Training agents (current policy) for train_pids
    train_team: Dict[int, RLGuanDanAgent] = {}
    for pid in train_pids:
        train_team[pid] = RLGuanDanAgent(current_net, feature_encoder, action_generator, device=device)

    # Opponent agents (same type for both opp seats)
    opp_team: Dict[int, BaseAgent] = {}

    if mode == 'rule':
        opp_a0 = RuleBasedAgent(env)
        opp_a1 = RuleBasedAgent(env)
        opp_team[opp_pids[0]] = opp_a0
        opp_team[opp_pids[1]] = opp_a1
        return train_team, opp_team

    if mode == 'random':
        opp_a0 = RandomAgent(env)
        opp_a1 = RandomAgent(env)
        opp_team[opp_pids[0]] = opp_a0
        opp_team[opp_pids[1]] = opp_a1
        return train_team, opp_team

    if mode == 'self':
        # Opponent is same (current) RL policy; still "two same-type agents"
        opp_team[opp_pids[0]] = RLGuanDanAgent(current_net, feature_encoder, action_generator, device=device)
        opp_team[opp_pids[1]] = RLGuanDanAgent(current_net, feature_encoder, action_generator, device=device)
        return train_team, opp_team

    if mode == 'past':
        # Sample a past checkpoint and create frozen RL opponents
        ckpt_path = random.choice(checkpoint_pool)
        past_net = load_checkpoint_net(
            ckpt_path,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim
        )
        opp_team[opp_pids[0]] = RLGuanDanAgent(past_net, feature_encoder, action_generator, device=device)
        opp_team[opp_pids[1]] = RLGuanDanAgent(past_net, feature_encoder, action_generator, device=device)
        return train_team, opp_team

    # Fallback
    opp_a0 = RuleBasedAgent(env)
    opp_a1 = RuleBasedAgent(env)
    opp_team[opp_pids[0]] = opp_a0
    opp_team[opp_pids[1]] = opp_a1
    return train_team, opp_team


# =========================
# Reward shaping helpers (unchanged)
# =========================

def compute_speed_bonuses(env: GuanDanEnv, *, t_ref: float = 100.0, alpha: float = 0.5, beta: float = 0.25) -> dict:
    bonuses = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    if not getattr(env, 'done', False):
        return bonuses
    cleared = list(getattr(env, 'cleared', []))
    if len(cleared) < 1:
        return bonuses
    T = float(getattr(env, 'round', 0.0))
    speed_score = (t_ref - T) / t_ref
    if speed_score < 0.0:
        speed_score = 0.0
    elif speed_score > 1.0:
        speed_score = 1.0

    first_finisher = int(cleared[0])
    if len(cleared) == 2:
        teammate_finisher = int(cleared[1])
    elif len(cleared) >= 3 and ((cleared[2] - cleared[0]) % 2 == 0):
        teammate_finisher = int(cleared[2])
    else:
        teammate_finisher = int((cleared[0] + 2) % 4)

    bonuses[first_finisher] += alpha * speed_score
    if teammate_finisher != first_finisher:
        bonuses[teammate_finisher] += beta * speed_score
    return bonuses


def apply_terminal_rewards_to_last_steps(transitions_by_player: dict, terminal_rewards: dict) -> None:
    for pid, traj in transitions_by_player.items():
        if not traj:
            continue
        traj[-1].reward += float(terminal_rewards.get(pid, 0.0))
        traj[-1].done = True


# =========================
# Modified rollout collector (supports mixed opponents)
# =========================

def collect_rollout(
    env: GuanDanEnv,
    current_net: PolicyValueNet,
    feature_encoder: FeatureEncoder,
    action_generator: ActionGenerator,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    device: str,
    *,
    update_idx: int,
    warm_up: int,
    checkpoint_pool: List[str],
    state_dim: int,
    action_dim: int
) -> None:
    '''
    Collect rollouts for PPO with mixed opponents.

    Key changes:
      - Each episode samples an opponent mode with the required probabilities.
      - Each episode randomly decides whether training team is (0,2) or (1,3)
        to randomize which side has player0 (and thus who starts).
      - Only transitions for training team players are recorded (two players).
      - Terminal rewards are patched onto each training player's last transition.
    '''
    has_past = (len(checkpoint_pool) > 0)

    for episode_id in range(num_episodes):
        level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
        obs = env.reset({'level': level})

        mode = choose_opponent_mode(update_idx, warm_up, has_past_ckpt=has_past)

        train_team, opp_team = build_episode_agents(
            env=env,
            feature_encoder=feature_encoder,
            action_generator=action_generator,
            current_net=current_net,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            mode=mode,
            checkpoint_pool=checkpoint_pool
        )

        # Only keep trajectories for training team players
        transitions_by_player = {pid: [] for pid in train_team.keys()}

        while True:
            if not obs:
                break

            current_player = int(list(obs.keys())[0])
            obs_player = obs[current_player]

            if current_player in train_team:
                # --- training policy samples with exploration ---
                agent = train_team[current_player]
                candidate_actions, probabilities, log_probabilities, state_vec, action_vecs = agent._compute_action_distribution(
                    obs_player, explore=True
                )
                action_idx = torch.multinomial(probabilities, 1).item()
                chosen_action = candidate_actions[action_idx]
                chosen_log_prob = log_probabilities[action_idx].detach()

                # env step
                obs = env.step(chosen_action)
                done = bool(env.done)

                # per-step reward is 0; terminal rewards patched later
                transition = Transition(
                    episode_id=int(episode_id),
                    player_id=int(current_player),
                    state=np.asarray(state_vec, dtype=np.float32),
                    candidate_action_feats=np.asarray(action_vecs, dtype=np.float32),
                    action_index=int(action_idx),
                    reward=0.0,
                    done=False,
                    log_prob=float(chosen_log_prob)
                )
                transitions_by_player[current_player].append(transition)

            else:
                # --- opponent acts (no data collected) ---
                opp_agent = opp_team[current_player]
                # For RL opponents (self/past), use explore=False to keep them stable.
                if isinstance(opp_agent, RLGuanDanAgent):
                    action = opp_agent.select_action(obs_player, explore=False)
                else:
                    action = opp_agent.select_action(obs_player)
                obs = env.step(action)
                done = bool(env.done)

            if done:
                base_terminal = dict(env.reward) if isinstance(env.reward, dict) else {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                speed_bonus = compute_speed_bonuses(env, t_ref=100.0, alpha=0.5, beta=0.25)
                terminal_rewards = {pid: float(base_terminal.get(pid, 0.0)) + float(speed_bonus.get(pid, 0.0)) for pid in range(4)}

                # Patch terminal reward ONLY onto training players' last transitions
                apply_terminal_rewards_to_last_steps(transitions_by_player, terminal_rewards)

                for pid, traj in transitions_by_player.items():
                    for t in traj:
                        replay_buffer.add_transition(t)
                break


# =========================
# PPO / value / eval (unchanged from yours)
# =========================

def compute_returns_and_advantages_grouped(
    transitions: List[Transition],
    values: torch.Tensor,
    gamma: float,
    lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(transitions)
    returns = torch.zeros(n, dtype=torch.float32, device=values.device)
    advantages = torch.zeros(n, dtype=torch.float32, device=values.device)

    groups = {}
    for idx, t in enumerate(transitions):
        key = (int(t.episode_id), int(t.player_id))
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    for _, idxs in groups.items():
        gae = 0.0
        next_value = 0.0
        for k in reversed(idxs):
            done = 1.0 if transitions[k].done else 0.0
            reward = float(transitions[k].reward)
            mask = 1.0 - done
            delta = reward + gamma * next_value * mask - float(values[k].item())
            gae = delta + gamma * lam * mask * gae
            advantages[k] = gae
            returns[k] = advantages[k] + values[k]
            next_value = float(values[k].item())

    return returns, advantages


def compute_state_values(policy_value_net: PolicyValueNet, transitions: List[Transition], device: str) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for t in transitions:
            state_tensor = torch.tensor(t.state, dtype=torch.float32, device=device)
            action_tensor = torch.tensor(t.candidate_action_feats, dtype=torch.float32, device=device)
            state_batch = state_tensor.unsqueeze(0).repeat(action_tensor.shape[0], 1)
            _, vals = policy_value_net(state_batch, action_tensor)
            values.append(vals[0, 0].detach().item())
    return torch.tensor(values, dtype=torch.float32, device=device)


def ppo_update(
    policy_value_net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    epochs: int
) -> None:
    num_samples = len(transitions)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = torch.randperm(num_samples)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for idx in indices:
            t = transitions[int(idx)]
            ret = returns[int(idx)]
            adv = advantages[int(idx)]

            state_tensor = torch.tensor(t.state, dtype=torch.float32, device=ret.device)
            action_tensor = torch.tensor(t.candidate_action_feats, dtype=torch.float32, device=ret.device)
            state_batch = state_tensor.unsqueeze(0).repeat(action_tensor.shape[0], 1)

            logits, values = policy_value_net(state_batch, action_tensor)
            logits = logits.squeeze(-1)
            log_probs_new = F.log_softmax(logits, dim=0)
            probs_new = log_probs_new.exp()

            new_log_prob = log_probs_new[t.action_index]
            v_s = values[0, 0]
            old_log_prob = torch.tensor(t.log_prob, dtype=torch.float32, device=ret.device)

            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2)

            value_loss = F.mse_loss(v_s, ret)
            entropy = -(probs_new * log_probs_new).sum()

            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss = total_value_loss + value_loss
            total_entropy = total_entropy + entropy

        total_policy_loss = total_policy_loss / num_samples
        total_value_loss = total_value_loss / num_samples
        total_entropy = total_entropy / num_samples

        loss = total_policy_loss + value_loss_coef * total_value_loss - entropy_coef * total_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_value_net.parameters(), max_norm=0.5)
        optimizer.step()


def evaluate_policy(env: GuanDanEnv, opponent_agent: BaseAgent, policy_value_net: PolicyValueNet, feature_encoder: FeatureEncoder, action_generator: ActionGenerator, num_episodes: int, device: str) -> float:
    rl_agents = [
        RLGuanDanAgent(policy_value_net, feature_encoder, action_generator, device=device),
        RLGuanDanAgent(policy_value_net, feature_encoder, action_generator, device=device)
    ]
    opp_agents = [opponent_agent(env), opponent_agent(env)]

    rl_team = {0: rl_agents[0], 2: rl_agents[1]}
    opp_team = {1: opp_agents[0], 3: opp_agents[1]}

    wins = 0
    small = 0
    medium = 0
    big = 0
    for _ in range(num_episodes):
        level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
        obs = env.reset({'level': level})
        while True:
            if not obs:
                break
            current_player = int(list(obs.keys())[0])
            obs_player = obs[current_player]
            if current_player in rl_team:
                action = rl_team[current_player].select_action(obs_player, explore=False)
            else:
                action = opp_team[current_player].select_action(obs_player)
            obs = env.step(action)
            if env.done:
                break

        team_reward_rl = env.reward.get(0, 0) + env.reward.get(2, 0)
        team_reward_opp = env.reward.get(1, 0) + env.reward.get(3, 0)

        if team_reward_rl > team_reward_opp:
            wins += 1

        if team_reward_rl == 2:
            small += 1
        elif team_reward_rl == 4:
            medium += 1
        elif team_reward_rl == 6:
            big += 1

    win_rate = wins / max(1, num_episodes)
    small_rate = small / max(1, num_episodes)
    medium_rate = medium / max(1, num_episodes)
    big_rate = big / max(1, num_episodes)
    return [win_rate, small_rate, medium_rate, big_rate]


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)

    env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim = make_shared_components(device)
    print(f'state_dim: {state_dim}\naction_dim: {action_dim}\n')

    optimizer = torch.optim.Adam(policy_value_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=50000)

    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01

    rollout_episodes_per_update = 16
    total_updates = 1000
    eval_interval = 50
    num_eval_episodes = 30
    save_interval = 50

    # === warmup and checkpoint pool for "past RL opponents" ===
    warm_up = 100  # set this to your intended warmup boundary
    ckpt_dir = os.path.join('models', '3')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load existing checkpoints if any (so training can resume with past-opponent pool)
    checkpoint_pool: List[str] = []
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith('.pt') and fn.startswith('ppo_checkpoint'):
            checkpoint_pool.append(os.path.join(ckpt_dir, fn))

    for update_idx in range(1, total_updates + 1):
        # === MODIFIED: mixed-opponent rollout ===
        collect_rollout(
            env=env,
            current_net=policy_value_net,
            feature_encoder=feature_encoder,
            action_generator=action_generator,
            replay_buffer=replay_buffer,
            num_episodes=rollout_episodes_per_update,
            device=device,
            update_idx=update_idx,
            warm_up=warm_up,
            checkpoint_pool=checkpoint_pool,
            state_dim=state_dim,
            action_dim=action_dim
        )

        transitions = replay_buffer.to_training_batch()
        values = compute_state_values(policy_value_net, transitions, device=device)
        returns, advantages = compute_returns_and_advantages_grouped(transitions, values, gamma=gamma, lam=lam)

        ppo_update(
            policy_value_net,
            optimizer,
            transitions,
            returns,
            advantages,
            clip_epsilon,
            value_loss_coef,
            entropy_coef,
            epochs=4
        )
        replay_buffer.clear()

        # Episode reward reporting (only training players exist in buffer; keep your logic)
        episode_rewards = {}
        for t in transitions:
            if t.done:
                episode_rewards.setdefault(t.episode_id, []).append(t.reward)

        episode_avg_rewards = [sum(rs) / len(rs) for rs in episode_rewards.values()]
        avg_episode_reward = sum(episode_avg_rewards) / max(1, len(episode_avg_rewards))

        print(f'Update {update_idx}: avg episode reward {avg_episode_reward:.3f}')

        if update_idx % eval_interval == 0:
            eval_ret = evaluate_policy(env, RuleBasedAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(
                f'Eval after update {update_idx}: RL v.s. Rule_based:\n'
                f'win rate {eval_ret[0]:.2f}\n+1 rate {eval_ret[1]:.2f}\n+2 rate {eval_ret[2]:.2f}\n+3 rate {eval_ret[3]:.2f}\n'
            )
            eval_ret = evaluate_policy(env, RandomAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(
                f'Eval after update {update_idx}: RL v.s. Random:\n'
                f'win rate {eval_ret[0]:.2f}\n+1 rate {eval_ret[1]:.2f}\n+2 rate {eval_ret[2]:.2f}\n+3 rate {eval_ret[3]:.2f}\n'
            )

        if update_idx % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f'ppo_checkpoint{update_idx}.pt')
            torch.save(policy_value_net.state_dict(), ckpt_path)
            checkpoint_pool.append(ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')


if __name__ == '__main__':
    main()
