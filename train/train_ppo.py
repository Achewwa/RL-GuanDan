import os
import random
from typing import List, Tuple

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
# Toggle to train across all levels or stick to one fixed level.
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
    action_vec = feature_encoder.encode_action(candidate_actions[0], obs_player)

    state_dim = len(state_vec)
    action_dim = len(action_vec)

    policy_value_net = PolicyValueNet(state_dim, action_dim, hidden_dim=128).to(device)

    return env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim


def create_rl_agents(env: GuanDanEnv, net: PolicyValueNet, feature_encoder: FeatureEncoder, action_generator: ActionGenerator, device: str) -> List[RLGuanDanAgent]:
    return [RLGuanDanAgent(net, feature_encoder, action_generator, device=device) for _ in range(4)]

def compute_speed_bonuses(env: GuanDanEnv, *, t_ref: float = 100.0, alpha: float = 0.5, beta: float = 0.25) -> dict:
    '''
    Scheme 1 speed shaping computed in TRAIN (no env modifications).

    Returns:
      bonuses: dict {player_id: bonus_float}

    Logic:
    - Only meaningful if env.done == True and env.cleared is valid.
    - speed_score = clip((t_ref - env.round) / t_ref, 0, 1)
    - First finisher (cleared[0]) gets alpha * speed_score
    - Winning teammate finisher gets beta * speed_score
      - double-dweller: teammate = cleared[1]
      - else if cleared[0] and cleared[2] same team: teammate = cleared[2]
      - else teammate = (cleared[0] + 2) % 4
    '''
    bonuses = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    if not getattr(env, 'done', False):
        return bonuses

    # Safety: cleared should exist and contain winners order.
    cleared = list(getattr(env, 'cleared', []))
    if len(cleared) < 1:
        return bonuses

    # --- speed score based on total "player-turn" steps ---
    T = float(getattr(env, 'round', 0.0))
    speed_score = (t_ref - T) / t_ref
    if speed_score < 0.0:
        speed_score = 0.0
    elif speed_score > 1.0:
        speed_score = 1.0

    first_finisher = int(cleared[0])

    # Determine the teammate finisher consistent with env reward logic.
    # We mirror env._set_reward's winner identity logic, but only for teammate assignment.
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
    '''
    Fix rollout reward assignment bug:

    - env.reward at terminal contains rewards for ALL players,
      but the naive rollout code only writes reward for current_player once.
    - This function adds each player's terminal reward to THAT player's last transition.reward.

    transitions_by_player: {pid: [Transition, Transition, ...]} (episode-local)
    terminal_rewards: {pid: float}
    '''
    for pid, traj in transitions_by_player.items():
        if not traj:
            continue
        traj[-1].reward += float(terminal_rewards.get(pid, 0.0))
        traj[-1].done = True  # mark the last transition of each player's trajectory as terminal


def collect_rollout(
    env: GuanDanEnv,
    agents: List[RLGuanDanAgent],
    feature_encoder: FeatureEncoder,
    action_generator: ActionGenerator,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    device: str
) -> None:
    '''
    Collect rollouts for PPO.

    Fixes:
    1) Terminal reward distribution bug:
       - At env.done, env.reward contains rewards for all 4 players.
       - We add each player's terminal reward (plus speed bonus) to THAT player's last transition.
    2) (Optional but recommended) Prepare data for per-player GAE by storing episode_id/player_id.
    '''
    for episode_id in range(num_episodes):
        level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
        obs = env.reset({'level': level})

        # Episode-local trajectories per player (so we can patch terminal rewards correctly)
        transitions_by_player = {0: [], 1: [], 2: [], 3: []}

        while True:
            if not obs:
                break

            current_player = list(obs.keys())[0]
            obs_player = obs[current_player]

            # --- policy sampling (unchanged) ---
            candidate_actions, probabilities, log_probabilities, state_vec, action_vecs = agents[current_player]._compute_action_distribution(
                obs_player, explore=True
            )
            action_idx = torch.multinomial(probabilities, 1).item()
            chosen_action = candidate_actions[action_idx]
            chosen_log_prob = log_probabilities[action_idx].detach()

            # --- env step ---
            obs = env.step(chosen_action)
            done = bool(env.done)

            # IMPORTANT:
            # Per-step rewards are typically 0 in your env until done.
            # Do NOT attempt to read 'env.reward[current_player]' only.
            # We write step reward as 0.0 here, and patch terminal rewards later.
            step_reward = 0.0

            # --- create transition for the acting player only ---
            transition = Transition(
                episode_id=int(episode_id),
                player_id=int(current_player),
                state=np.asarray(state_vec, dtype=np.float32),
                candidate_action_feats=np.asarray(action_vecs, dtype=np.float32),
                action_index=int(action_idx),
                reward=float(step_reward),
                done=False,  # will be set True on last step of each player's own trajectory
                log_prob=float(chosen_log_prob)
            )
            transitions_by_player[current_player].append(transition)

            if done:
                # --- terminal rewards (env base) ---
                # env.reward is a dict {0..3: reward} at terminal
                base_terminal = dict(env.reward) if isinstance(env.reward, dict) else {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

                # --- speed bonuses (scheme 1) ---
                speed_bonus = compute_speed_bonuses(env, t_ref=100.0, alpha=0.5, beta=0.25)

                # --- final terminal rewards per player ---
                terminal_rewards = {pid: float(base_terminal.get(pid, 0.0)) + float(speed_bonus.get(pid, 0.0)) for pid in range(4)}

                # Patch rewards to each player's last transition and mark done.
                apply_terminal_rewards_to_last_steps(transitions_by_player, terminal_rewards)

                # Push all players' transitions to global buffer
                for pid in range(4):
                    for t in transitions_by_player[pid]:
                        replay_buffer.add_transition(t)

                break

def compute_returns_and_advantages(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages[step] = gae
        returns[step] = advantages[step] + values[step]
        next_value = values[step]
    return returns, advantages

def compute_returns_and_advantages_grouped(
    transitions: List[Transition],
    values: torch.Tensor,
    gamma: float,
    lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Compute returns/advantages by grouping transitions into (episode_id, player_id) trajectories.

    This fixes the theoretical bug where 4 players' steps are interleaved but treated as one sequence.

    Args:
      transitions: list aligned with values (same order)
      values: tensor [N] with V(s_t) for each transition
    Returns:
      returns: [N]
      advantages: [N]
    '''
    n = len(transitions)
    returns = torch.zeros(n, dtype=torch.float32, device=values.device)
    advantages = torch.zeros(n, dtype=torch.float32, device=values.device)

    # Build groups: key -> list of indices in time order (already appended in time order per player per episode)
    groups = {}
    for idx, t in enumerate(transitions):
        key = (int(t.episode_id), int(t.player_id))
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    # For each trajectory, do reversed GAE
    for key, idxs in groups.items():
        gae = 0.0
        next_value = 0.0

        # Walk backwards within that player's trajectory
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


def ppo_update(policy_value_net: PolicyValueNet, optimizer: torch.optim.Optimizer, transitions: List[Transition], returns: torch.Tensor, advantages: torch.Tensor, clip_epsilon: float, value_loss_coef: float, entropy_coef: float, epochs: int) -> None:
    num_samples = len(transitions)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = torch.randperm(num_samples)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for idx in indices:
            t = transitions[idx]
            ret = returns[idx]
            adv = advantages[idx]

            state_tensor = torch.tensor(t.state, dtype=torch.float32, device=ret.device)
            action_tensor = torch.tensor(t.candidate_action_feats, dtype=torch.float32, device=ret.device)
            state_batch = state_tensor.unsqueeze(0).repeat(action_tensor.shape[0], 1)

            logits, values = policy_value_net(state_batch, action_tensor)
            logits = logits.squeeze(-1)
            log_probs_new = F.log_softmax(logits, dim=0)
            probs_new = log_probs_new.exp()

            # PPO ratio computed with log-softmax over all candidates (old log_prob stored the same way).
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
            current_player = list(obs.keys())[0]
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
    agents = create_rl_agents(env, policy_value_net, feature_encoder, action_generator, device)

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

    for update_idx in range(1, total_updates + 1):
        collect_rollout(env, agents, feature_encoder, action_generator, replay_buffer, rollout_episodes_per_update, device)

        transitions = replay_buffer.to_training_batch()
        values = compute_state_values(policy_value_net, transitions, device=device)
        returns, advantages = compute_returns_and_advantages_grouped(transitions, values, gamma=gamma, lam=lam)

        ppo_update(policy_value_net, optimizer, transitions, returns, advantages, clip_epsilon, value_loss_coef, entropy_coef, epochs=4)
        replay_buffer.clear()

        episode_rewards = {}
        for t in transitions:
            if t.done:
                episode_rewards.setdefault(t.episode_id, []).append(t.reward)

        episode_avg_rewards = [
            sum(rs) / len(rs) for rs in episode_rewards.values()
        ]

        avg_episode_reward = sum(episode_avg_rewards) / len(episode_avg_rewards)

        print(f'Update {update_idx}: avg episode reward {avg_episode_reward:.3f}')

        if update_idx % eval_interval == 0:
            eval_ret = evaluate_policy(env, RuleBasedAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(f'Eval after update {update_idx}: RL v.s. Rule_based:\nwin rate {eval_ret[0]:.2f}\n+1 rate {eval_ret[1]:.2f}\n+2 rate {eval_ret[2]:.2f}\n+3 rate {eval_ret[3]:.2f}\n')
            eval_ret = evaluate_policy(env, RandomAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(f'Eval after update {update_idx}: RL v.s. Random:\nwin rate {eval_ret[0]:.2f}\n+1 rate {eval_ret[1]:.2f}\n+2 rate {eval_ret[2]:.2f}\n+3 rate {eval_ret[3]:.2f}\n')

        if update_idx % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(policy_value_net.state_dict(), f'models/1/ppo_checkpoint{update_idx}.pt')
            print(f'Checkpoint saved to models/1/ppo_checkpoint{update_idx}.pt')

if __name__ == '__main__':
    main()
