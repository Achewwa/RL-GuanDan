import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from agents.action_generator import ActionGenerator
from agents.feature_encoder import FeatureEncoder
from agents.rl_agent import RLGuanDanAgent
from agents.rule_agent import RuleBasedAgent
from env import GuanDanEnv
from models.policy_value_net import PolicyValueNet
from selfplay.replay_buffer import ReplayBuffer, Transition

TRAIN_LEVELS = list(range(13))
# Toggle to train across all levels or stick to one fixed level.
train_all_levels = True
fixed_level = 0


def make_shared_components(device: str) -> Tuple[GuanDanEnv, FeatureEncoder, ActionGenerator, PolicyValueNet, int, int]:
    env = GuanDanEnv()
    feature_encoder = FeatureEncoder()
    action_generator = ActionGenerator(env)

    obs = env.reset({})
    first_player = list(obs.keys())[0]
    obs_player = obs[first_player]
    state_vec = feature_encoder.encode_state(obs_player)
    candidate_actions = action_generator.generate_legal_actions(obs_player)
    if not candidate_actions:
        candidate_actions = [{'player': first_player, 'action': [], 'claim': []}]
    action_vec = feature_encoder.encode_action(candidate_actions[0])

    state_dim = len(state_vec)
    action_dim = len(action_vec)

    policy_value_net = PolicyValueNet(state_dim, action_dim, hidden_dim=128).to(device)

    return env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim


def create_rl_agents(env: GuanDanEnv, net: PolicyValueNet, feature_encoder: FeatureEncoder, action_generator: ActionGenerator, device: str) -> List[RLGuanDanAgent]:
    return [RLGuanDanAgent(net, feature_encoder, action_generator, device=device) for _ in range(4)]


def collect_rollout(env: GuanDanEnv, agents: List[RLGuanDanAgent], feature_encoder: FeatureEncoder, action_generator: ActionGenerator, replay_buffer: ReplayBuffer, num_episodes: int, device: str) -> None:
    for _ in range(num_episodes):
        level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
        obs = env.reset({'level': level})
        while True:
            if not obs:
                break
            current_player = list(obs.keys())[0]
            obs_player = obs[current_player]

            state_vec = feature_encoder.encode_state(obs_player)
            chosen_action, chosen_index, log_prob = agents[current_player].select_action_with_logprob(obs_player, explore=True)
            action_feat = feature_encoder.encode_action(chosen_action)

            obs = env.step(chosen_action)
            reward = env.reward.get(current_player, 0)
            done = env.done

            transition = Transition(
                state=state_vec,
                action_feat=action_feat,
                action_index=chosen_index,
                reward=float(reward),
                done=bool(done),
                log_prob=float(log_prob)
            )
            replay_buffer.add_transition(transition)

            if done:
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


def ppo_update(policy_value_net: PolicyValueNet, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], clip_epsilon: float, value_loss_coef: float, entropy_coef: float, epochs: int, minibatch_size: int) -> None:
    states = batch['states']
    action_feats = batch['action_feats']
    actions_idx = batch['actions_idx']
    returns = batch['returns']
    advantages = batch['advantages']
    old_log_probs = batch['old_log_probs']

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    num_samples = states.shape[0]
    for _ in range(epochs):
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            states_mb = states[mb_idx]
            action_feats_mb = action_feats[mb_idx]
            returns_mb = returns[mb_idx]
            advantages_mb = advantages[mb_idx]
            old_log_probs_mb = old_log_probs[mb_idx]

            logits, values = policy_value_net(states_mb, action_feats_mb)
            new_log_probs = logits.squeeze(-1)

            ratios = torch.exp(new_log_probs - old_log_probs_mb)
            surr1 = ratios * advantages_mb
            surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns_mb)

            # Approximate entropy using a binary distribution over the chosen action probability.
            probs = torch.sigmoid(new_log_probs)
            entropy = (-probs * torch.log(probs + 1e-8) - (1.0 - probs) * torch.log(1.0 - probs + 1e-8)).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_value_net.parameters(), max_norm=0.5)
            optimizer.step()


def evaluate_policy(env: GuanDanEnv, policy_value_net: PolicyValueNet, feature_encoder: FeatureEncoder, action_generator: ActionGenerator, num_episodes: int, device: str) -> float:
    rl_agents = [
        RLGuanDanAgent(policy_value_net, feature_encoder, action_generator, device=device),
        RLGuanDanAgent(policy_value_net, feature_encoder, action_generator, device=device)
    ]
    rule_agents = [RuleBasedAgent(env), RuleBasedAgent(env)]

    rl_team = {0: rl_agents[0], 2: rl_agents[1]}
    rule_team = {1: rule_agents[0], 3: rule_agents[1]}

    wins = 0
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
                action = rule_team[current_player].select_action(obs_player)
            obs = env.step(action)
            if env.done:
                break

        team_reward_rl = env.reward.get(0, 0) + env.reward.get(2, 0)
        team_reward_rule = env.reward.get(1, 0) + env.reward.get(3, 0)
        if team_reward_rl > team_reward_rule:
            wins += 1

    win_rate = wins / max(1, num_episodes)
    return win_rate


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim = make_shared_components(device)
    agents = create_rl_agents(env, policy_value_net, feature_encoder, action_generator, device)

    optimizer = torch.optim.Adam(policy_value_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=50000)

    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.02
    rollout_episodes_per_update = 8
    total_updates = 200
    eval_interval = 20
    num_eval_episodes = 10
    save_interval = 50

    for update_idx in range(1, total_updates + 1):
        collect_rollout(env, agents, feature_encoder, action_generator, replay_buffer, rollout_episodes_per_update, device)

        batch = replay_buffer.to_training_batch(device=device)
        with torch.no_grad():
            _, values = policy_value_net(batch['states'], batch['action_feats'])
            values = values.squeeze(-1)
        returns, advantages = compute_returns_and_advantages(batch['rewards'], batch['dones'], values, gamma=gamma, lam=lam)
        batch['returns'] = returns
        batch['advantages'] = advantages

        ppo_update(policy_value_net, optimizer, batch, clip_epsilon, value_loss_coef, entropy_coef, epochs=4, minibatch_size=64)
        replay_buffer.clear()

        avg_reward = batch['rewards'].mean().item()
        print(f'Update {update_idx}: avg reward {avg_reward:.3f}')

        if update_idx % eval_interval == 0:
            win_rate = evaluate_policy(env, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(f'Eval after update {update_idx}: RL win rate {win_rate:.2f}')

        if update_idx % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(policy_value_net.state_dict(), 'models/ppo_checkpoint.pt')
            print('Checkpoint saved to models/ppo_checkpoint.pt')


if __name__ == '__main__':
    main()
