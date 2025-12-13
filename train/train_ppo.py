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

    initial_level = TRAIN_LEVELS[0]
    obs = env.reset({'level': initial_level})
    action_generator = ActionGenerator(env)
    feature_encoder = FeatureEncoder(env)
    
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

            candidate_actions, probabilities, log_probabilities, state_vec, action_vecs = agents[current_player]._compute_action_distribution(obs_player, explore=True)
            action_idx = torch.multinomial(probabilities, 1).item()

            chosen_action = candidate_actions[action_idx]
            chosen_log_prob = log_probabilities[action_idx].detach()

            obs = env.step(chosen_action)
            reward = env.reward.get(current_player, 0)
            done = env.done

            transition = Transition(
                state=np.asarray(state_vec, dtype=np.float32),
                candidate_action_feats=np.asarray(action_vecs, dtype=np.float32),
                action_index=action_idx,
                reward=float(reward),
                done=bool(done),
                log_prob=float(chosen_log_prob)  # log-softmax of chosen action
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
    rule_agents = [opponent_agent(env), opponent_agent(env)]

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
    print('device: ', device)

    env, feature_encoder, action_generator, policy_value_net, state_dim, action_dim = make_shared_components(device)
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
    save_interval = 100

    for update_idx in range(1, total_updates + 1):
        collect_rollout(env, agents, feature_encoder, action_generator, replay_buffer, rollout_episodes_per_update, device)

        transitions = replay_buffer.to_training_batch()
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        dones = torch.tensor([1.0 if t.done else 0.0 for t in transitions], dtype=torch.float32, device=device)
        values = compute_state_values(policy_value_net, transitions, device=device)
        returns, advantages = compute_returns_and_advantages(rewards, dones, values, gamma=gamma, lam=lam)

        ppo_update(policy_value_net, optimizer, transitions, returns, advantages, clip_epsilon, value_loss_coef, entropy_coef, epochs=4)
        replay_buffer.clear()

        avg_reward = rewards.mean().item()
        print(f'Update {update_idx}: avg reward {avg_reward:.3f}')

        if update_idx % eval_interval == 0:
            win_rate = evaluate_policy(env, RuleBasedAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(f'Eval after update {update_idx}: RL v.s. Rule_based win rate {win_rate:.2f}')
            win_rate = evaluate_policy(env, RandomAgent, policy_value_net, feature_encoder, action_generator, num_eval_episodes, device)
            print(f'Eval after update {update_idx}: RL v.s. Random win rate {win_rate:.2f}')

        if update_idx % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(policy_value_net.state_dict(), 'models/ppo_checkpoint.pt')
            print('Checkpoint saved to models/ppo_checkpoint.pt')


if __name__ == '__main__':
    main()
