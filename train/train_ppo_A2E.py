import os
import sys
import argparse

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

from collections import Counter
from typing import Dict, List, Any

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

from collections import Counter
from typing import Dict, List, Any

def calculate_shaping_reward(env: GuanDanEnv, player_id: int, obs_before_action: dict, chosen_action: dict, action_generator: ActionGenerator) -> float:
    """
    根据用户定义的规则计算额外的 shaping reward。
    """
    shaping_reward = 0.0
    utils = env.Utils
    
    # === 1. 提取基础信息 ===
    my_deck = obs_before_action.get('deck', [])
    last_move = obs_before_action.get('last_move', {})
    
    # 当前动作信息
    chosen_cards = chosen_action.get('action', [])
    chosen_claim = chosen_action.get('claim', [])
    # 修改点 2：这里把原本的 _ 改成了 current_points，因为后面要用
    current_type, current_points = env._check_poker_type(chosen_claim) if chosen_claim else ('pass', ())
    
    # 上家动作信息
    last_player = last_move.get('player', -1)
    last_claim = last_move.get('claim', [])
    last_type = 'pass'
    last_rank_char = ''
    if last_claim:
        last_type, last_points = env._check_poker_type(last_claim)
        # 获取牌的大小用于后续判断 (简化版：取第一个点数)
        if isinstance(last_points, tuple) and len(last_points) > 0:
             last_rank_char = last_points[-1] 
        elif isinstance(last_points, str):
             last_rank_char = last_points
    
    # 身份判断
    teammate_id = (player_id + 2) % 4
    is_teammate = (last_player == teammate_id)
    is_opponent = (last_player != -1 and not is_teammate)
    
    # 炸弹类型集合
    bomb_types = ['bomb', 'rocket', 'straight_flush']
    # 简单牌型集合 (根据你的要求 Rule A)
    simple_types = ['single', 'pair', 'three', 'set']

    # =========================================================================
    # Rule A: 对手出普通牌型时的炸弹使用策略
    # 逻辑：如果炸的是 single, pair, three, set，且对手手牌 > 5 张，扣 0.2
    # =========================================================================
    if is_opponent and current_type in bomb_types:
        # 即使在 obs 中没有对手手牌数，训练时我们可以直接访问 env.player_decks
        # 注意：这里需要确保 env.player_decks 还没被 step 更新，或者逻辑上这是上一步的状态
        # 在 collect_rollout 中调用 step 之前调用此函数时，env.player_decks 还是旧的，
        # 但我们这里需要的是“上家”的手牌数，通常可以用 env.player_decks[last_player] 获取实时数量
        opponent_card_count = len(env.player_decks[last_player])
        
        if last_type in simple_types and opponent_card_count > 5:
            shaping_reward -= 0.2

    # =========================================================================
    # Rule B: 滥用逢人配 (Wild Card Abuse)
    # 逻辑：如果使用了逢人配，但没有组成 炸弹、同花顺 或 顺子，扣 0.5
    # =========================================================================
    wild_card_str = 'h' + env.level
    # 检查物理牌 action 中是否包含级牌
    uses_wild = any(utils.Num2Poker(c) == wild_card_str for c in chosen_cards)
    
    # 允许的类型：炸弹(bomb)、同花顺(straight_flush)、顺子(straight)
    allowed_wild_types = ['bomb', 'straight_flush', 'straight', 'rocket'] # rocket包含双王，虽然不含级牌但防守一下
    
    if uses_wild:
        if current_type not in allowed_wild_types:
            shaping_reward -= 0.5

    # =========================================================================
    # Rule C & D: 队友配合与防内耗 (沿用之前的逻辑)
    # 1. 炸弹压队友 (-0.5)
    # 2. 阻碍队友过小牌 (-0.1)
    # =========================================================================
    
    # C.1 炸弹压队友
    if is_teammate and last_type in bomb_types and current_type in bomb_types:
        shaping_reward -= 0.5
        
    # C.2 队友送小牌却 Pass (阻碍队友)
    # 逻辑：队友出单张小牌，自己 Pass
    if is_teammate and last_type == 'single' and current_type == 'pass':
        # 判断是否是小牌 (例如 < 10)
        rank_idx = -1
        if last_rank_char in env.point_order:
            rank_idx = env.point_order.index(last_rank_char)
        # 假设 0 (10) 的索引是 8，那么 < 8 的是 2-9
        if 0 <= rank_idx < 8:
            shaping_reward -= 0.1

    # =========================================================================
    # Rule E: 拆炸弹判定 (已修改)
    # 1. 拆炸弹组同花顺 -> 奖励 +0.1
    # 2. 拆炸弹打普通牌 -> 惩罚 -0.4
    # =========================================================================
    if current_type != 'pass':
        rank_counts = Counter([utils.Num2Poker(c)[1] for c in my_deck])
        played_ranks = [utils.Num2Poker(c)[1] for c in chosen_cards]
        
        is_breaking_bomb = False
        for r in set(played_ranks):
            original_count = rank_counts[r]
            played_count = played_ranks.count(r)
            remaining_count = original_count - played_count
            
            # 核心判定：原本是炸弹(>=4)，打完没打光，还剩1~3张，视为拆了
            if original_count >= 4 and 0 < remaining_count < 4:
                is_breaking_bomb = True
                break
        
        if is_breaking_bomb:
            # Case 1: 为了打出同花顺而拆炸弹 -> 鼓励
            if current_type == 'straight_flush':
                shaping_reward += 0.1
            # Case 2: 打出的不是炸弹类（如单张、对子等），却拆了炸弹 -> 惩罚
            # 注意：如果 current_type 是 'bomb' (比如5张拆成4张打)，通常不视为坏行为，这里排除
            elif current_type not in ['bomb', 'rocket']: 
                shaping_reward -= 0.4

    return shaping_reward



# train_ppo.py 中的 collect_rollout 修改部分

def collect_rollout(env: GuanDanEnv, agents: List[RLGuanDanAgent], feature_encoder: FeatureEncoder, action_generator: ActionGenerator, replay_buffer: ReplayBuffer, num_episodes: int, device: str) -> None:
    for _ in range(num_episodes):
        level = random.choice(TRAIN_LEVELS) if train_all_levels else fixed_level
        obs = env.reset({'level': level})
        while True:
            if not obs:
                break
            current_player = list(obs.keys())[0]
            obs_player = obs[current_player] # 这是 obs_before_action

            # 1. 计算动作分布
            candidate_actions, probabilities, log_probabilities, state_vec, action_vecs = agents[current_player]._compute_action_distribution(obs_player, explore=True)
            action_idx = torch.multinomial(probabilities, 1).item()

            chosen_action = candidate_actions[action_idx]
            chosen_log_prob = log_probabilities[action_idx].detach()

            # === 2. 计算 Rule-based Shaping Reward (在 Step 之前还是之后算都可以，这里选择 Step 后) ===
            # 为了获取最准确的上下文，我们在 Step 之前先“快照”一下需要判断环境状态的数据
            # 但实际上 env 对象是一直存在的，只要我们能在函数里访问到就行。
            
            # 先缓存一下 calculation 需要的 shaping reward
            # 注意：calculate_shaping_reward 内部使用了 env.player_decks
            # 此时 env.player_decks 包含当前玩家还没出的牌，以及对手还没变的牌数
            step_shaping_reward = calculate_shaping_reward(
                env, 
                current_player, 
                obs_player, 
                chosen_action, 
                action_generator  # <--- 新增参数
            )

            # 3. 执行环境步
            obs = env.step(chosen_action)
            
            # 4. 获取环境原始奖励
            raw_reward = env.reward.get(current_player, 0)
            
            # 5. 合并奖励
            total_reward = float(raw_reward) + step_shaping_reward
            
            # 打印调试信息（可选，训练稳定后注释掉）
            # if step_shaping_reward != 0:
            #     print(f"Player {current_player} Shaping Reward: {step_shaping_reward}")
            # ==============================

            done = env.done

            transition = Transition(
                state=np.asarray(state_vec, dtype=np.float32),
                candidate_action_feats=np.asarray(action_vecs, dtype=np.float32),
                action_index=action_idx,
                reward=float(total_reward), # 使用合并后的奖励
                done=bool(done),
                log_prob=float(chosen_log_prob)
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

    # === 1. 新增：解析命令行参数 ===
    parser = argparse.ArgumentParser(description="GuanDan PPO Training")
    parser.add_argument('--run_name', type=str, default='default_run', help='本次训练的名称，用于区分模型文件')
    args = parser.parse_args()
    
    # 打印一下当前运行的名字
    print(f"Starting training run: {args.run_name}")
    # ============================

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
    rollout_episodes_per_update = 32
    total_updates = 2000
    eval_interval = 50
    num_eval_episodes = 100
    save_interval = 50

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
            # === 修改开始 ===
            # 1. 创建基于 run_name 的独立文件夹
            # 例如: models/run_A/ 或 models/run_B/
            save_dir = f'models/A2E'
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. 文件名只需要包含 update_idx
            # 最终路径: models/run_A/checkpoint_100.pt
            save_path = f'{save_dir}/checkpoint_{update_idx}.pt' 
            
            torch.save(policy_value_net.state_dict(), save_path)
            print(f'[{args.run_name}] Checkpoint saved to {save_path}')
            # === 修改结束 ===

if __name__ == '__main__':
    main()
