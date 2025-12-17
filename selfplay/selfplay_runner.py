from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from env import GuanDanEnv


@dataclass
class Trajectory:
    observations: List[Dict[str, Any]] = field(default_factory=list)
    legal_actions: List[Optional[Any]] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


class SelfPlayRunner:
    def __init__(self, agent_cls: Callable[[], Any], num_games: int = 1) -> None:
        self.agent_cls = agent_cls
        self.num_games = num_games
        self.env = GuanDanEnv()
        self.agents = [self.agent_cls() for _ in range(4)]

    def run(self) -> List[List[Trajectory]]:
        all_game_trajectories: List[List[Trajectory]] = []
        for _ in range(self.num_games):
            game_trajectories = [Trajectory() for _ in range(4)]
            obs = self.env.reset({})
            while not self.env.done:
                if not obs:
                    break
                current_players = list(obs.keys())
                if not current_players:
                    break
                current_player = current_players[0]
                obs_for_player = obs[current_player]
                agent = self.agents[current_player]
                response = agent.select_action(obs_for_player)
                if isinstance(response, dict) and 'player' not in response:
                    response['player'] = current_player
                try:
                    next_obs = self.env.step(response)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f'Error during env.step: {exc}')
                    break
                reward = 0.0
                if isinstance(self.env.reward, dict):
                    reward = float(self.env.reward.get(current_player, 0))
                legal_actions = None
                if isinstance(obs_for_player, dict):
                    legal_actions = obs_for_player.get('legal_actions')
                game_trajectories[current_player].observations.append(obs_for_player)
                game_trajectories[current_player].legal_actions.append(legal_actions)
                if isinstance(response, dict):
                    game_trajectories[current_player].actions.append(response.get('action'))
                else:
                    game_trajectories[current_player].actions.append(None)
                game_trajectories[current_player].rewards.append(reward)
                game_trajectories[current_player].dones.append(self.env.done)
                obs = next_obs
            all_game_trajectories.append(game_trajectories)
        return all_game_trajectories
