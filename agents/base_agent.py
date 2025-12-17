# agents/base_agent.py

from abc import ABC, abstractmethod
from env import GuanDanEnv
from utils import Utils
from agents.action_generator import ActionGenerator


class BaseAgent(ABC):
    '''Common base class for all Guandan agents.'''

    def __init__(self, env: GuanDanEnv):
        self.env = env
        self.action_generator = ActionGenerator(env)
        self.utils = Utils()

    @abstractmethod
    def select_action(self, obs_for_player: dict) -> dict:
        '''Return one legal action given the observation.'''
        pass