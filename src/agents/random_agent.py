import numpy as np
from gym import Env

from agents.agent_abc import AgentABC


class RandomAgent(AgentABC):
    def __init__(self, env: Env):
        super().__init__(0, env)

    def act(self, state: np.ndarray, explore: bool) -> int:
        return np.random.choice(self._n_actions)
