from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np
from gym import Env

from agents.experience import Experience


class AgentABC(ABC):
    def __init__(self, memory_size: int, env: Env):
        self._n_actions = env.action_space.n
        self._state_shape = env.observation_space.shape
        self._memory: Deque[Experience] = deque(maxlen=memory_size)

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool) -> int:
        pass

    def memorize(self, exp: Experience):
        self._memory.append(exp)

    def learn(self):
        pass
