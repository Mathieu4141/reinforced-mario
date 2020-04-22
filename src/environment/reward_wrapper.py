from typing import Dict, Any

from gym import Wrapper
from gym_super_mario_bros import SuperMarioBrosEnv


class RewardScoreWrapper(Wrapper):
    def reset(self, **kwargs):
        self._previous_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, info), done, info

    def reward(self, reward, info: Dict[str, Any]):
        reward += max(0, (info["score"] - self._previous_score) // 25)
        self._previous_score = info["score"]
        return reward


def mario_bros_reward(self: SuperMarioBrosEnv):
    return self._x_reward // 2 + self._time_penalty + (self._death_penalty * 16)


SuperMarioBrosEnv._get_reward = mario_bros_reward
SuperMarioBrosEnv.reward_range = (-300, 20)
