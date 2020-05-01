import gym
from gym import Env


class SkipWrapper(gym.Wrapper):
    def __init__(self, env: Env, k: int):
        super().__init__(env)
        self.k = k

    def step(self, action: int):
        sum_rewards = 0
        for _ in range(self.k):
            frame, reward, done, info = self.env.step(action)
            sum_rewards += reward
            if done:
                break
        return frame, sum_rewards, done, info
