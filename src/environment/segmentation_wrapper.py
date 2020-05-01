import numpy as np
from gym import Wrapper, Env
from gym.spaces import Box

from segmentation.sprites_based.segmentator import Segmentator


class SegmentationWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._segmentator = Segmentator()
        h, w, _ = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8,)

        self.k = 0

    def step(self, action):
        frame, reward, done, info = self.env.step(action)

        return self._segmentator.segmentate(frame, info["x_pos"], info["y_pos"])[:, :, np.newaxis], reward, done, info
