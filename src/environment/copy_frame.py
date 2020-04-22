import numpy as np
from gym import ObservationWrapper


class CopyFrame(ObservationWrapper):
    def observation(self, observation):
        return np.copy(observation)
