import random

import numpy as np
from gym import Env
import tensorflow as tf

SEED = 4242


def seed_all(env: Env = None, s: int = SEED):
    if env is not None:
        env.seed(s)
    np.random.seed(s)
    random.seed(s)
    tf.random.set_random_seed(s)
