"""
Use this script to evaluate the performance of the random agents
"""

import numpy as np

from agents.random_agent import RandomAgent
from environment.env import make_environment
from environment.play import play
from utils.reproductibility import seed_all


if __name__ == "__main__":
    e = make_environment()
    seed_all(e)

    a = RandomAgent(env=e)
    steps: np.ndarray = np.array(play(a, e, display=False, episodes=20))

    print(f"Random agent")
    print(f"Steps: {steps.mean():.2f} +/- {steps.std():.2f}")

    # 475.75 +/- 524.26
