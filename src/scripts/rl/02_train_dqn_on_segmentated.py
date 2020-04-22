"""
Use this script to train the Double DQN agent that takes the segmented frames as input.

This script will take about a day to run, depending on your computer performances.

Logs and networks will be saved in /exp/rl

At any point, you can visualize the agent playing by turning the `dqn-segm` (or the name of the model) field to True
 in the file /display.yml
"""

from agents.dqn_agent import DQNAgent
from agents.training import train
from environment.env import make_environment_for_dqn_with_segm
from utils.reproductibility import seed_all

if __name__ == "__main__":
    e = make_environment_for_dqn_with_segm()
    seed_all(e)
    a = DQNAgent(e, "dqn-segm")
    train(a, e)
