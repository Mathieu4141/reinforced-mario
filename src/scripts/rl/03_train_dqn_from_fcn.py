"""
Use this script to train the Double DQN with transfer learning from a FCN model.

This script will take about a day to run, depending on your computer performances.

Logs and networks will be saved in /exp/rl

At any point, you can visualize the agent playing by turning the
 `dqn-from-fcn__f16-k3_s2__f32-k3_f32-k3_s2__d8-with-conv` (or the name of the model) field to True in the file /display.yml
"""

from agents.dqn_from_fcn import DQNFromFrozenFCN
from agents.training import train
from environment.env import make_environment_for_dqn_from_fcn
from utils.reproductibility import seed_all

if __name__ == "__main__":
    e = make_environment_for_dqn_from_fcn()
    seed_all(e)
    a = DQNFromFrozenFCN(e, "fcn__f16-k3_s2__f32-k3_f32-k3_s2__d8", with_conv=True)
    train(a, e)
