"""
Use this script to see the trained agent playing. Uncomment the agent you want to see playing in the main
"""

from gym import Env

from agents.saved_dqn import SavedDQN
from environment.env import (
    make_environment_for_dqn,
    make_environment_for_dqn_from_fcn,
    make_environment_for_dqn_with_segm,
)
from environment.play import play
from utils.reproductibility import seed_all


def _play_on_env(env: Env, name: str, episodes: int):
    a = SavedDQN(name, env)
    play(a, env, display=True, episodes=episodes)


def play_qdn(episodes: int, name: str = "dqn"):
    _play_on_env(make_environment_for_dqn(), name, episodes)


def play_qdn_with_segmentation(episodes: int, name: str = "dqn-segm"):
    _play_on_env(make_environment_for_dqn_with_segm(), name, episodes)


def play_qdn_from_fcn(episodes: int, name: str = "dqn-from-fcn__f16-k3_s2__f32-k3_f32-k3_s2__d8-with-conv"):
    _play_on_env(make_environment_for_dqn_from_fcn(), name, episodes)


if __name__ == "__main__":
    seed_all(s=4242)

    # Uncomment one of the following:

    # play_qdn(episodes=5)
    # play_qdn_with_segmentation(episodes=5)
    play_qdn_from_fcn(episodes=5)
