from typing import List, Type

import cv2
import gym_super_mario_bros
from gym import Env
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from environment.frame_wrapper import FrameWrapper
from environment.no_op_reset_wrapper import NoopResetEnv
from environment.reward_wrapper import RewardScoreWrapper
from environment.segmentation_wrapper import SegmentationWrapper
from environment.skip_wrapper import SkipWrapper
from environment.stack_wrapper import FrameStack, LazyFrames, LazyVStackedFrames


def make_environment(
    actions: List[List[str]] = RIGHT_ONLY,
    with_segmentation: bool = False,
    lazy_class: Type = LazyFrames,
    grayscale: bool = True,
) -> Env:
    e = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    e = JoypadSpace(e, actions)
    e = RewardScoreWrapper(e)
    e = SkipWrapper(e, 4)
    if with_segmentation:
        e = SegmentationWrapper(e)
    e = FrameWrapper(
        e,
        width=84,
        height=84,
        grayscale=grayscale and not with_segmentation,
        interpolation=cv2.INTER_NEAREST if with_segmentation else cv2.INTER_AREA,
    )
    e = FrameStack(e, 4, lazy_class=lazy_class)
    e = NoopResetEnv(e, 4)
    return e


def make_environment_for_dqn() -> Env:
    return make_environment(actions=RIGHT_ONLY, with_segmentation=False, grayscale=True, lazy_class=LazyFrames)


def make_environment_for_dqn_from_fcn() -> Env:
    return make_environment(actions=RIGHT_ONLY, with_segmentation=False, grayscale=False, lazy_class=LazyVStackedFrames)


def make_environment_for_dqn_with_segm() -> Env:
    return make_environment(actions=RIGHT_ONLY, with_segmentation=True, grayscale=False, lazy_class=LazyFrames)
