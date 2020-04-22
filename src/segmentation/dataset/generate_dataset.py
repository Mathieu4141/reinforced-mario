from pathlib import Path
from shutil import rmtree

import cv2
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from agents.random_agent import RandomAgent
from environment.copy_frame import CopyFrame
from environment.no_op_reset_wrapper import NoopResetEnv
from environment.play import play
from environment.skip_wrapper import SkipWrapper
from environment.stack_wrapper import FrameStack
from segmentation.dataset.sets import Set
from segmentation.sprites_based.segmentator import Segmentator
from utils.reproductibility import seed_all


def _make_frames(set_: Set, n: int):
    e = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    e = JoypadSpace(e, RIGHT_ONLY)
    e = SkipWrapper(e, 5)
    e = CopyFrame(e)
    e = FrameStack(e, 4)
    e = NoopResetEnv(e, 4)
    a = RandomAgent(env=e)
    play(
        a,
        e,
        frames_directory=set_.path / "img",
        display=False,
        n=n * 10,
        save_each=15,
        state2img=lambda frames: cv2.cvtColor(np.vstack(frames._frames), cv2.COLOR_RGB2BGR),
    )


def _make_frames_from_selected(set_: Set, directory: Path):
    frame_paths = list(sorted(directory.glob("frame-*.png")))
    assert not len(frame_paths) % 4
    for i in range(0, len(frame_paths), 4):
        cv2.imwrite(
            str(set_.path / "img" / frame_paths[i].name),
            np.vstack([cv2.imread(str(frame_path)) for frame_path in frame_paths[i : i + 4]]),
        )


def _labelize_frames(set_: Set):
    (set_.path / "gt").mkdir(exist_ok=True)
    segm = Segmentator()
    for img_path, gt_path in set_.images_gt_paths:
        m_x, m_y = None, None
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        s = 84
        gt = np.zeros((s * 4, s), dtype=np.uint8)
        resized_img = np.zeros((s * 4, s, 3), dtype=np.uint8)
        for i in range(4):
            frame = img[i * 240 : i * 240 + 240, :, :]
            frame_gt = segm.segmentate(frame, m_x, m_y)
            gt[i * s : i * s + s, :] = cv2.resize(frame_gt, (s, s))
            resized_img[i * s : i * s + s, :] = cv2.resize(frame, (s, s))

        cv2.imwrite(str(gt_path), gt)
        cv2.imwrite(str(img_path), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))


def _make_set(set_: Set, n: int):
    rmtree(set_.path, ignore_errors=True)
    _make_frames(set_, n)
    _labelize_frames(set_)


if __name__ == "__main__":
    seed_all()
    _make_set(Set.TRAIN, 1_000)
    _make_set(Set.TEST, 200)
