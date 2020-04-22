from pathlib import Path
from shutil import rmtree
from typing import Callable, List

import cv2
import numpy as np
from gym import Env

from agents.agent_abc import AgentABC


def play(
    agent: AgentABC,
    env: Env,
    frames_directory: Path = None,
    display: bool = False,
    n: int = 1_000_000,
    state2img: Callable = None,
    save_each: int = None,
    episodes: int = None,
) -> List[int]:
    if frames_directory is not None:
        rmtree(str(frames_directory), ignore_errors=True)
        frames_directory.mkdir(parents=True)
    done = True
    episode = 0
    steps = [0]
    for step in range(n):
        if done:
            state = env.reset()
            episode += 1
            steps.append(0)
            if episode == episodes:
                break
        action = agent.act(np.asarray(state), explore=True)
        state, reward, done, info = env.step(action)

        steps[-1] += 1

        if save_each is not None and not step % save_each:
            cv2.imwrite(
                str(frames_directory / f"frame-{step:04}-{info['x_pos']}-{info['y_pos']}.png"), state2img(state),
            )
        if display:
            env.render()

    return steps
