from itertools import count
from time import time

import numpy as np
import tqdm
import yaml
from gym import Env

from agents.dqn_agent import DQNAgent, EpisodeMetrics
from agents.experience import Experience
from constants import PROJECT_DIRECTORY


def train(agent: DQNAgent, env: Env, episodes: int = 10_000):
    display = False

    progression = tqdm.trange(episodes, desc=f"Training {agent.name}", unit="episode")
    fps = 0

    for episode in progression:
        state = env.reset()

        mean_reward = 0
        return_ = 0
        x_pos = 0

        for step in count(1):
            t = time()
            action = agent.act(np.asarray(state), explore=True)
            next_state, reward, done, info = env.step(action)
            agent.memorize(Experience((state, next_state, action, done, reward)))
            state = next_state
            agent.learn()

            mean_reward += (reward - mean_reward) / step
            return_ += reward
            x_pos = max(x_pos, info["x_pos"])
            fps = fps * 0.9 + 0.1 / (time() - t)

            if not step % 100:
                try:
                    display = (
                        yaml.load((PROJECT_DIRECTORY / "display.yml").read_text())
                        .get(agent.name, {})
                        .get("display", False)
                    )
                except:
                    pass
            if display:
                env.render()

            if done or info["flag_get"]:
                break

        progression.set_description(
            f"Training {agent.name}; "
            f"Frames: {agent.step} ({fps:.0f} FPS); "
            f"last progression: {x_pos} ({x_pos/3260:.1%}); "
            f"eps: {agent.eps:.2f}"
        )

        agent.register_episode(EpisodeMetrics(episode=episode, x_pos=x_pos, return_=return_, steps=step))

    agent.save_model()
