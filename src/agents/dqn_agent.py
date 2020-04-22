"""
Disclaimer:

Some parts of this implementation are inspired from:
https://github.com/sebastianheinz/super-mario-reinforcement-learning/blob/master/agent.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from random import sample, randint, random
from shutil import rmtree

import numpy as np
import tensorflow as tf
from gym import Env

from agents.agent_abc import AgentABC
from constants import PROJECT_DIRECTORY

EXP_DIR: Path = PROJECT_DIRECTORY / "exp" / "rl"


@dataclass
class EpisodeMetrics:
    episode: int
    x_pos: int
    return_: int
    steps: int


class DQNAgentABC(AgentABC, ABC):
    def __init__(self, memory_size: int, env: Env, eps: float, name: str):
        super().__init__(memory_size=memory_size, env=env)
        self.eps = eps
        self.name = name

        self.session = tf.Session()
        self._build_model()

    def act(self, state: np.ndarray, explore: bool) -> int:
        if explore and random() < self.eps:
            return randint(0, self._n_actions - 1)
        q = self._predict_q_values(np.expand_dims(state, 0)).flatten()
        return q.argmax()

    @abstractmethod
    def _build_model(self):
        """Should make the self.input placeholder, and define a self.output layer"""
        self.output = None
        self.input = None
        pass

    def _predict_q_values(self, state: np.ndarray) -> np.ndarray:
        return self._predict_from_fetches(self.output, state)

    def _predict_from_fetches(self, fetches: tf.placeholder, state: np.ndarray) -> np.ndarray:
        return self.session.run(fetches=fetches, feed_dict={self.input: state})


class DQNAgent(DQNAgentABC):

    EPS_INIT: float = 1.0

    EPS_DECAY: float = 0.999975
    EPS_MIN: float = 0.02

    LR: float = 0.00025

    GAMMA: float = 0.90

    MEMORY_SIZE: int = 100_000
    BATCH_SIZE: int = 16
    BURNIN: int = 10_000
    COPY_EACH: int = 10_000
    LEARN_EACH: int = 4
    SAVE_EACH: int = 200_000

    def __init__(self, env: Env, name: str):
        super().__init__(memory_size=self.MEMORY_SIZE, env=env, eps=self.EPS_INIT, name=name)
        self.step: int = 0

    def learn(self):
        self.eps = max(self.EPS_MIN, self.eps * self.EPS_DECAY)
        self.step += 1

        if self.step < self.BURNIN:
            return

        if not self.step % self.COPY_EACH:
            self._copy_model()

        if not self.step % self.SAVE_EACH:
            self.save_model()

        if not self.step % self.LEARN_EACH:
            self._improve_online()

    def register_episode(self, episode_metrics: EpisodeMetrics):
        (episode_summary,) = self.session.run(
            fetches=[self.episode_summary],
            feed_dict={
                self.epsilon: self.eps,
                self.x_pos: episode_metrics.x_pos,
                self.steps_per_episode: episode_metrics.steps,
                self.return_: episode_metrics.return_,
            },
        )
        self.writer.add_summary(episode_summary, episode_metrics.episode)

    def save_model(self):
        self.saver.save(sess=self.session, save_path=str(self._model_dir), global_step=self.step)

    def _copy_model(self):
        self.session.run(
            [
                tf.assign(target_variable, online_variable)
                for (target_variable, online_variable) in zip(
                    tf.trainable_variables("target"), tf.trainable_variables("online")
                )
            ]
        )

    def _improve_online(self):
        batch = sample(self._memory, self.BATCH_SIZE)
        states, next_states, actions_taken, dones, rewards_obtained = map(np.array, zip(*batch))
        target_q = self._infer_q_target(dones, next_states, rewards_obtained)
        summary, _ = self.session.run(
            fetches=[self.summaries, self.train],
            feed_dict={
                self.input: states,
                self.q_targets: np.array(target_q),
                self.actions_taken: np.array(actions_taken),
                self.rewards: np.mean(rewards_obtained),
            },
        )
        self.writer.add_summary(summary, self.step)

    def _infer_q_target(self, dones, next_states, rewards):
        next_q = self._predict_target_q_values(next_states)
        q = self._predict_q_values(next_states)
        actions = np.argmax(q, axis=1)
        target_q = rewards + (1.0 - dones) * self.GAMMA * next_q[np.arange(0, self.BATCH_SIZE), actions]
        return target_q

    def _predict_target_q_values(self, state: np.ndarray) -> np.ndarray:
        return self._predict_from_fetches(self.output_target, state)

    def _build_model(self):
        self._make_models()
        self._make_optimizer()
        self._make_summaries()
        self._make_saver()
        self.session.run(tf.global_variables_initializer())

    def _make_models(self):
        self._make_models_input()
        self._make_model_returns()
        self._make_online_network()
        self._make_target_network()

    def _make_models_input(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=(None,) + self._state_shape)
        self.input_float = tf.to_float(self.input) / 255.0

    def _make_model_returns(self):
        self.q_targets = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.actions_taken = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[])

    def _make_target_network(self):
        self.output_target = self._make_network("target")

    def _make_online_network(self):
        self.output = self._make_network("online")

    def _make_network(self, name: str):
        with tf.variable_scope(name):
            conv_1 = tf.layers.conv2d(
                inputs=self.input_float,
                filters=32,
                kernel_size=8,
                strides=4,
                activation=tf.nn.relu,
                name=f"conv_1_{name}",
            )
            conv_2 = tf.layers.conv2d(
                inputs=conv_1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name=f"conv_2_{name}"
            )
            conv_3 = tf.layers.conv2d(
                inputs=conv_2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, name=f"conv_3_{name}"
            )
            flatten = tf.layers.flatten(inputs=conv_3)
            dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu, name=f"dense_{name}")
            if name == "target":  # We don't want to update target at training, so we stop the gradient descent
                output = tf.stop_gradient(tf.layers.dense(inputs=dense, units=self._n_actions, name=f"output_target"))
            else:
                output = tf.layers.dense(inputs=dense, units=self._n_actions, name=f"output")
            return output

    def _make_optimizer(self):
        self.q_predictions = tf.gather_nd(
            params=self.output,
            indices=tf.stack([tf.range(tf.shape(self.actions_taken)[0]), self.actions_taken], axis=1),
        )
        self.loss = tf.losses.huber_loss(labels=self.q_targets, predictions=self.q_predictions)
        self.train = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss)

    def _make_saver(self):
        self.saver = tf.train.Saver()

    def _make_summaries(self):
        self._make_step_summary()
        self._make_episode_summary()
        self._make_writer()

    def _make_step_summary(self):
        with tf.name_scope("step-metrics"):
            self.summaries = tf.summary.merge(
                [
                    tf.summary.scalar("reward", self.rewards),
                    tf.summary.scalar("loss", self.loss),
                    tf.summary.scalar("max_q", tf.reduce_max(self.output)),
                ]
            )

    def _make_episode_summary(self):
        self.x_pos = tf.placeholder(tf.float16)
        self.return_ = tf.placeholder(tf.float16)
        self.steps_per_episode = tf.placeholder(tf.float16)
        self.epsilon = tf.placeholder(tf.float16)
        with tf.name_scope("episode-metrics"):
            self.episode_summary = tf.summary.merge(
                [
                    tf.summary.scalar("epsilon", self.epsilon),
                    tf.summary.scalar("x_pos", self.x_pos),
                    tf.summary.scalar("return", self.return_),
                    tf.summary.scalar("steps", self.steps_per_episode),
                    tf.summary.scalar("return/steps", self.return_ / self.steps_per_episode),
                    tf.summary.scalar("progression", self.x_pos / 31.60),
                ]
            )

    def _make_writer(self):
        rmtree(str(self._logs_dir), ignore_errors=True)
        self.writer = tf.summary.FileWriter(logdir=str(self._logs_dir), graph=self.session.graph)

    @property
    def _logs_dir(self) -> Path:
        return EXP_DIR / f"{self.name}-logs"

    @property
    def _model_dir(self) -> Path:
        return EXP_DIR / f"{self.name}" / "model"
