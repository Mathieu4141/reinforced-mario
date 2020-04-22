import tensorflow as tf
from gym import Env

from agents.dqn_agent import DQNAgent
from segmentation.fcn.fcn_abc import SEGM_EXP_DIR


class DQNFromFrozenFCN(DQNAgent):
    MEMORY_SIZE: int = 50_000
    BATCH_SIZE: int = 128
    LEARN_EACH: int = 16

    def __init__(self, env: Env, fcn_name: str, with_conv: bool):
        self.fcn_name = fcn_name
        self.with_conv = with_conv
        super().__init__(env, f"dqn-from-{fcn_name}" + ("-with-conv" * with_conv))

    def _make_models_input(self):
        ckpt = tf.train.latest_checkpoint(str(SEGM_EXP_DIR / self.fcn_name))
        saver = tf.train.import_meta_graph(ckpt + ".meta")
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("image_input:0")
        self.last_pool_layer = tf.stop_gradient(graph.get_tensor_by_name("pool_2/MaxPool:0"))
        saver.restore(self.session, ckpt)

    def _make_network(self, name: str):
        with tf.variable_scope(name):
            layer = self.last_pool_layer

            if self.with_conv:
                layer = tf.layers.conv2d(
                    inputs=layer, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, name=f"trainable_conv",
                )
            flatten = tf.layers.flatten(inputs=layer)
            dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu, name=f"dense_{name}")
            if name == "target":  # We don't want to update target at training, so we stop the gradient descent
                output = tf.stop_gradient(tf.layers.dense(inputs=dense, units=self._n_actions, name=f"output_target"))
            else:
                output = tf.layers.dense(inputs=dense, units=self._n_actions, name=f"output")
            return output
