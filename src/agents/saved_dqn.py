import tensorflow as tf
from gym import Env

from agents.dqn_agent import DQNAgentABC, EXP_DIR


class SavedDQN(DQNAgentABC):
    def __init__(self, name: str, env: Env, eps: float = 0.02):
        super().__init__(memory_size=0, env=env, eps=eps, name=name)

    def _build_model(self):
        ckpt = tf.train.latest_checkpoint(str(EXP_DIR / self.name))
        saver = tf.train.import_meta_graph(ckpt + ".meta")
        graph = tf.get_default_graph()
        try:
            self.input = graph.get_tensor_by_name("input:0")
        except KeyError:
            self.input = graph.get_tensor_by_name("image_input:0")
        self.output = graph.get_tensor_by_name("online/output/BiasAdd:0")
        saver.restore(self.session, ckpt)
