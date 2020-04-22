import tensorflow as tf

from segmentation.fcn.fcn_abc import FCN_ABC, SEGM_EXP_DIR


class SavedFCN(FCN_ABC):
    def _build_model(self):
        ckpt = tf.train.latest_checkpoint(str(SEGM_EXP_DIR / self.name))
        saver = tf.train.import_meta_graph(ckpt + ".meta")
        graph = tf.get_default_graph()
        self.image_input = graph.get_tensor_by_name("image_input:0")
        self.ground_truths = graph.get_tensor_by_name("ground_truths:0")
        self.output_layer = graph.get_tensor_by_name("output_layer/BiasAdd:0")
        saver.restore(self.session, ckpt)
