from abc import abstractmethod, ABC
from shutil import rmtree
from time import time

import tensorflow as tf
from tensorflow.layers import conv2d
from tensorflow.python.ops.ragged.ragged_util import repeat
from tqdm import tqdm, trange

from constants import PROJECT_DIRECTORY, VGG_PATH
from segmentation.dataset.batch_generator import BatchGenerator

SEGM_EXP_DIR = PROJECT_DIRECTORY / "exp" / "segmentation"


class FCN_ABC(ABC):

    KEEP_PROB: float = 0.5

    def __init__(self, n_classes: int, name: str):
        self.name = name
        self.step = 0
        self.n_classes = n_classes
        self.session = tf.Session()
        self._build_model()
        self._make_comparison_operation()

    @abstractmethod
    def _build_model(self):
        self.image_input, self.ground_truths, self.output_layer = None

    def _class_maps_to_colored_maps(self, class_map):
        gray_gts = tf.argmax(class_map, axis=3, output_type=tf.int32) * (255 // self.n_classes)
        temp = tf.expand_dims(gray_gts, 3)
        rgb_gt = repeat(temp, 3, axis=3)
        return tf.cast(rgb_gt, dtype=tf.uint8)

    def _make_comparison_operation(self):
        if hasattr(self, "comparison"):
            return
        self.comparison = tf.concat(
            (
                tf.cast(self.image_input, dtype=tf.uint8),
                self._class_maps_to_colored_maps(self.ground_truths),
                self._class_maps_to_colored_maps(self.output_layer),
            ),
            axis=2,
        )


class TrainableFCN(FCN_ABC, ABC):
    LR: float = 0.001
    SIDE: int = 84
    HEIGHT: int = SIDE * 4
    WIDTH: int = SIDE

    def train(self, epochs: int, train_batch_generator: BatchGenerator, validation_batch_generator: BatchGenerator):

        self.step = 0
        self._testing_step(validation_batch_generator)
        for epoch in trange(epochs, desc=self.name, unit="epoch"):
            self._training_step(train_batch_generator)
            self._testing_step(validation_batch_generator)

            self.save()

    def _training_step(self, train_batch_generator):
        tqdm_b_g = tqdm(train_batch_generator, desc=self.name, unit="batch", leave=False)
        for X_batch, gt_batch in tqdm_b_g:
            t = time()
            summaries, _ = self.session.run(
                [self.summaries, self.train_op], feed_dict={self.image_input: X_batch, self.ground_truths: gt_batch},
            )

            tqdm_b_g.set_description(f"Training (graph FPS: {len(X_batch)/ (time() - t):.1f})")

            self.step += 1

            self.train_writer.add_summary(summaries, self.step)

    def _testing_step(self, validation_batch_generator):
        images, gts = next(iter(validation_batch_generator))
        summary = self.session.run(
            fetches=self.validation_summaries, feed_dict={self.image_input: images, self.ground_truths: gts},
        )
        self.validation_writer.add_summary(summary, self.step)

    def save(self):
        self.saver.save(sess=self.session, save_path=str(SEGM_EXP_DIR / self.name / "fcn"), global_step=self.step)

    def _build_model(self):
        self._make_conv_layers()
        self._make_deconv_layers()
        self._make_optimizer()
        self._make_loggers()
        self.session.run(tf.global_variables_initializer())

    def _make_optimizer(self):
        self.ground_truths = tf.placeholder(
            tf.float32, [None, self.HEIGHT, self.WIDTH, self.n_classes], name="ground_truths"
        )
        logits = tf.reshape(self.output_layer, (-1, self.n_classes), name="fcn_logits")
        correct_label_reshaped = tf.reshape(self.ground_truths, (-1, self.n_classes))

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
        self.loss_op = tf.reduce_mean(cross_entropy_loss, name="fcn_loss")

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_op, name="fcn_train_op")

        self.saver = tf.train.Saver()

    def _make_loggers(self):

        logdir = SEGM_EXP_DIR / f"{self.name}-logs"

        loss_summary = tf.summary.scalar("loss", self.loss_op)
        self.summaries = tf.summary.merge([loss_summary])
        rmtree(str(logdir), ignore_errors=True)
        self.train_writer = tf.summary.FileWriter(logdir=str(logdir / "train"), graph=self.session.graph)

        self._make_comparison_operation()
        self.validation_summaries = tf.summary.merge(
            [tf.summary.image("input/gt/pred", self.comparison, max_outputs=12), loss_summary]
        )
        self.validation_writer = tf.summary.FileWriter(logdir=str(logdir / "validation"))

    @abstractmethod
    def _make_conv_layers(self):
        pass

    @abstractmethod
    def _make_deconv_layers(self):
        pass
