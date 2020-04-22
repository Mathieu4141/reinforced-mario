from dataclasses import dataclass
from typing import List

import tensorflow as tf
from numpy import prod
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.pooling import max_pooling2d
from tensorflow.python.ops.gen_nn_ops import relu

from segmentation.fcn.fcn_abc import TrainableFCN


@dataclass
class ConvolutionConfig:
    n_filters: int
    kernel_size: int

    def __str__(self):
        return f"f{self.n_filters}-k{self.kernel_size}"


@dataclass
class PoolingConfig:
    convolutions: List[ConvolutionConfig]
    strides: int

    def __str__(self):
        return "_".join(map(str, self.convolutions)) + f"_s{self.strides}"


@dataclass
class FCNConfig:
    poolings: List[PoolingConfig]
    deconvolution_kernel: int

    def __str__(self):
        return f"fcn__" + "__".join(map(str, self.poolings)) + f"__d{self.deconvolution_kernel}"


class CustomFCN(TrainableFCN):
    def __init__(self, n_classes: int, config: FCNConfig):
        self.config = config
        super().__init__(n_classes, str(config))

    def _make_conv_layers(self):
        self.image_input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.HEIGHT, self.WIDTH, 3], name="image_input"
        )
        layer = self.image_input
        for p, pooling_config in enumerate(self.config.poolings, 1):
            for c, conv_config in enumerate(pooling_config.convolutions, 1):
                layer = conv2d(
                    inputs=layer,
                    filters=conv_config.n_filters,
                    kernel_size=conv_config.kernel_size,
                    activation=relu,
                    padding="SAME",
                    name=f"conv_{p}_{c}",
                )
            layer = max_pooling2d(inputs=layer, pool_size=2, strides=pooling_config.strides, name=f"pool_{p}")
        self.last_pool_layer = layer

    def _make_deconv_layers(self):
        total_strides = prod([pool.strides for pool in self.config.poolings])
        layer = tf.layers.conv2d(self.last_pool_layer, filters=self.n_classes, kernel_size=1)
        self.output_layer = tf.layers.conv2d_transpose(
            layer,
            filters=self.n_classes,
            kernel_size=self.config.deconvolution_kernel,
            strides=(total_strides, total_strides),
            padding="SAME",
            name="output_layer",
        )
