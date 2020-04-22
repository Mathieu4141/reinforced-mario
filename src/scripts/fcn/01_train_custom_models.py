"""
Use this script to train a FCN model to perform image segmentation

This script will take a few minutes to run, depending on the architecture you choose.

Logs and networks will be saved in /exp/segmentation
Results on the validation data are included in the tensorboard logs, and can be visualized at training.
"""

from segmentation.dataset.batch_generator import BatchGenerator
from segmentation.dataset.sets import Set
from segmentation.fcn.custom_models import ConvolutionConfig, PoolingConfig, FCNConfig, CustomFCN
from utils.reproductibility import seed_all

if __name__ == "__main__":
    seed_all()
    _vg = BatchGenerator(set_=Set.TEST, batch_size=8, randomize_before=False,)

    _tg = BatchGenerator(set_=Set.TRAIN, batch_size=64, randomize_before=True)

    _conf = FCNConfig(
        [
            PoolingConfig([ConvolutionConfig(16, 3)], 2),
            PoolingConfig([ConvolutionConfig(32, 3), ConvolutionConfig(32, 3)], 2),
        ],
        8,
    )

    CustomFCN(len(_vg.color2id), _conf).train(epochs=30, train_batch_generator=_tg, validation_batch_generator=_vg)
