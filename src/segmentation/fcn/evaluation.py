from time import time
from typing import Tuple

from pandas import np

from segmentation.dataset.batch_generator import BatchGenerator
from segmentation.fcn.fcn_abc import FCN_ABC


def _class_maps_to_colored_maps(class_map: np.ndarray) -> np.ndarray:
    return np.argmax(class_map, axis=3)


def evaluate_accuracy(fcn: FCN_ABC, batch_generator: BatchGenerator) -> Tuple[float, float]:
    ac = 0
    n = 1
    mean_time = 0
    for imgs, gts in batch_generator:
        t = time()
        preds = fcn.session.run(fetches=fcn.output_layer, feed_dict={fcn.image_input: imgs},)
        mean_time += (time() - t - mean_time) / n
        gts, preds = map(_class_maps_to_colored_maps, [gts, preds])
        ac += ((gts == preds).mean() - ac) / n
    fps = 1 / mean_time
    print(f"{fcn.name}: {ac:.2%} - {fps:.1f} FPS")
    return ac, fps
