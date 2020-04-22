from typing import Iterable, Tuple

import numpy as np

from segmentation.sprites_based.object_finder import ObjectFinder
from segmentation.sprites_based.sprite import Sprite


class TileFinder(ObjectFinder):
    def _find_matches(self, img: np.ndarray, sprite: Sprite, dx: int) -> Iterable[Tuple[int, int]]:
        h, w = img.shape
        for x in range(dx, w - 15, 16):
            for y in range(0, h - 15, 16):
                if (img[6 + y : y + 10, 6 + x : x + 10] == sprite.img[6:10, 6:10]).all():
                    yield y, x
