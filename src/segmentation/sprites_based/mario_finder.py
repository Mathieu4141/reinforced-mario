from typing import Iterable, Tuple

import numpy as np

from segmentation.sprites_based.object_finder import ObjectFinder


class MarioFinder(ObjectFinder):
    def __init__(self, id_: int):
        super().__init__("mario", 0, id_, 1, x2=150)

    def find_all(self, img: np.ndarray, mario_x: int, mario_y: int, dx: int) -> Iterable[Tuple[int, int, int, int]]:
        try:
            img = img[:, self.x1 :, self.channel]

            for sprite in self.sprites:
                matches = self._find_matches(img, sprite, dx)
                for y, x in matches:
                    return [(self.x1 + x, y, sprite.w, sprite.h)]
        except Exception as e:
            print(e)
            pass

        return []
