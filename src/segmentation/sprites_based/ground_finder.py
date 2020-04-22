from typing import Iterable, Tuple

import numpy as np

from segmentation.sprites_based.sprite import Sprite
from segmentation.sprites_based.tile_finder import TileFinder


class GroundFinder(TileFinder):
    def __init__(self, id_: int):
        super().__init__("ground", 0, id_, 2, y1=240 - 16)

    def find_all(self, img: np.ndarray, mario_x: int, mario_y: int, dx: int) -> Iterable[Tuple[int, int, int, int]]:
        for x, y, w, h in super().find_all(img, mario_x, mario_y, dx):
            yield x, y, w, h
            yield x, y - 16, w, h

        for sprite in self.sprites:
            if self._match_first(img, dx, sprite):
                yield 0, 240 - 16, dx, 16
                yield 0, 240 - 32, dx, 16
            if self._match_last(img, dx, sprite):
                yield 256 - 16 + dx, 240 - 16, 16 - dx, 16
                yield 256 - 16 + dx, 240 - 32, 16 - dx, 16

    def _match_first(self, img: np.ndarray, dx: int, sprite: Sprite) -> bool:
        return (img[240 - 16 : 240, 0:dx, self.channel] == sprite.img[:, 16 - dx :]).all()

    def _match_last(self, img: np.ndarray, dx: int, sprite: Sprite) -> bool:
        return (img[240 - 16 : 240, 256 - 16 + dx :, self.channel] == sprite.img[:, : 16 - dx]).all()
