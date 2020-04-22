from typing import Iterable, Tuple

import cv2
import numpy as np
from cv2.cv2 import matchTemplate

from constants import SPRITES_DIRECTORY
from segmentation.sprites_based.sprite import Sprite


class ObjectFinder:
    def __init__(
        self,
        name: str,
        channel: int,
        id_: int,
        threshold: float,
        x1: int = 0,
        x2: int = 256,
        y1: int = 32,
        y2: int = 240,
        debug: bool = False,
    ):
        self.debug = debug
        self.channel = channel
        self.threshold = threshold
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.id = id_
        self.sprites = self._load_sprites(name)

    def find_all(self, img: np.ndarray, mario_x: int, mario_y: int, dx: int) -> Iterable[Tuple[int, int, int, int]]:
        img = self._preprocess_img(img)
        for sprite in self.sprites:
            matches = self._find_matches(img, sprite, dx)
            for y, x in matches:
                yield self.x1 + x, self.y1 + y, sprite.w, sprite.h

    def _find_matches(self, img: np.ndarray, sprite: Sprite, dx: int) -> Iterable[Tuple[int, int]]:
        scores = matchTemplate(img, sprite.img, cv2.TM_SQDIFF, mask=sprite.mask)
        if self.debug:
            print(list(sorted(scores.flatten()))[:5], scores.mean())
        return zip(*(np.where(scores <= self.threshold)))

    def _preprocess_img(self, img: np.ndarray):
        return img[self.y1 : self.y2, self.x1 : self.x2, self.channel]

    def _load_sprites(self, name):
        return [
            Sprite.from_filepath(sprite_filepath, self.channel)
            for sprite_filepath in sorted(SPRITES_DIRECTORY.glob(f"sprite-{name}-*.png"))
        ]
