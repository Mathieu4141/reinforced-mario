from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Sprite:
    img: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.h, self.w = self.img.shape

    @staticmethod
    def from_filepath(sprite_filepath: Path, channel: int) -> "Sprite":
        rgba_sprite = cv2.cvtColor(cv2.imread(str(sprite_filepath), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        return Sprite(img=rgba_sprite[:, :, channel], mask=rgba_sprite[:, :, 3])
