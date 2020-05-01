from itertools import chain

import numpy as np

from segmentation.sprites_based.ground_finder import GroundFinder
from segmentation.sprites_based.mario_finder import MarioFinder
from segmentation.sprites_based.object_finder import ObjectFinder
from segmentation.sprites_based.tile_finder import TileFinder


def _infer_dx(image) -> int:
    grounds = image[240 - 15, :, 0]
    black_positions = np.where(grounds == 0)[0]
    for x1, x2 in zip(black_positions, black_positions[1:]):
        if (x2 - x1) == 10:
            return (x1 + 1) % 16


class Segmentator:
    def __init__(self):
        blocking = 125
        self.finders = [
            MarioFinder(250),
            ObjectFinder("goomba", 0, 50, 10, y1=64),
            ObjectFinder("koopa", 0, 75, 1, y1=100),
            ObjectFinder("shell", 0, 100, 1, y1=100),
            ObjectFinder("hard", 0, blocking, 5),
            ObjectFinder("mushroom", 0, 200, 1, y1=120),
            GroundFinder(blocking),
            TileFinder("block", 0, 150, 2, y1=80),
            ObjectFinder("pipe", 1, blocking, 21, y1=130),
            TileFinder("?", 1, 25, 2, y1=80),
        ]
        blockings = 3  # pipe, hard, ground
        assert len(self.finders) == len({f.id for f in chain(self.finders)}) + (blockings - 1)

    def segmentate(self, image: np.ndarray, mario_x: int = None, mario_y: int = None) -> np.ndarray:
        if mario_y is not None:
            mario_y = 240 - mario_y
        rv = np.zeros((image.shape[0], image.shape[1]))
        dx = _infer_dx(image)
        for finder in self.finders:
            for (x, y, w, h) in finder.find_all(image, mario_x, mario_y, dx):
                rv[y : y + h, x : x + w] = finder.id
        return rv
