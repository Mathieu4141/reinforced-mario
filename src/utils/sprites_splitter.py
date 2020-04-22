"""
Those functions were used to extract the sprites from the sheets
"""

from typing import Tuple, List, Iterable

import cv2
import numpy as np

from constants import SPRITES_DIRECTORY
from utils.union_find import UnionFind


def load_sprites_picture(name) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(SPRITES_DIRECTORY / f"{name}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)


def find_active_pixels(image: np.ndarray) -> List[Tuple[int, int]]:
    return list(zip(*np.where(image[:, :, 3] == 255)))


def next_neighbours(x: int, y: int) -> List[Tuple[int, int]]:
    return [(x + dx, y + dy) for dx in [0, 1] for dy in [0, 1] if dx or dy]


def find_blobs(pixels: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    pixels = set(pixels)
    blobs = UnionFind()
    for pixel in pixels:
        for neighbour in next_neighbours(*pixel):
            if neighbour in pixels:
                blobs.union(pixel, neighbour)
    return list(blobs.groups())


def find_boxes(characters_pixels: List[List[Tuple[int, int]]]) -> Iterable[Tuple[int, int, int, int]]:
    for pixels in characters_pixels:
        xmin = min(x for x, y in pixels)
        xmax = max(x for x, y in pixels)
        ymin = min(y for x, y in pixels)
        ymax = max(y for x, y in pixels)
        yield xmin, xmax + 1, ymin, ymax + 1


def split_sprites(name: str, start: int):
    sprites_picture = load_sprites_picture(name)
    active_pixels = find_active_pixels(sprites_picture)
    characters_pixels = find_blobs(active_pixels)
    characters_boxes = list(find_boxes(characters_pixels))
    characters_boxes.sort(key=lambda box: (box[1], box[0]))
    for i, (xmin, xmax, ymin, ymax) in enumerate(characters_boxes, start):
        cv2.imwrite(
            str(SPRITES_DIRECTORY / f"sprite-{i:03}.png"),
            cv2.cvtColor(sprites_picture[xmin:xmax, ymin:ymax, :], cv2.COLOR_RGBA2BGRA),
        )


# if __name__ == "__main__":
#     # split_sprites("characters", 0)
#     split_sprites("all", 500)
