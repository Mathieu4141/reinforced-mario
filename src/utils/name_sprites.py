from shutil import move
from typing import Iterable

from constants import SPRITES_DIRECTORY


def name(indices: Iterable[int], name: str):
    indices = set(indices)
    for img_path in SPRITES_DIRECTORY.glob("sprite-*.png"):
        indice = int(img_path.stem[-3:])
        if indice in indices:
            move(str(img_path), str(img_path.parent / f"sprite-{name}-{indice:03}.png"))


if __name__ == "__main__":
    name(range(0, 58), "mario")
    name(range(157, 161), "goomba")
    name(range(164, 180), "koopa")
    name(range(181, 186), "shell")
    name(range(590, 593), "?")
    name(range(507, 510), "mushroom")
    name({521, 537}, "pipe")
    name({514}, "block")
    name(range(556, 560), "star")
    name({574, 578, 526, 541}, "hard")
