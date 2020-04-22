from enum import Enum
from pathlib import Path
from typing import Iterable, Tuple

from constants import SEGMENTATION_DATASET


class Set(Enum):
    TRAIN = "train"
    TEST = "test"

    @property
    def path(self) -> Path:
        return SEGMENTATION_DATASET / self.value

    @property
    def images_gt_paths(self) -> Iterable[Tuple[Path, Path]]:
        for img_path in self.path.glob("img/frame-*.png"):
            yield img_path, img_path.parent.parent / "gt" / img_path.name
