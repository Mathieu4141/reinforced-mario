import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Iterator, Sequence

import cv2
import numpy as np

from segmentation.dataset.sets import Set
from segmentation.sprites_based.segmentator import Segmentator


@dataclass
class BatchGenerator(Iterable[List[Tuple[np.ndarray, np.ndarray]]]):

    set_: Set
    batch_size: int
    randomize_before: bool
    max_len: int = 1_000_000

    def __post_init__(self):
        segm = Segmentator()
        colors = [0] + list(sorted({finder.id for finder in segm.finders}))
        self.color2id = {c: i for i, c in enumerate(colors)}

    def __iter__(self) -> Iterator[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]:
        img_gt_paths = self._get_paths()

        for batch_start in range(0, len(img_gt_paths), self.batch_size):
            batch_img_gt_paths = img_gt_paths[batch_start : batch_start + self.batch_size]
            yield (
                np.asarray([self._img_from_path(img_path) for img_path, _ in batch_img_gt_paths]),
                np.asarray([self._gt_from_path(gt_path) for _, gt_path in batch_img_gt_paths]),
            )

    def __len__(self) -> int:
        return len(self._get_paths()) // self.batch_size

    def _get_paths(self):
        img_gt_paths = list(sorted(self.set_.images_gt_paths))[: self.max_len]
        if self.randomize_before:
            random.shuffle(img_gt_paths)
        return img_gt_paths

    def _img_from_path(self, img_path: Path):
        # return self._correct_height(cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE))[:, :, np.newaxis]
        # return self._correct_height(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
        # return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    def _gt_from_path(self, gt_path: Path):
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        one_hot = np.zeros((gt.shape[0], gt.shape[1], len(self.color2id)))
        for color, id_ in self.color2id.items():
            one_hot[:, :, id_] = gt == color
        # one_hot = self._correct_height(one_hot)
        # one_hot[240:, :, 0] = 1
        return one_hot

    #
    # @staticmethod
    # def _correct_height(img: np.ndarray):
    #     n_dims = len(img.shape)
    #     return np.pad(img, [(0, 16)] + [(0, 0)] * (n_dims - 1), "constant", constant_values=(1, 1))
    #     # return np.vstack((img, np.zeros((16, img.shape[1], img.shape[2]), dtype=img.dtype)))


if __name__ == "__main__":
    g = BatchGenerator(randomize_before=True, batch_size=16, set_=Set.TEST)
    imgs, gts = next(iter(g))

    import matplotlib.pyplot as plt

    plt.imshow(imgs[0])
    plt.show()
    for i in range(gts.shape[-1]):
        plt.imshow(gts[0, :, :, i], cmap="gray")
        plt.show()

    print(imgs.shape, gts.shape)
