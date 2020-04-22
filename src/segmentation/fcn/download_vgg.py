from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

from constants import VGG_PATH


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize // 1_000_000
        self.update(b * bsize // 1_000_000 - self.n)


def download_vgg():
    VGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp_dir, TqdmUpTo(unit="MB", desc="Downloading vgg.zip") as t:
        vgg_zip = temp_dir + "/vgg.zip"
        urlretrieve(
            "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip", vgg_zip, reporthook=t.update_to
        )

        with ZipFile(vgg_zip, "r") as zip_ref:
            zip_ref.extractall(VGG_PATH.parent)
            zip_ref.close()


if __name__ == "__main__":
    download_vgg()
