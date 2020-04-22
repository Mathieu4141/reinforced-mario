from pathlib import Path

PROJECT_DIRECTORY: Path = Path(__file__).parent.parent
SPRITES_DIRECTORY: Path = PROJECT_DIRECTORY / "data" / "sprites"
VGG_PATH: Path = PROJECT_DIRECTORY / "data" / "pretrained" / "vgg"
SEGMENTATION_DATASET: Path = PROJECT_DIRECTORY / "data" / "segm-dataset-stacked-84-h-v2"
REPORT_DIR: Path = PROJECT_DIRECTORY / "report" / "resources"
