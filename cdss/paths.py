"""Filesystem locations shared by all CDSS scripts."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
GLAUCOMA_DIR = PROJECT_DIR.parent
RAW_DATA_DIR = GLAUCOMA_DIR / "data"
DOWNLOADS = Path.home() / "Downloads"

KAGGLE_GLAUCOMA_ZIP = DOWNLOADS / "archive (11).zip"  # arnavjain1/glaucoma-datasets
REFUGE2_ZIP = DOWNLOADS / "archive (12).zip"  # victorlemosml/refuge2

DATA_DIR = RAW_DATA_DIR / "cdss"
FULL_DIR = DATA_DIR / "full"  # unified images (long side <= 1024) and label maps
ROI_DIR = DATA_DIR / "roi"  # disc-centred crops
MANIFEST = DATA_DIR / "manifest.csv"
SPLITS = DATA_DIR / "splits.csv"
ROI_MANIFEST = DATA_DIR / "roi_manifest.csv"

RUNS_DIR = PROJECT_DIR / "runs"

# Label-map convention for every mask written by the pipeline.
BACKGROUND, DISC, CUP = 0, 1, 2  # disc region = label >= DISC
