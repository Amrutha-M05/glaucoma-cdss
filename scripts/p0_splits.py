"""P0.2 - Fixed development splits (METHODOLOGY.md section 2).

Internal test: ORIGA 130, REFUGE 160 (80 per source), stratified by label.
The remaining labelled development images are split 80/20 train/val, stratified per source.
Unlabelled REFUGE2 images (role dev_seg) go to train (segmentation loss only).
External sets are tagged "external" and never enter training or selection.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.paths import MANIFEST, SPLITS  # noqa: E402

SEED = 2026
TEST_SIZE = {"ORIGA": 130, "REFUGE2": 80, "REFUGE1-val": 80}

df = pd.read_csv(MANIFEST)
df["split"] = ""
df.loc[df.role == "external", "split"] = "external"
df.loc[df.role == "dev_seg", "split"] = "train"

for dataset, n_test in TEST_SIZE.items():
    part = df[(df.dataset == dataset) & (df.role == "dev")]
    rest, test = train_test_split(part.index, test_size=n_test, stratify=part.label, random_state=SEED)
    train, val = train_test_split(rest, test_size=0.2, stratify=part.loc[rest, "label"], random_state=SEED)
    df.loc[test, "split"], df.loc[train, "split"], df.loc[val, "split"] = "test", "train", "val"

assert (df.split != "").all()
df.to_csv(SPLITS, index=False)
summary = df.assign(glaucoma=df.label.eq(1)).groupby(["split", "dataset"]).agg(
    images=("image_id", "size"), glaucoma=("glaucoma", "sum"))
print(summary.to_string())
print(f"wrote {SPLITS}")
