"""P1 - B1 stage 3: the paper's Lightweight ViT + biomarker concatenation classifier.

Biomarkers come from B1's own predicted masks (biomarkers_pred.csv from p1_b1_seg.py) and are
z-scored with training-set statistics. Optimiser per the paper: Adam, lr 1e-4, batch 16.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.biomarkers import PAPER_FEATURES  # noqa: E402
from cdss.models.paper_b1 import PaperViTFusion  # noqa: E402
from cdss.paths import RUNS_DIR  # noqa: E402
from cdss.trainer import set_seed, train_classifier  # noqa: E402

SEED = 2026
OUT = RUNS_DIR / "p1" / "B1_paper" / f"seed{SEED}"


def main():
    set_seed(SEED)
    bio = pd.read_csv(OUT / "biomarkers_pred.csv")
    train = bio[(bio.split == "train") & (bio.label >= 0)]
    mean, std = train[PAPER_FEATURES].mean(), train[PAPER_FEATURES].std().replace(0, 1)
    table = {(r.dataset, r.image_id): ((np.array([getattr(r, f) for f in PAPER_FEATURES]) - mean.values) / std.values)
             for r in bio.itertuples()}

    def extra(batch, rows):
        idx = batch["index"].numpy()
        feats = np.stack([table[(rows.dataset.iat[i], rows.image_id.iat[i])] for i in idx]).astype(np.float32)
        return {"biomarkers": torch.from_numpy(feats).cuda()}

    model = PaperViTFusion(n_biomarkers=len(PAPER_FEATURES))
    train_classifier(model, "B1_paper", OUT, preprocessing="paper_gray", optimizer="adam", lr=1e-4, batch=16,
                     epochs=50, seed=SEED, extra=extra)


if __name__ == "__main__":
    main()
