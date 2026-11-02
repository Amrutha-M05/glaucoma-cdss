"""P1 - B5: biomarker-only logistic regression (clinical-rule reference).

Fitted on training-split biomarkers from B1's predicted masks. Two feature sets are reported:
the paper's set and our clinical set. The same model on expert-mask biomarkers is reported as a
ceiling (what perfect segmentation would give these features).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.biomarkers import CLINICAL_FEATURES, PAPER_FEATURES  # noqa: E402
from cdss.paths import RUNS_DIR  # noqa: E402
from cdss.trainer import summarize  # noqa: E402

B1 = RUNS_DIR / "p1" / "B1_paper" / "seed2026"

for source in ("pred", "gt"):
    bio = pd.read_csv(B1 / f"biomarkers_{source}.csv")
    bio = bio[bio.label >= 0]
    for fs_name, feats in (("paper", PAPER_FEATURES), ("clinical", CLINICAL_FEATURES)):
        tr = bio[bio.split == "train"]
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        clf.fit(tr[feats], tr.label)
        rows = bio[bio.split != "train"].copy()
        rows["prob"] = clf.predict_proba(rows[feats])[:, 1]
        name = f"B5_{fs_name}_{source}"
        out = RUNS_DIR / "p1" / name / "seed2026"
        out.mkdir(parents=True, exist_ok=True)
        rows[["dataset", "image_id", "split", "label", "prob"]].to_csv(out / "predictions.csv", index=False)
        summarize(rows, name, out)
