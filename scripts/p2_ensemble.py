"""Average the saved predictions of several runs (e.g. seeds) and score the ensemble.

Usage: python scripts/p2_ensemble.py <out_dir> <run_dir> [<run_dir> ...] [--file predictions.csv]
Thresholds are re-fitted on the ensemble's validation predictions (never on test).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.trainer import summarize  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--file", default="predictions.csv")
    a = ap.parse_args()
    keys = ["dataset", "image_id", "split", "label"]
    frames = [pd.read_csv(r / a.file).set_index(keys)["prob"] for r in a.runs]
    ens = pd.concat(frames, axis=1).mean(axis=1).rename("prob").reset_index()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_csv(a.out_dir / "predictions.csv", index=False)
    summarize(ens, f"{a.out_dir.name} (n={len(a.runs)})", a.out_dir,
              {"members": [str(r) for r in a.runs], "source_file": a.file})
