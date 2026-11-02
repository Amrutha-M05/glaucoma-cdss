"""P1 - Train an image-only baseline (B3 / B4) under the shared protocol.

Usage: python scripts/p1_train_baseline.py <resnet50|densenet121|deit_tiny|vit_small> [--seed N] [--epochs N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.models.baselines import BASELINES, build_baseline  # noqa: E402
from cdss.paths import RUNS_DIR  # noqa: E402
from cdss.trainer import set_seed, train_classifier  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(BASELINES))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    set_seed(a.seed)
    out = RUNS_DIR / "p1" / f"{BASELINES[a.model]}_{a.model}" / f"seed{a.seed}"
    train_classifier(build_baseline(a.model), a.model, out, epochs=a.epochs, seed=a.seed)
