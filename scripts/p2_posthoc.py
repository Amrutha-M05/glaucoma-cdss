"""Re-evaluate a trained run without retraining: both threshold rules, with and without flip TTA.

Usage: python scripts/p2_posthoc.py <run_dir> [--model btvit|purevit] [--backbone tiny|small]
Writes metrics_notta.json / metrics_tta.json (+ predictions_*.csv) next to the run.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.models.btvit import BTViT, BTViTv2  # noqa: E402
from cdss.models.pure_vit import PureViT  # noqa: E402
from cdss.trainer import evaluate  # noqa: E402

BACKBONES = {"tiny": "deit_tiny_patch16_224.fb_in1k", "small": "deit_small_patch16_224.fb_in1k"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", choices=["btvit", "btvit2", "purevit"], default="btvit")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="tiny")
    a = ap.parse_args()
    model = (BTViT(pretrained=False, backbone=BACKBONES[a.backbone]) if a.model == "btvit"
             else BTViTv2(pretrained=False, backbone=BACKBONES[a.backbone]) if a.model == "btvit2"
             else PureViT(BACKBONES[a.backbone], pretrained=False)).cuda()
    model.load_state_dict(torch.load(a.run_dir / "model.pth", map_location="cuda", weights_only=True))
    name = a.run_dir.parent.name
    for tta in (False, True):
        tag = "tta" if tta else "notta"
        evaluate(model, f"{name} [{tag}]", a.run_dir, "rgb_clahe", tta=tta, filename=f"predictions_{tag}.csv",
                 metrics_file=f"metrics_{tag}.json", extra_info={"tta": tta})
