"""P2 - Train BT-ViT or the pure-ViT baseline under the shared protocol (cdss/config.py).

Usage: python scripts/p2_train_btvit.py [--model btvit|purevit] [--arch v1|v2] [--seed N] [--tag NAME]
                                         [--lambda-bio X] [--tta] [--ema D] [--label-smoothing E]
                                         [--open-gates] [--no-clinical-tokens] [--no-rim-bias] [--scratch]
Outputs go to runs/p2/<BTViT|PureViT>_<tag>/seed<N>/ (model.pth, metrics.json, predictions.csv,
train_log.csv and, for BT-ViT, seg_metrics.json). The final model is
`--arch v2 --lambda-bio 1.0 --tta --tag v2_bio1`; prefer `python run.py train` for the standard runs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss import console  # noqa: E402
from cdss.config import BACKBONES, BTVIT, TRAIN  # noqa: E402
from cdss.data import RoiDataset  # noqa: E402
from cdss.models.btvit import FEATURES, BTViT, BTViTv2, btvit_aux_loss, soft_biomarkers  # noqa: E402
from cdss.models.pure_vit import PureViT  # noqa: E402
from cdss.paths import CUP, DISC, RUNS_DIR  # noqa: E402
from cdss.trainer import load_rows, set_seed, train_classifier  # noqa: E402


@torch.no_grad()
def expert_biomarkers(rows, device):
    """Biomarkers of the expert masks (no augmentation), via the same differentiable definition."""
    ds = RoiDataset(rows.to_dict("records"), train=False)
    feats = []
    for b in DataLoader(ds, 32, num_workers=0):
        onehot = F.one_hot(b["mask"].to(device), 3).permute(0, 3, 1, 2).float()
        feats.append(soft_biomarkers(onehot[:, 1] + onehot[:, 2], onehot[:, 2], BTViT.gray(b["image"].to(device))).cpu())
    return torch.cat(feats)


@torch.no_grad()
def evaluate_segmentation(model, out_dir, device):
    """Per-image foreground Dice/IoU of BT-ViT's own disc and cup masks + biomarker error."""
    model.eval()
    rows = load_rows()
    rows = rows[rows.split != "train"].reset_index(drop=True)
    ds = RoiDataset(rows.to_dict("records"), train=False)
    recs, k = [], 0
    for b in DataLoader(ds, 32, num_workers=0):
        x, gt = b["image"].to(device), b["mask"].to(device)
        out = model(x)
        pred = out["seg_logits"].argmax(1)
        onehot = F.one_hot(gt, 3).permute(0, 3, 1, 2).float()
        target = soft_biomarkers(onehot[:, 1] + onehot[:, 2], onehot[:, 2], BTViT.gray(x))
        for i in range(len(x)):
            r = rows.iloc[k]; k += 1
            rec = {"dataset": r.dataset, "image_id": r.image_id, "split": r.split}
            for name, p, g in (("disc", pred[i] >= DISC, gt[i] >= DISC), ("cup", pred[i] == CUP, gt[i] == CUP)):
                inter, tot, uni = (p & g).sum().item(), (p.sum() + g.sum()).item(), (p | g).sum().item()
                rec[f"{name}_dice"], rec[f"{name}_iou"] = (2 * inter / tot if tot else 1.0), (inter / uni if uni else 1.0)
            for j, f in enumerate(("vcdr", "cdr_area")):
                idx = FEATURES.index(f)
                rec[f"{f}_abs_err"] = abs(out["biomarkers"][i, idx] - target[i, idx]).item()
            recs.append(rec)
    seg = pd.DataFrame(recs)
    seg.to_csv(out_dir / "seg_per_image.csv", index=False)
    seg["group"] = np.where(seg.split == "external", seg.dataset, seg.split)
    summary = seg.groupby("group")[["disc_dice", "disc_iou", "cup_dice", "cup_iou", "vcdr_abs_err",
                                    "cdr_area_abs_err"]].mean().round(4)
    (out_dir / "seg_metrics.json").write_text(json.dumps(summary.to_dict("index"), indent=2))
    console.section("segmentation by the model's own decoder (per-image foreground Dice / IoU)")
    console.table(["set", "disc Dice", "disc IoU", "cup Dice", "cup IoU", "vCDR err", "CDR err"],
                  [[g, *(f"{v:.3f}" for v in r)] for g, r in summary.iterrows()])
    print(f"  saved: {out_dir / 'seg_metrics.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["btvit", "purevit"], default="btvit")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="tiny")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1",
                    help="v1: conv-stem BT-ViT (8 blocks); v2: full pretrained DeiT + clinical tokens in blocks 9-12")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--no-clinical-tokens", action="store_true")
    ap.add_argument("--no-rim-bias", action="store_true")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--lambda-seg", type=float, default=BTVIT["lambda_seg"])
    ap.add_argument("--lambda-bio", type=float, default=BTVIT["lambda_bio"])
    ap.add_argument("--open-gates", action="store_true",
                    help="v2 only: no weight decay on alpha/beta/clin_gate and clinical-token gate starting at -2")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--ema", type=float, default=None, help="EMA decay of weights, e.g. 0.999")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    a = ap.parse_args()
    set_seed(a.seed)
    device = "cuda"
    prefix = "BTViT" if a.model == "btvit" else "PureViT"
    out = RUNS_DIR / "p2" / f"{prefix}_{a.tag}" / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    common = dict(preprocessing=TRAIN["preprocessing"], epochs=a.epochs, batch=TRAIN["batch"], lr=TRAIN["lr"],
                  lr_new=TRAIN["lr_new"], weight_decay=TRAIN["weight_decay"], patience=TRAIN["patience"],
                  seed=a.seed, tta=a.tta, ema_decay=a.ema, label_smoothing=a.label_smoothing)

    if a.model == "purevit":
        model = PureViT(BACKBONES[a.backbone], pretrained=not a.scratch).to(device)
        train_classifier(model, f"{prefix}_{a.tag}", out, **common)
        return

    kw = dict(pretrained=not a.scratch, clinical_tokens=not a.no_clinical_tokens, rim_bias=not a.no_rim_bias,
              backbone=BACKBONES[a.backbone])
    if a.arch == "v2":
        model = BTViTv2(**kw, beta_init=-2.0 if a.open_gates else -4.0, gate_weight_decay=not a.open_gates).to(device)
    else:
        model = BTViT(**kw).to(device)
    rows = load_rows()
    bio = expert_biomarkers(rows[rows.split == "train"], device)
    model.bio_mean.copy_(bio.mean(0).to(device))
    model.bio_std.copy_(bio.std(0).clamp(min=1e-3).to(device))
    print(f"biomarker normalisation fitted on {len(bio)} training expert masks "
          f"(e.g. vCDR mean {bio[:, FEATURES.index('vcdr')].mean():.3f}); "
          f"loss weights: seg {a.lambda_seg}, bio {a.lambda_bio}", flush=True)

    result = train_classifier(model, f"{prefix}_{a.tag}", out, aux_loss=btvit_aux_loss(model, a.lambda_seg, a.lambda_bio),
                              unlabelled_fraction=BTVIT["unlabelled_fraction"], **common)
    # With EMA, train_classifier evaluates the averaged weights; segmentation is scored on the same weights.
    if a.ema:
        model.load_state_dict(torch.load(out / "model.pth", map_location=device, weights_only=True))
    evaluate_segmentation(model, out, device)
    return result


if __name__ == "__main__":
    main()
