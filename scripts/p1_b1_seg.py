"""P1 - B1 stages 1-2: train the paper's segmentation CNN, then extract biomarkers for every image.

Trains on every train-split ROI with a mask (labelled + segmentation-only REFUGE2), selects on
validation mean per-image foreground Dice (disc, cup). Writes:
  seg_metrics.json         per-split / per-dataset disc & cup Dice and IoU (foreground only, per image)
  biomarkers_pred.csv      biomarkers from B1's predicted masks (inputs to the B1 classifier and B5)
  biomarkers_gt.csv        the same biomarkers from expert masks (ceiling / BT-ViT targets)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.biomarkers import compute_biomarkers  # noqa: E402
from cdss.data import RoiDataset  # noqa: E402
from cdss.models.paper_b1 import PaperSegCNN, seg_loss  # noqa: E402
from cdss.paths import CUP, DISC, RUNS_DIR  # noqa: E402
from cdss.trainer import load_rows, set_seed  # noqa: E402

EPOCHS, BATCH, LR, PATIENCE, SEED = 80, 8, 1e-4, 20, 2026
OUT = RUNS_DIR / "p1" / "B1_paper" / f"seed{SEED}"


def dice_iou(pred, gt):
    inter, total, union = (pred & gt).sum(), pred.sum() + gt.sum(), (pred | gt).sum()
    return (2 * inter / total if total else 1.0), (inter / union if union else 1.0)


@torch.no_grad()
def predict_masks(model, rows, device):
    model.eval()
    ds = RoiDataset(rows.to_dict("records"), train=False, preprocessing="paper_gray")
    for b in DataLoader(ds, 16, num_workers=0):
        pred = model(b["image"].to(device))["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
        yield b["image"][:, 0].numpy(), pred, b["mask"].numpy().astype(np.uint8)


def mean_dice(model, rows, device):
    scores = []
    for _, pred, gt in predict_masks(model, rows, device):
        for p, g in zip(pred, gt):
            scores.append((dice_iou(p >= DISC, g >= DISC)[0] + dice_iou(p == CUP, g == CUP)[0]) / 2)
    return float(np.mean(scores))


def main():
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    rows = load_rows()
    train_rows, val_rows = rows[rows.split == "train"], rows[rows.split == "val"]
    model = PaperSegCNN().to(device)
    # --predict-only: reuse the saved best checkpoint (e.g. after an interrupted run) and only extract.
    predict_only = "--predict-only" in sys.argv
    if not predict_only:
        dl = DataLoader(RoiDataset(train_rows.to_dict("records"), train=True, preprocessing="paper_gray"), BATCH,
                        shuffle=True, num_workers=1, persistent_workers=True, drop_last=True)
        opt = torch.optim.Adam(model.parameters(), LR)
    best, best_epoch = -1.0, 0
    for epoch in range(1, 0 if predict_only else EPOCHS + 1):
        t0, losses = time.time(), []
        model.train()
        for b in dl:
            loss = seg_loss(model(b["image"].to(device)), b["mask"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        score = mean_dice(model, val_rows, device)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), OUT / "seg_model.pth")
        print(f"[B1-seg] epoch {epoch:03d} loss={np.mean(losses):.4f} val_mean_dice={score:.4f} "
              f"best={best:.4f}@{best_epoch} ({time.time() - t0:.0f}s)", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break

    if predict_only:  # recover the selected epoch from the interrupted run's log
        log = Path(__file__).resolve().parents[1] / "runs" / "logs" / "p1_b1_seg_train.log"
        found = re.findall(r"best=[0-9.]+@(\d+)", log.read_text()) if log.exists() else []
        best_epoch = int(found[-1]) if found else -1
    model.load_state_dict(torch.load(OUT / "seg_model.pth", map_location=device, weights_only=True))
    pred_rows, gt_rows, seg = [], [], []
    keys = rows[["dataset", "image_id", "split", "label"]].to_dict("records")
    k = 0
    for gray, pred, gt in predict_masks(model, rows, device):
        for g_img, p, g in zip(gray, pred, gt):
            key = keys[k]; k += 1
            gray8 = (g_img * 255).astype(np.uint8)
            pred_rows.append({**key, **compute_biomarkers(p, gray8)})
            gt_rows.append({**key, **compute_biomarkers(g, gray8)})
            (dd, di), (cd, ci) = dice_iou(p >= DISC, g >= DISC), dice_iou(p == CUP, g == CUP)
            seg.append({**key, "disc_dice": dd, "disc_iou": di, "cup_dice": cd, "cup_iou": ci})
    pd.DataFrame(pred_rows).to_csv(OUT / "biomarkers_pred.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(OUT / "biomarkers_gt.csv", index=False)
    seg = pd.DataFrame(seg)
    seg.to_csv(OUT / "seg_per_image.csv", index=False)
    seg["group"] = np.where(seg.split == "external", seg.dataset, seg.split)
    summary = seg.groupby("group")[["disc_dice", "disc_iou", "cup_dice", "cup_iou"]].mean().round(4)
    (OUT / "seg_metrics.json").write_text(json.dumps({"best_epoch": best_epoch, **summary.to_dict("index")}, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main()
