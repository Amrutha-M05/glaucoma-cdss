"""P0.3 - Train the optic-disc localiser on the development train split; select on val.

Validation uses two views of every val image: the whole fundus (letterboxed) and a simulated
tight disc crop (1.15x the disc diameter, like a pre-cropped dataset). The checkpoint with the
best mean Dice over both views is kept, so model selection never touches the external sets.
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.localizer import DiscUNet, LocalizerDataset, dice_bce_loss  # noqa: E402
from cdss.paths import RUNS_DIR, SPLITS  # noqa: E402

EPOCHS, BATCH, LR, SEED = 40, 16, 1e-3, 2026
TIGHT_ZOOM = 1.15


@torch.no_grad()
def val_dice(model, loader, device):
    inter = total = 0.0
    for x, y in loader:
        pred = (model(x.to(device)) > 0).float()
        y = y.to(device)
        inter += (pred * y).sum().item()
        total += (pred.sum() + y.sum()).item()
    return 2 * inter / max(total, 1)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    df = pd.read_csv(SPLITS)
    val_rows = df[df.split == "val"].to_dict("records")
    train = LocalizerDataset(df[df.split == "train"].to_dict("records"), train=True)
    # Only 7.4 GB RAM on this machine: one loader worker for training, none for validation.
    train_dl = DataLoader(train, BATCH, shuffle=True, num_workers=1, drop_last=True, persistent_workers=True)
    val_whole = DataLoader(LocalizerDataset(val_rows, train=False), BATCH)
    val_tight = DataLoader(LocalizerDataset(val_rows, train=False, zoom=TIGHT_ZOOM), BATCH)

    device = "cuda"
    model = DiscUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(train_dl))
    # fp32 throughout: fp16 autocast yields non-finite activations on the GTX 1650 Ti.
    out = RUNS_DIR / "localizer"
    out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0, losses = time.time(), []
        model.train()
        for x, y in train_dl:
            loss = dice_bce_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        whole, tight = val_dice(model, val_whole, device), val_dice(model, val_tight, device)
        score = (whole + tight) / 2
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "disc_unet.pth")
        print(f"epoch {epoch:02d} loss={np.mean(losses):.4f} val_dice whole={whole:.4f} tight={tight:.4f} "
              f"best_mean={best:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
