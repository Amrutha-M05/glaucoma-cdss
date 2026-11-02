"""Training / evaluation loop shared by every classifier, so all models get the same protocol.

- Labelled training rows only (segmentation-only rows are used by models that request them).
- Class-balanced sampling, cross-entropy (+ any auxiliary losses the model returns).
- Model selection on pooled validation AUC; early stopping.
- Thresholds fixed on validation: Youden (primary) and max-accuracy (reported alongside);
  predictions saved for val, test and both external sets.
Optional, applied identically to any model: EMA of weights, label smoothing, flip test-time
augmentation (TTA).
"""
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from cdss import console
from cdss.data import RoiDataset, balanced_sampler
from cdss.metrics import accuracy_threshold, classification_report, youden_threshold
from cdss.paths import ROI_MANIFEST


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_rows():
    return pd.read_csv(ROI_MANIFEST)


@torch.no_grad()
def predict(model, rows, preprocessing, device, batch=32, extra=None, tta=False):
    """Glaucoma probability per row. `extra(batch, rows)` may add model inputs (e.g. biomarkers).

    tta=True averages the probabilities of the image and its horizontal mirror (a left/right
    eye flip, which is anatomically plausible).
    """
    model.eval()
    ds = RoiDataset(rows.to_dict("records"), train=False, preprocessing=preprocessing)
    probs = []
    for b in DataLoader(ds, batch, num_workers=0):
        kwargs = extra(b, rows) if extra else {}
        x = b["image"].to(device)
        p = torch.softmax(model(x, **kwargs)["logits"].float(), 1)[:, 1]
        if tta:
            p = (p + torch.softmax(model(x.flip(-1), **kwargs)["logits"].float(), 1)[:, 1]) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def train_classifier(model, name, out_dir, *, preprocessing="rgb_clahe", epochs=50, batch=16, lr=1e-4,
                     weight_decay=0.05, optimizer="adamw", patience=15, seed=2026, extra=None,
                     aux_loss=None, unlabelled_fraction=0.0, lr_new=None, ema_decay=None,
                     label_smoothing=0.0, tta=False, device="cuda"):
    """Train `model` (returns {"logits", ...}) and write predictions + metrics to out_dir.

    unlabelled_fraction > 0 also draws segmentation-only rows (label -1), which contribute to
    `aux_loss` but not to the classification loss. If the model defines param_groups() and
    lr_new is given, new layers use lr_new and pretrained ones use lr. With ema_decay, the
    exponential moving average of the weights is what gets validated, selected and saved.
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    keep = (rows.split == "train") & ((rows.label >= 0) | (unlabelled_fraction > 0))
    train_rows = rows[keep].reset_index(drop=True)
    val_rows = rows[rows.split == "val"].reset_index(drop=True)
    ds = RoiDataset(train_rows.to_dict("records"), train=True, preprocessing=preprocessing)
    dl = DataLoader(ds, batch, sampler=balanced_sampler(train_rows.label.values, unlabelled_fraction=unlabelled_fraction),
                    num_workers=1, persistent_workers=True, drop_last=True)

    model.to(device)
    if lr_new is not None and hasattr(model, "param_groups"):
        params = model.param_groups(lr_new=lr_new, lr_pretrained=lr)
        max_lr = [g["lr"] for g in params]
    else:
        params, max_lr = [p for p in model.parameters() if p.requires_grad], lr
    opt = (torch.optim.Adam(params, lr) if optimizer == "adam"
           else torch.optim.AdamW(params, lr, weight_decay=weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=epochs * len(dl), pct_start=0.1)
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
           if ema_decay else None)
    console.banner(f"Training {name}", f"output -> {out_dir}")
    console.kv([
        ("parameters", f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"),
        ("training images", f"{(train_rows.label >= 0).sum()} labelled (glaucoma {(train_rows.label == 1).sum()})"
         + (f" + {(train_rows.label < 0).sum()} segmentation-only" if unlabelled_fraction > 0 else "")),
        ("validation images", f"{len(val_rows)} (glaucoma {(val_rows.label == 1).sum()})"),
        ("epochs / batch", f"{epochs} (early stop after {patience} without improvement) / {batch}"),
        ("optimiser", f"{optimizer}, lr {lr:g}" + (f" (new layers {lr_new:g})" if lr_new else "")
         + (f", weight decay {weight_decay:g}" if optimizer != "adam" else "")),
        ("extras", ", ".join(x for x in [f"EMA {ema_decay}" if ema_decay else "",
                                         f"label smoothing {label_smoothing}" if label_smoothing else "",
                                         "flip TTA" if tta else "", f"seed {seed}"] if x)),
    ])
    console.section("progress (model selected on validation AUC)")
    best_auc, best_epoch, log, t_start = -1.0, 0, [], time.time()
    for epoch in range(1, epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for b in dl:
            kwargs = extra(b, train_rows) if extra else {}
            out = model(b["image"].to(device), **kwargs)
            labels = b["label"].to(device)
            labelled = labels >= 0
            loss = (F.cross_entropy(out["logits"][labelled], labels[labelled], label_smoothing=label_smoothing)
                    if labelled.any() else out["logits"].sum() * 0.0)
            if aux_loss is not None:
                loss = loss + aux_loss(out, b, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            losses.append(loss.item())
        eval_model = ema.module if ema is not None else model
        val_probs = predict(eval_model, val_rows, preprocessing, device, extra=extra, tta=tta)
        auc = classification_report(val_rows.label.values, val_probs, 0.5)["auc"]
        improved = auc > best_auc
        if improved:
            best_auc, best_epoch = auc, epoch
            torch.save(eval_model.state_dict(), out_dir / "model.pth")
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": auc})
        console.epoch_line(name, epoch, epochs, float(np.mean(losses)), auc, best_auc, best_epoch,
                           time.time() - t0, time.time() - t_start, improved)
        if epoch - best_epoch >= patience:
            print(f"  early stop: no improvement for {patience} epochs", flush=True)
            break
    print(f"  training finished in {console.fmt_seconds(time.time() - t_start)}; "
          f"best validation AUC {best_auc:.4f} at epoch {best_epoch}", flush=True)
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    final = ema.module if ema is not None else model
    final.load_state_dict(torch.load(out_dir / "model.pth", map_location=device, weights_only=True))
    return evaluate(final, name, out_dir, preprocessing, device, extra=extra, tta=tta,
                    extra_info={"best_epoch": best_epoch, "tta": tta, "ema_decay": ema_decay,
                                "label_smoothing": label_smoothing})


def evaluate(model, name, out_dir, preprocessing, device="cuda", extra=None, extra_info=None, tta=False,
             filename="predictions.csv", metrics_file="metrics.json"):
    rows = load_rows()
    rows = rows[(rows.split != "train") & (rows.label >= 0)].reset_index(drop=True)
    rows["prob"] = predict(model, rows, preprocessing, device, extra=extra, tta=tta)
    rows[["dataset", "image_id", "split", "label", "prob"]].to_csv(out_dir / filename, index=False)
    return summarize(rows, name, out_dir, extra_info, filename=metrics_file)


def summarize(rows, name, out_dir, extra_info=None, filename="metrics.json"):
    """Metrics per split and per external dataset, thresholds fixed on pooled validation."""
    val = rows[rows.split == "val"]
    thr = youden_threshold(val.label.values, val.prob.values)
    thr_acc = accuracy_threshold(val.label.values, val.prob.values)
    groups = {"val": val, "test": rows[rows.split == "test"]}
    for ds in ("RIM-ONE", "DRISHTI-GS"):
        groups[ds] = rows[(rows.split == "external") & (rows.dataset == ds)]
    result = {"model": name, "threshold": thr, "threshold_acc": thr_acc, **(extra_info or {}),
              "metrics": {g: classification_report(d.label.values, d.prob.values, thr) for g, d in groups.items()},
              "metrics_accthr": {g: classification_report(d.label.values, d.prob.values, thr_acc)
                                 for g, d in groups.items()}}
    (out_dir / filename).write_text(json.dumps(result, indent=2))
    console.section(f"results: {name}  (thresholds fitted on validation: Youden {thr:.3f}, max-acc {thr_acc:.3f})")
    console.table(
        ["set", "n", "glaucoma", "AUC", "acc (Youden)", "sens", "spec", "acc (max-acc)", "F1 (max-acc)"],
        [[g, m["n"], m["positives"], f"{m['auc']:.3f}", f"{m['accuracy']:.3f}", f"{m['sensitivity']:.3f}",
          f"{m['specificity']:.3f}", f"{result['metrics_accthr'][g]['accuracy']:.3f}",
          f"{result['metrics_accthr'][g]['f1']:.3f}"] for g, m in result["metrics"].items()])
    print(f"  saved: {out_dir / filename}", flush=True)
    return result
