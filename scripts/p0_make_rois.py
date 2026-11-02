"""P0.4 - Localise the disc on every image and write disc-centred ROI crops.

Each crop is centred on the *predicted* disc (no ground truth at any stage) with side
ROI_CONTEXT x the predicted diameter, stored at ROI_SIZE px. Training code takes the central
ROI_WINDOW x diameter window (with jitter) from it, so the disc fills ~1/1.4 of the final input width.

Pre-cropped datasets (RIM-ONE DL is published as disc-centred crops with the disc filling ~3/4
of the width) are used as provided: the original image *is* the final ROI window, padded to the
same context. This is a fixed per-dataset rule that uses no per-image masks or labels. Their
localiser-based crops are also written (roi_auto_* columns) for a sensitivity analysis.
Ground-truth disc geometry is recorded only to evaluate the localiser.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.localizer import DiscUNet, disc_geometry, localize, square_crop  # noqa: E402
from cdss.paths import DISC, ROI_DIR, ROI_MANIFEST, RUNS_DIR, SPLITS  # noqa: E402

ROI_CONTEXT, ROI_WINDOW, ROI_SIZE = 2.0, 1.4, 384
PRECROPPED = {"RIM-ONE"}


def write_crop(image, label_map, cx, cy, d, folder, image_id):
    folder.mkdir(parents=True, exist_ok=True)
    img_path, mask_path = folder / f"{image_id}.png", folder / f"{image_id}_mask.png"
    side = ROI_CONTEXT * d
    cv2.imwrite(str(img_path), square_crop(image, cx, cy, side, ROI_SIZE))
    cv2.imwrite(str(mask_path), square_crop(label_map, cx, cy, side, ROI_SIZE, cv2.INTER_NEAREST))
    return img_path.as_posix(), mask_path.as_posix()


def main():
    device = "cuda"
    model = DiscUNet().to(device).eval()
    model.load_state_dict(torch.load(RUNS_DIR / "localizer" / "disc_unet.pth", map_location=device,
                                     weights_only=True))
    df = pd.read_csv(SPLITS)
    rows = []
    for r in df.to_dict("records"):
        image = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        label_map = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        cx, cy, d, ok = localize(model, image, device)
        auto = write_crop(image, label_map, cx, cy, d, ROI_DIR / "auto" / r["dataset"], r["image_id"])
        if r["dataset"] in PRECROPPED:
            h, w = image.shape[:2]
            roi_cx, roi_cy, roi_d, mode = w / 2.0, h / 2.0, min(h, w) / ROI_WINDOW, "dataset_crop"
            primary = write_crop(image, label_map, roi_cx, roi_cy, roi_d, ROI_DIR / r["dataset"], r["image_id"])
        else:
            roi_cx, roi_cy, roi_d, mode, primary = cx, cy, d, "localizer", auto
        gt = disc_geometry(label_map >= DISC)
        gx, gy, gd = gt if gt is not None else (np.nan, np.nan, np.nan)
        rows.append({**r, "roi_image": primary[0], "roi_mask": primary[1], "roi_mode": mode,
                     "roi_cx": roi_cx, "roi_cy": roi_cy, "roi_diameter": roi_d,
                     "roi_auto_image": auto[0], "roi_auto_mask": auto[1],
                     "pred_cx": cx, "pred_cy": cy, "pred_diameter": d, "localized": ok,
                     "gt_cx": gx, "gt_cy": gy, "gt_diameter": gd,
                     "centre_error": np.hypot(roi_cx - gx, roi_cy - gy) / gd, "diameter_ratio": roi_d / gd,
                     "auto_centre_error": np.hypot(cx - gx, cy - gy) / gd, "auto_diameter_ratio": d / gd})
    roi = pd.DataFrame(rows)
    roi.to_csv(ROI_MANIFEST, index=False)

    # ROI quality, in units of the true disc diameter. "Usable" = disc centre within 0.25
    # diameters and size within +/-25%, i.e. the disc lands near-centred at the intended scale.
    for prefix, label in (("", "primary ROI"), ("auto_", "localiser-only ROI")):
        roi["usable"] = (roi[f"{prefix}centre_error"] < 0.25) & roi[f"{prefix}diameter_ratio"].between(0.75, 1.25)
        print(f"--- {label}")
        print(roi.groupby(["split", "dataset"]).agg(
            n=("image_id", "size"), centre_err_median=(f"{prefix}centre_error", "median"),
            centre_err_p95=(f"{prefix}centre_error", lambda s: s.quantile(0.95)),
            diam_ratio_median=(f"{prefix}diameter_ratio", "median"), usable=("usable", "mean")).round(3).to_string())
    print(f"wrote {ROI_MANIFEST}")


if __name__ == "__main__":
    main()
