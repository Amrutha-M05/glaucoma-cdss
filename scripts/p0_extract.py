"""P0.1 - Extract all datasets into one format.

Writes FULL_DIR/<dataset>/<id>.jpg (long side <= 1024 px) and <id>_mask.png
(0 background, 1 disc rim, 2 cup), plus MANIFEST with one row per image:
dataset, image_id, image, mask, label (1 glaucoma, 0 normal, -1 unknown), source_split, role.

role: "dev" = labelled development data, "dev_seg" = masks only (segmentation loss),
      "external" = held-out test sets (RIM-ONE DL, DRISHTI-GS).
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.paths import (CUP, DISC, FULL_DIR, KAGGLE_GLAUCOMA_ZIP, MANIFEST, RAW_DATA_DIR,  # noqa: E402
                        REFUGE2_ZIP)

MAX_SIDE = 1024
FIELDS = ["dataset", "image_id", "image", "mask", "label", "source_split", "role"]


def decode(data, flags):
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def save(dataset, image_id, image, label_map):
    scale = MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        label_map = cv2.resize(label_map, size, interpolation=cv2.INTER_NEAREST)
    out = FULL_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    img_path, mask_path = out / f"{image_id}.jpg", out / f"{image_id}_mask.png"
    cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(mask_path), label_map.astype(np.uint8))
    return img_path.as_posix(), mask_path.as_posix()


def label_map_from(disc, cup):
    m = np.zeros(disc.shape, np.uint8)
    m[disc] = DISC
    m[cup & disc] = CUP
    return m


def origa(zf):
    labels = {r["Filename"].rsplit(".", 1)[0]: int(r["Glaucoma"])
              for r in csv.DictReader(io.StringIO(zf.read("ORIGA/OrigaList.csv").decode()))}
    sets = {r["Filename"].rsplit(".", 1)[0]: r["Set"]
            for r in csv.DictReader(io.StringIO(zf.read("ORIGA/OrigaList.csv").decode()))}
    for stem in sorted(labels):
        img = decode(zf.read(f"ORIGA/Images/{stem}.jpg"), cv2.IMREAD_COLOR)
        m = decode(zf.read(f"ORIGA/Masks/{stem}.png"), cv2.IMREAD_GRAYSCALE)  # 0/1/2 already
        yield "ORIGA", stem, img, m, labels[stem], f"set{sets[stem]}", "dev"


def refuge2(zf):
    for split in ("train", "val", "test"):
        names = sorted(n for n in zf.namelist() if n.startswith(f"REFUGE2/{split}/images/") and not n.endswith("/"))
        masks = {Path(n).stem: n for n in zf.namelist() if n.startswith(f"REFUGE2/{split}/mask/")}
        for n in names:
            stem = Path(n).stem
            img = decode(zf.read(n), cv2.IMREAD_COLOR)
            raw = decode(zf.read(masks[stem]), cv2.IMREAD_GRAYSCALE)  # 0 cup, 128 rim, 255 background
            m = label_map_from(raw < 192, raw < 64)
            label = {"g": 1, "n": 0}.get(stem[0], -1) if split == "train" else -1
            yield "REFUGE2", stem, img, m, label, split, "dev" if label >= 0 else "dev_seg"


def refuge_val(zf):
    index = json.loads(zf.read("REFUGE/val/index.json"))
    for item in sorted(index.values(), key=lambda d: d["ImgName"]):
        stem = Path(item["ImgName"]).stem
        img = decode(zf.read(f"REFUGE/val/Images/{stem}.jpg"), cv2.IMREAD_COLOR)
        m = decode(zf.read(f"REFUGE/val/Masks/{stem}.png"), cv2.IMREAD_GRAYSCALE)  # 0/1/2 already
        yield "REFUGE1-val", stem, img, m, int(item["Label"]), "val", "dev"


def from_manifest(dataset, manifest, threshold):
    with open(manifest) as f:
        for r in csv.DictReader(f):
            img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
            disc = cv2.imread(r["od_mask_path"], cv2.IMREAD_GRAYSCALE) > threshold
            cup = cv2.imread(r["oc_mask_path"], cv2.IMREAD_GRAYSCALE) > threshold
            if disc.shape != img.shape[:2]:  # RIM-ONE masks are stored at image size; guard anyway
                disc = cv2.resize(disc.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                cup = cv2.resize(cup.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
            stem = Path(r["image_path"]).stem
            split = Path(r["image_path"]).parent.parent.name  # e.g. training_set / Training
            yield dataset, stem, img, label_map_from(disc, cup), int(r["label"]), split, "external"


def main():
    rows = []
    with zipfile.ZipFile(KAGGLE_GLAUCOMA_ZIP) as k, zipfile.ZipFile(REFUGE2_ZIP) as r2:
        sources = [
            origa(k), refuge2(r2), refuge_val(k),
            from_manifest("RIM-ONE", RAW_DATA_DIR / "RIM-ONE" / "RIM-ONE_manifest.csv", 127),
            # DRISHTI-GS masks were already thresholded at the authors' >180 consensus level.
            from_manifest("DRISHTI-GS", RAW_DATA_DIR / "DRISHTI-GS" / "DRISHTI-GS_manifest.csv", 127),
        ]
        for source in sources:
            for dataset, image_id, img, m, label, split, role in source:
                img_path, mask_path = save(dataset, image_id, img, m)
                rows.append(dict(dataset=dataset, image_id=image_id, image=img_path, mask=mask_path,
                                 label=label, source_split=split, role=role))
            print(f"{rows[-1]['dataset']}: done ({sum(r['dataset'] == rows[-1]['dataset'] for r in rows)} images)",
                  flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
