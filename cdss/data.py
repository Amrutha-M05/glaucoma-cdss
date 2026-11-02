"""ROI dataset shared by every classifier (baselines and BT-ViT).

Each stored ROI is a 384 px crop with ROI_CONTEXT (2.0) x the disc diameter. The model input is
the central ROI_WINDOW (1.4) x diameter window resampled to `size` px, with the training
augmentation of METHODOLOGY.md section 3 folded into a single affine warp (rotation, scale,
centre jitter, flips), followed by photometric jitter.

Preprocessing modes:
  "rgb_clahe"   - ours: RGB, CLAHE on the LAB L channel, ImageNet mean/std normalisation.
  "paper_gray"  - the reference paper (B1): grayscale, 3x3 median filter, CLAHE, min-max to [0,1].
"""
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

ROI_CONTEXT, ROI_WINDOW = 2.0, 1.4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(bgr, mode):
    if mode == "rgb_clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = _CLAHE.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return torch.from_numpy((rgb - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)
    if mode == "paper_gray":
        gray = _CLAHE.apply(cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 3)).astype(np.float32)
        gray = (gray - gray.min()) / max(gray.max() - gray.min(), 1.0)
        return torch.from_numpy(gray)[None]
    raise ValueError(f"unknown preprocessing mode {mode!r}")


def _photometric(bgr):
    x = bgr.astype(np.float32)
    x = x * random.uniform(0.8, 1.2) + random.uniform(-20, 20)  # contrast / brightness
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-5, 5)) % 180  # hue
    hsv[..., 1] *= random.uniform(0.8, 1.2)  # saturation
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    x = np.power(x, random.uniform(0.8, 1.25)) * 255.0  # gamma
    x = np.clip(x, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.5, 1.5))
    if random.random() < 0.3:  # camera / compression variation
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 90)])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


class RoiDataset(Dataset):
    """Rows from roi_manifest.csv. Returns dict(image, label, mask, index)."""

    def __init__(self, rows, train, preprocessing="rgb_clahe", size=224, image_col="roi_image",
                 mask_col="roi_mask"):
        self.rows = list(rows)
        self.train, self.mode, self.size = train, preprocessing, size
        self.image_col, self.mask_col = image_col, mask_col

    def __len__(self):
        return len(self.rows)

    def _warp(self, src_size):
        # Output pixel (u, v) maps to the central window of the stored crop.
        window = src_size * ROI_WINDOW / ROI_CONTEXT
        angle, scale, dx, dy, flip_h, flip_v = 0.0, 1.0, 0.0, 0.0, False, False
        if self.train:
            angle = random.uniform(-20, 20)
            scale = random.uniform(0.9, 1.1)
            disc = src_size / ROI_CONTEXT
            dx, dy = (random.uniform(-0.1, 0.1) * disc for _ in range(2))
            flip_h, flip_v = random.random() < 0.5, random.random() < 0.2
        c = src_size / 2.0
        k = self.size / (window * scale)
        a = math.radians(angle)
        cos, sin = math.cos(a) * k, math.sin(a) * k
        sx, sy = (-1 if flip_h else 1), (-1 if flip_v else 1)
        m = np.array([[sx * cos, sx * sin, 0.0], [-sy * sin, sy * cos, 0.0]], np.float32)
        centre = np.array([c + dx, c + dy], np.float32)
        m[:, 2] = self.size / 2.0 - m[:, :2] @ centre
        return m

    def __getitem__(self, i):
        r = self.rows[i]
        bgr = cv2.imread(r[self.image_col], cv2.IMREAD_COLOR)
        mask = cv2.imread(r[self.mask_col], cv2.IMREAD_GRAYSCALE)
        m = self._warp(bgr.shape[0])
        bgr = cv2.warpAffine(bgr, m, (self.size, self.size), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(mask, m, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=0)
        if self.train:
            bgr = _photometric(bgr)
        return {"image": preprocess(bgr, self.mode), "label": int(r["label"]),
                "mask": torch.from_numpy(mask.astype(np.int64)), "index": i}


def balanced_sampler(labels, num_samples=None, unlabelled_fraction=0.0):
    """Class-balanced sampling (same for every model).

    Unlabelled rows (-1) are drawn with total probability `unlabelled_fraction` (0 = never);
    the epoch is lengthened so the number of labelled draws stays the same.
    """
    labels = np.asarray(labels)
    weights = np.zeros(len(labels), np.float64)
    for c in (0, 1):
        weights[labels == c] = 0.5 * (1 - unlabelled_fraction) / (labels == c).sum()
    if unlabelled_fraction > 0:
        weights[labels < 0] = unlabelled_fraction / (labels < 0).sum()
    n = num_samples or int(round((labels >= 0).sum() / (1 - unlabelled_fraction)))
    return WeightedRandomSampler(torch.from_numpy(weights), n, replacement=True)
