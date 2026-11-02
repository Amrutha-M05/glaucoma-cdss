"""Optic-disc localiser: a small U-Net that finds the disc on a whole fundus image.

It replaces the reference paper's centre-crop + brightness threshold. Training mixes whole
images with random zoomed-in views so the same model also works on pre-cropped images such
as RIM-ONE DL. Inference runs twice: a coarse pass on the whole image, then a refined pass
on a 3x-disc crop around the first estimate.
"""
import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from cdss.paths import DISC

INPUT = 256
REFINE_SCALE = 3.0


class _Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DiscUNet(nn.Module):
    """4-level U-Net, 16 base channels (~0.5M parameters), one-channel disc logit."""

    def __init__(self, base=16):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList([_Block(3, ch[0])] + [_Block(ch[i], ch[i + 1]) for i in range(3)])
        self.mid = _Block(ch[3], ch[3])
        self.up = nn.ModuleList([_Block(ch[i] * 2, ch[max(i - 1, 0)]) for i in range(3, -1, -1)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.mid(F.max_pool2d(x, 2))
        for block, skip in zip(self.up, reversed(skips)):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.out(x)


def square_crop(image, cx, cy, side, out_size, interp=cv2.INTER_AREA):
    """Square crop centred on (cx, cy), zero-padded where it leaves the image, resized to out_size."""
    half = side / 2.0
    src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
    dst = np.float32([[0, 0], [out_size, 0], [0, out_size]])
    return cv2.warpAffine(image, cv2.getAffineTransform(src, dst), (out_size, out_size), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def disc_geometry(disc_mask):
    """Centre (x, y) and diameter (larger bounding-box side) of the largest disc component, or None."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(disc_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[k, :4]
    return x + w / 2.0, y + h / 2.0, float(max(w, h))


def to_tensor(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy((rgb - 0.5) / 0.25).permute(2, 0, 1)


class LocalizerDataset(Dataset):
    """zoom=None: whole image (eval) or random views (train); zoom=k: fixed k x disc-diameter view."""

    def __init__(self, rows, train, zoom=None):
        self.rows, self.train, self.zoom = list(rows), train, zoom

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        disc = (cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) >= DISC).astype(np.uint8)
        h, w = disc.shape
        geo = disc_geometry(disc)
        zoom = self.zoom if self.zoom is not None else (
            math.exp(random.uniform(math.log(1.05), math.log(6.0))) if self.train and random.random() < 0.6 else None)
        if geo is not None and zoom is not None:
            # Zoomed-in view: crop side 1.05-6x the disc (log-uniform, so tight, pre-cropped-style
            # views such as RIM-ONE DL are common), disc kept inside the crop.
            cx, cy, d = geo
            side = d * zoom
            slack = max(side / 2 - d / 2, 0)
            rng = random if self.train else random.Random(i)  # fixed offsets for evaluation views
            cx += rng.uniform(-slack, slack) * 0.8
            cy += rng.uniform(-slack, slack) * 0.8
        else:
            cx, cy, side = w / 2.0, h / 2.0, float(max(h, w))  # whole image, letterboxed
        if self.train:
            angle = random.uniform(-15, 15)
            m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR)
            disc = cv2.warpAffine(disc, m, (w, h), flags=cv2.INTER_NEAREST)
        x = square_crop(image, cx, cy, side, INPUT)
        y = square_crop(disc, cx, cy, side, INPUT, cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            x = np.clip(x.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                x = cv2.GaussianBlur(x, (5, 5), 0)
        return to_tensor(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).float()[None]


def dice_bce_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


@torch.no_grad()
def _predict_view(model, image, cx, cy, side, device):
    x = square_crop(image, cx, cy, side, INPUT)
    prob = torch.sigmoid(model(to_tensor(x)[None].to(device)))[0, 0].cpu().numpy()
    geo = disc_geometry(prob > 0.5)
    if geo is None:
        return None
    px, py, pd = geo
    scale = side / INPUT
    return cx - side / 2 + px * scale, cy - side / 2 + py * scale, pd * scale


@torch.no_grad()
def localize(model, image, device):
    """Return (cx, cy, diameter, ok) in image pixels; falls back to the image centre if nothing is found."""
    h, w = image.shape[:2]
    coarse = _predict_view(model, image, w / 2.0, h / 2.0, float(max(h, w)), device)
    if coarse is None:
        return w / 2.0, h / 2.0, 0.25 * min(h, w), False
    cx, cy, d = coarse
    fine = _predict_view(model, image, cx, cy, max(d * REFINE_SCALE, 64.0), device)
    return (*fine, True) if fine is not None else (cx, cy, d, True)
