"""Clinical biomarkers computed from a disc/cup label map (0 background, 1 disc rim, 2 cup).

PAPER_FEATURES are the reference paper's set (vCDR, cup/disc area, vertical diameters and
edge-based boundary strength, Eq. 9-11). CLINICAL_FEATURES add the measures BT-ViT uses
(area CDR, hCDR, ISNT rim fractions, rim ratio). The ISNT split uses the image diagonals; nasal
and temporal are only used as a sum, so eye laterality (not available for every dataset) does
not matter.
"""
import cv2
import numpy as np

from cdss.paths import CUP, DISC

PAPER_FEATURES = ["vcdr", "cup_area", "disc_area", "cup_vdiam", "disc_vdiam", "disc_edge", "cup_edge"]
CLINICAL_FEATURES = ["cdr_area", "vcdr", "hcdr", "rim_inferior", "rim_superior", "rim_nasal_temporal",
                     "nrr", "isnt_violation", "disc_area", "disc_ellipticity"]


def _extent(mask, axis):
    proj = mask.any(axis=axis)
    idx = np.flatnonzero(proj)
    return float(idx[-1] - idx[0] + 1) if idx.size else 0.0


def _boundary_strength(gray, mask):
    if not mask.any():
        return 0.0
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2)[edge].mean())


def _quadrants(h, w):
    y, x = np.mgrid[:h, :w]
    dy, dx = (y - (h - 1) / 2) / h, (x - (w - 1) / 2) / w
    superior, inferior = (dy < -np.abs(dx)), (dy > np.abs(dx))
    return inferior, superior, ~(superior | inferior)


def compute_biomarkers(label_map, gray=None):
    """Dict of biomarkers; areas are fractions of the image so they are resolution independent."""
    disc, cup = label_map >= DISC, label_map == CUP
    h, w = label_map.shape
    da, ca = float(disc.sum()), float(cup.sum())
    dv, cv_ = _extent(disc, 1), _extent(cup, 1)
    dh, ch = _extent(disc, 0), _extent(cup, 0)
    rim = disc & ~cup
    inferior, superior, nasal_temporal = _quadrants(h, w)
    ri, rs, rnt = float((rim & inferior).sum()), float((rim & superior).sum()), float((rim & nasal_temporal).sum())
    rim_total = max(ri + rs + rnt, 1.0)
    out = {
        "cdr_area": ca / da if da else 0.0, "vcdr": cv_ / dv if dv else 0.0, "hcdr": ch / dh if dh else 0.0,
        "cup_area": ca / (h * w), "disc_area": da / (h * w),
        "cup_vdiam": cv_ / h, "disc_vdiam": dv / h,
        "rim_inferior": ri / rim_total, "rim_superior": rs / rim_total, "rim_nasal_temporal": rnt / rim_total,
        "nrr": (1 + ri + rs) / (1 + rnt),
        # Healthy rims are thickest inferiorly then superiorly (I >= S >= N,T); count violations.
        "isnt_violation": float((ri < rs) + (rs < rnt / 2) + (ri < rnt / 2)),
        "disc_ellipticity": dv / dh if dh else 0.0,
    }
    if gray is not None:
        g = gray.astype(np.float32)
        out["disc_edge"], out["cup_edge"] = _boundary_strength(g, disc), _boundary_strength(g, cup)
    return out
