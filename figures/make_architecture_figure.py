"""Journal-style BT-ViT figure (vector PDF/SVG + 600-dpi PNG, 7.2 in two-column width).

(a) pipeline drawn with feature-map slabs and token sequences;
(b) rim-guided attention: rim prior and the real CLS attention of the last block;
(c) clinical-token construction and the training objective.
Every image, mask, biomarker value, probability and attention map is computed by the trained
BT-ViT on a real ORIGA training image.  Usage: python figures/make_architecture_figure.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdss.data import IMAGENET_MEAN, IMAGENET_STD, RoiDataset  # noqa: E402
from cdss.models.btvit import FEATURES, BTViT  # noqa: E402
from cdss.paths import ROI_MANIFEST, RUNS_DIR  # noqa: E402

OUT = ROOT / "figures"
COL = {"conv": "#4C72B0", "vit": "#7A68A6", "seg": "#3A923A", "bio": "#D9822B", "head": "#5F5F5F",
       "loss": "#C44E52", "cls": "#4B3B78", "patch": "#B9AEDC", "ink": "#222222"}
TOKEN_COLS = ["#D9822B", "#E6A055", "#EFBD83", "#F6D9B5"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------- data from the model
def run_model():
    rows = pd.read_csv(ROI_MANIFEST)
    r = rows[(rows.split == "train") & (rows.dataset == "ORIGA") & (rows.label == 1)].iloc[3]
    sample = RoiDataset([r.to_dict()], train=False)[0]
    model = BTViT(pretrained=False).eval()
    model.load_state_dict(torch.load(RUNS_DIR / "p2" / "BTViT_full" / "seed2026" / "model.pth",
                                     map_location="cpu", weights_only=True))
    captured = {}

    def grab(_module, args):
        captured["t"], captured["bias"] = args[0], args[1]
    handle = model.post[-1].register_forward_pre_hook(grab)
    with torch.no_grad():
        out = model(sample["image"][None])
        # CLS attention of the last block, averaged over heads (bias included, as in the forward pass)
        blk = model.post[-1].blk
        h = blk.norm1(captured["t"])
        b, n, c = h.shape
        qkv = blk.attn.qkv(h).reshape(b, n, 3, blk.attn.num_heads, c // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        logits = (q @ k.transpose(-2, -1)) * blk.attn.scale + captured["bias"] * model.alpha[-1]
        attn = logits.softmax(-1)[0, :, 0, 1:197].mean(0).reshape(14, 14).numpy()
    handle.remove()
    img = np.clip(sample["image"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    pred = out["seg_logits"].argmax(1)[0].numpy()
    p = out["seg_logits"].softmax(1)[0]
    rim = F.adaptive_avg_pool2d((p[1:].sum(0) - p[2]).clamp(min=0)[None, None], 14)[0, 0].numpy()
    bio = dict(zip(FEATURES, out["biomarkers"][0].tolist()))
    prob = out["logits"].softmax(1)[0, 1].item()
    return img, pred, rim / rim.max(), attn / attn.max(), bio, prob


def mask_overlay(img, pred):
    ov = img.copy()
    for v, col, a in ((1, (0.23, 0.57, 0.23), 0.40), (2, (0.85, 0.51, 0.17), 0.55)):
        sel = pred >= v if v == 1 else pred == 2
        ov[sel] = (1 - a) * ov[sel] + a * np.array(col)
    for v, col in ((1, (0.13, 0.42, 0.13)), (2, (0.70, 0.38, 0.05))):
        cs, _ = cv2.findContours((pred >= v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, col, 2)
    return ov


def heat_overlay(img, heat, cmap="magma", alpha=0.55):
    hm = cv2.resize(heat.astype(np.float32), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    hm = plt.get_cmap(cmap)(np.clip(hm, 0, 1))[..., :3]
    return np.clip((1 - alpha) * img * 0.9 + alpha * hm, 0, 1)


# --------------------------------------------------------------------------- drawing primitives
def slab(ax, cx, cy, h, depth, color, w=2.2):
    """Pseudo-3D feature-map slab centred at (cx, cy): front face h tall, `depth` = channel thickness."""
    dx, dy = depth * 0.55, depth * 0.35
    x0, y0 = cx - (w + dx) / 2, cy - (h + dy) / 2
    front = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    top = [(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + dx, y0 + h + dy), (x0 + dx, y0 + h + dy)]
    side = [(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)]
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    for poly, shade in ((side, 0.55), (top, 0.80), (front, 1.0)):
        fc = 1 - (1 - rgb) * 0.55 * shade
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec=color, lw=0.6, zorder=3))


def token_col(ax, x, y_top, kinds, size=2.6, gap=0.5):
    """Vertical token sequence; kinds: 'cls', 'patch', 'dots', or an int index into TOKEN_COLS."""
    y = y_top
    for k in kinds:
        if k == "dots":
            for dy in (0.5, 1.3, 2.1):  # vertical ellipsis drawn as dots (Times has no U+22EE)
                ax.add_patch(plt.Circle((x + size / 2, y - dy), 0.22, color=COL["vit"], zorder=4))
        else:
            fc = COL["cls"] if k == "cls" else COL["patch"] if k == "patch" else TOKEN_COLS[k]
            ax.add_patch(Rectangle((x, y - size), size, size, fc=fc, ec="white", lw=0.5, zorder=4))
        y -= size + gap
    return y


def encoder(ax, x, y, w, h, title, sub, color=COL["vit"]):
    for k in (1.2, 0.6):
        ax.add_patch(FancyBboxPatch((x + k, y + k), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                    fc="white", ec=color, lw=0.5, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                fc="#F3F0FA", ec=color, lw=0.9, zorder=2))
    inner_w = w - 4
    for i, (lab, fc) in enumerate((("Multi-head attention", "#E3DDF3"), ("MLP (GELU)", "#E3DDF3"))):
        yy = y + h - 7.3 - i * 6.6
        ax.add_patch(FancyBboxPatch((x + 2, yy), inner_w, 4.6, boxstyle="round,pad=0.15,rounding_size=0.6",
                                    fc=fc, ec=color, lw=0.5, zorder=3))
        ax.text(x + w / 2, yy + 2.3, lab, ha="center", va="center", fontsize=5.4, zorder=4)
    ax.text(x + w / 2, y + 2.6, sub, ha="center", va="center", fontsize=5.6, color=color, style="italic", zorder=4)
    ax.text(x + w / 2 + 0.6, y + h + 2.4, title, ha="center", va="bottom", fontsize=7, weight="bold", color=color)


def arrow(ax, p, q, color=COL["ink"], lw=0.8, ls="-", rad=0.0, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=2))


def caption(ax, x, y, text, size=6.2, color=COL["ink"], **kw):
    ax.text(x, y, text, ha="center", va="top", fontsize=size, color=color, **kw)


def panel_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=9, weight="bold", va="top", ha="left")


# --------------------------------------------------------------------------- figure
def main():
    img, pred, rim, attn, bio, prob = run_model()
    fig = plt.figure(figsize=(7.2, 4.43))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 200); ax.set_ylim(0, 123); ax.set_aspect("equal"); ax.axis("off")

    # ===================== (a) pipeline =====================
    panel_label(ax, 1, 122, "(a)")
    Y = 103  # main-path centre line
    ax.imshow(img, extent=(3, 25, Y - 11, Y + 11), zorder=2)
    caption(ax, 14, Y - 12.3, "disc-centred ROI\n224×224, RGB")

    # conv stem: shrinking maps, growing channels
    for cx, h, d in ((32, 20, 2), (37.5, 15, 3.5), (43, 10.5, 5), (48.5, 7, 6.5)):
        slab(ax, cx, Y, h, d, COL["conv"])
    caption(ax, 40, Y + 17.0, "conv stem\n112→56→28→14", color=COL["conv"])
    arrow(ax, (25.6, Y), (28.8, Y))

    arrow(ax, (53, Y), (56.5, Y))
    token_col(ax, 57, Y + 11.4, ["cls", "patch", "patch", "patch", "dots", "patch", "patch"])
    caption(ax, 58.3, Y - 12.3, "[CLS] + 196\npatch tokens", color=COL["vit"])
    arrow(ax, (60.4, Y), (64, Y))
    encoder(ax, 64.5, Y - 10, 23, 20, "Encoder blocks 1–4", "DeiT-Tiny init")

    # long path to the fusion column
    arrow(ax, (89.5, Y), (131.5, Y))
    ax.text(110, Y + 1.2, "[CLS | 196 patch tokens]", ha="center", va="bottom", fontsize=5.8, color=COL["vit"])
    token_col(ax, 132, Y + 14.2, ["cls", "patch", "patch", "dots", "patch", 0, 1, 2, 3])
    ax.text(135.0, Y - 7.4, "}", fontsize=17, va="center", color=COL["bio"])
    ax.text(138.8, Y - 7.4, "4 clinical\ntokens", fontsize=5.6, va="center", color=COL["bio"])
    arrow(ax, (144.6, Y), (147.5, Y))
    encoder(ax, 148, Y - 10, 23, 20, "Encoder blocks 5–8", "rim-guided attention")

    # head and output
    arrow(ax, (172.5, Y), (175, Y))
    ax.add_patch(FancyBboxPatch((175.5, Y - 6), 6, 12, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc="#EDEDED", ec=COL["head"], lw=0.8, zorder=2))
    ax.text(178.5, Y, "MLP\nhead", ha="center", va="center", fontsize=5.6, zorder=3)
    arrow(ax, (182.2, Y), (184.5, Y))
    for i, (lab, v, col) in enumerate((("Glaucoma", prob, COL["loss"]), ("Normal", 1 - prob, "#8C8C8C"))):
        yy = Y + 3.5 - i * 7
        ax.add_patch(Rectangle((185, yy - 1.6), 13 * v, 3.2, fc=col, ec="none", alpha=0.85, zorder=3))
        ax.add_patch(Rectangle((185, yy - 1.6), 13, 3.2, fc="none", ec="#999999", lw=0.4, zorder=3))
        ax.text(185, yy + 2.0, f"{lab}  {v:.2f}", fontsize=5.6, va="bottom")
    caption(ax, 191.5, Y - 9.2, "calibrated\nprobability", size=5.6)

    # ---- segmentation / biomarker row
    Y2 = 72
    arrow(ax, (76, Y - 10.6), (76, Y2 + 9.5), color=COL["seg"])
    for cx, h, d in ((67.5, 7, 6.5), (73, 10.5, 5), (78.5, 15, 3.5), (84, 20, 1.5)):
        slab(ax, cx, Y2, h, d, COL["seg"])
    caption(ax, 76, Y2 - 12.0, "segmentation decoder\n14→28→56→112→224", color=COL["seg"])
    for (sx, sh), (dx_, dh) in (((37.5, 15), (78.5, 15)), ((43, 10.5), (73, 10.5)), ((48.5, 7), (67.5, 7))):
        arrow(ax, (sx, Y - sh / 2 - 1.8), (dx_ - 1.5, Y2 - dh / 2 - 1.2), color=COL["conv"], lw=0.6,
              ls=(0, (2.5, 1.8)), rad=0.32, head=4)
    ax.text(29, 74.5, "multi-scale\nskip connections", fontsize=5.4, color=COL["conv"], ha="center", style="italic")

    arrow(ax, (88.5, Y2), (92, Y2), color=COL["seg"])
    ax.imshow(mask_overlay(img, pred), extent=(92.5, 112.5, Y2 - 10, Y2 + 10), zorder=2)
    caption(ax, 102.5, Y2 - 11.2, "predicted disc / cup", color=COL["seg"])

    # biomarker mini chart (real values)
    arrow(ax, (113, Y2), (116.5, Y2), color=COL["bio"])
    show = [("vCDR", bio["vcdr"], 1.0), ("CDR", bio["cdr_area"], 1.0), ("hCDR", bio["hcdr"], 1.0),
            ("Rim I", bio["rim_inferior"], 0.5), ("Rim S", bio["rim_superior"], 0.5), ("NRR", bio["nrr"], 2.0)]
    ax.add_patch(FancyBboxPatch((117, Y2 - 10), 17, 23, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc="#FDF3E7", ec=COL["bio"], lw=0.8, zorder=1))
    ax.text(125.5, Y2 + 11.8, "12 biomarkers", ha="center", va="center", fontsize=5.8, color="#8A4F12", weight="bold")
    for i, (lab, v, vmax) in enumerate(show):
        yy = Y2 + 7.6 - i * 3.1
        ax.text(122.3, yy, lab, fontsize=5.0, ha="right", va="center")
        ax.add_patch(Rectangle((123, yy - 0.95), 7.0, 1.9, fc="white", ec="#D7B48A", lw=0.3, zorder=2))
        ax.add_patch(Rectangle((123, yy - 0.95), 7.0 * min(v / vmax, 1), 1.9, fc=COL["bio"], ec="none", zorder=3))
        ax.text(130.6, yy, f"{v:.2f}", fontsize=4.6, va="center")
    arrow(ax, (126.5, Y2 + 13.6), (132.6, Y + 14.2 - 9 * 3.1 - 0.5), color=COL["bio"], rad=-0.25)
    ax.text(136.0, 85.3, "token MLP\n+ type emb.", fontsize=5.2, color=COL["bio"], style="italic")

    # rim prior -> encoder 5-8
    ax.imshow(rim, extent=(150, 160, Y2 - 5, Y2 + 5), cmap="Greens", vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((150, Y2 - 5), 10, 10, fill=False, ec=COL["seg"], lw=0.5, zorder=3))
    caption(ax, 155, Y2 - 6.2, "rim prior\n14×14", color=COL["seg"], size=5.6)
    arrow(ax, (112.8, Y2 - 9.6), (149.6, Y2 - 4), color=COL["seg"], lw=0.6, rad=0.28)
    arrow(ax, (157, Y2 + 5.4), (161, Y - 10.6), color=COL["seg"], lw=0.7, ls=(0, (2.5, 1.8)))
    ax.text(163.5, 85.5, r"$+\alpha\,\log(\mathrm{rim})$" "\non CLS→patch", fontsize=5.6, color=COL["seg"])

    ax.plot([2, 198], [53.5, 53.5], color="#CCCCCC", lw=0.5)

    # ===================== (b) rim-guided attention =====================
    panel_label(ax, 1, 51, "(b)")
    ax.text(10, 50.4, "Rim-guided attention (blocks 5–8)", fontsize=7, weight="bold", va="top")
    tiles = [(img, "input ROI"), (mask_overlay(img, pred), "predicted disc / cup"),
             (heat_overlay(img, rim, "Greens", 0.5), "rim prior"), (heat_overlay(img, attn), "CLS attention, block 8")]
    for i, (im, lab) in enumerate(tiles):
        x0 = 3 + i * 23.5
        ax.imshow(im, extent=(x0, x0 + 21, 17, 38), zorder=2)
        caption(ax, x0 + 10.5, 15.8, lab, size=5.8)
        if i < 3:
            arrow(ax, (x0 + 21.4, 27.5), (x0 + 23.1, 27.5), lw=0.6, head=4)
    ax.text(48, 7.5, r"$\mathbf{A}_{\mathrm{CLS},j}=\mathrm{softmax}_j\!\left(\frac{q_{\mathrm{CLS}}\,k_j^{\top}}{\sqrt{d}}"
            r"+\alpha_l\,\log\!\left(r_j+\epsilon\right)\right)$", ha="center", fontsize=7.2)
    ax.text(48, 1.6, r"$r_j$: rim probability pooled to the $14\times14$ patch grid;  $\alpha_l$ learned per block",
            ha="center", fontsize=5.6, color="#555555")

    ax.plot([97, 97], [2, 51], color="#CCCCCC", lw=0.5)

    # ===================== (c) clinical tokens + objective =====================
    panel_label(ax, 99, 51, "(c)")
    ax.text(108, 50.4, "Biomarkers as tokens and training objective", fontsize=7, weight="bold", va="top")
    groups = [("cup–disc", "CDR, vCDR, hCDR"), ("rim / ISNT", "I, S, N+T, NRR, violation"),
              ("disc size", "area, ellipticity"), ("boundary", "disc & cup edge strength")]
    for i, (g, members) in enumerate(groups):
        yy = 41.5 - i * 7.2
        ax.add_patch(FancyBboxPatch((101, yy - 2.6), 33, 5.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#FDF3E7", ec=COL["bio"], lw=0.6, zorder=2))
        ax.text(102.5, yy + 0.7, g, fontsize=6, weight="bold", va="center", color="#8A4F12")
        ax.text(102.5, yy - 1.3, members, fontsize=5.2, va="center")
        arrow(ax, (134.6, yy), (139.5, yy), color=COL["bio"], lw=0.6, head=4)
        ax.add_patch(FancyBboxPatch((140, yy - 2), 7, 4, boxstyle="round,pad=0.15,rounding_size=0.5",
                                    fc="white", ec=COL["bio"], lw=0.5, zorder=2))
        ax.text(143.5, yy, "MLP", fontsize=5.2, ha="center", va="center")
        arrow(ax, (147.4, yy), (150.5, yy), color=COL["bio"], lw=0.6, head=4)
        ax.add_patch(Rectangle((151, yy - 1.5), 3, 3, fc=TOKEN_COLS[i], ec="white", lw=0.5, zorder=3))
        ax.text(155.2, yy, rf"$c_{i + 1}$", fontsize=6.5, va="center")
    ax.text(180, 44.5, "Training objective", fontsize=6.4, weight="bold", ha="center")
    ax.text(180, 37.5, r"$\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{seg}}\mathcal{L}_{\mathrm{seg}}"
            r"+\lambda_{\mathrm{bio}}\mathcal{L}_{\mathrm{bio}}$", fontsize=7.2, ha="center")
    for i, (t, d) in enumerate(((r"$\mathcal{L}_{\mathrm{cls}}$", "cross-entropy, labelled"),
                                (r"$\mathcal{L}_{\mathrm{seg}}$", "Dice + CE, all 2,250 masks"),
                                (r"$\mathcal{L}_{\mathrm{bio}}$", "smooth-L1 vs expert masks"))):
        yy = 29.5 - i * 5.2
        ax.text(162, yy, t, fontsize=7, color=COL["loss"], va="center")
        ax.text(168, yy, d, fontsize=5.6, va="center")
    ax.text(180, 11.5, r"$\lambda_{\mathrm{seg}}=1,\ \lambda_{\mathrm{bio}}=0.5$", fontsize=6.4, ha="center")
    ax.text(180, 5.5, "4.91 M parameters · trains on a 4 GB GPU", fontsize=5.6, ha="center", color="#555555")

    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"btvit_architecture.{ext}")
    fig.savefig(OUT / "btvit_architecture.png", dpi=600)
    print(f"glaucoma prob {prob:.3f}; vCDR {bio['vcdr']:.3f}; wrote figures/btvit_architecture.(pdf|svg|png)")


if __name__ == "__main__":
    main()
