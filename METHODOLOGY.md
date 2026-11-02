# Methodology: Biomarker-Token Vision Transformer (BT-ViT) for Glaucoma Decision Support

Reference paper: Nasrin et al., *A Lightweight Vision Transformer and Retinal Biomarker Fusion
Framework for Early Glaucoma Detection*, J. Clin. Med. 2026, 15, 5651 (hereafter **"the paper"**).

**Goal:** a custom ViT classifier that outperforms the paper's model. It is trained on the same
datasets (ORIGA, REFUGE) and tested on two unseen datasets (RIM-ONE DL, DRISHTI-GS).

---

## 1. Hypotheses and what "outperform" means

The paper's numbers can't be used as a direct target. They are internal-test only and its code is
not released. Some of them are also internally inconsistent: optic-cup Dice is 0.9994, and the
confusion matrix (600/600) doesn't match its 370-image test sets. So we **re-implement the paper's
model (baseline B1) and train it under exactly our protocol**, and compare against that.

- **H1 (primary):** BT-ViT gets a higher **external AUC** than B1 on *both* RIM-ONE and DRISHTI-GS.
  Test: paired DeLong test, p < 0.05 after Holm correction.
- **H2:** BT-ViT's internal-test AUC is at least as high as B1's (non-inferiority margin 0.01).
- **H3:** BT-ViT is better calibrated (lower ECE) and is still "lightweight": fewer parameters and
  FLOPs than B1.
- We also report our internal-test numbers next to the paper's *published* numbers, with a caveat
  that these are not directly comparable.

## 2. Data

| Role | Dataset | Images | Glaucoma | Used for |
|---|---|---|---|---|
| Development | ORIGA (Kaggle `arnavjain1` mirror, full resolution, OD/OC masks) | 650 | 168 | classification + segmentation |
| Development | REFUGE2 train (Kaggle `victorlemosml/refuge2`, the paper's source; labels from `g`/`n` filenames) | 400 | 40 | classification + segmentation |
| Development | REFUGE (2018) validation set (`arnavjain1` mirror, labels in `index.json`) | 400 | 40 | classification + segmentation |
| Development (segmentation only) | REFUGE2 val (new 1940 px images) + REFUGE2 test (= REFUGE test); masks, no labels | 800 | – | segmentation loss only |
| **External test 1** | RIM-ONE DL | 485 | 172 | final testing only |
| **External test 2** | DRISHTI-GS | 101 | 70 | final testing only |

**Development splits** are stratified by label within each dataset and fixed with a recorded seed.

- **Internal test** (mirrors the paper's sizes):
  - ORIGA: 130 images.
  - REFUGE: 160 images (20% of the 800 labelled, stratified by source: REFUGE2 train vs
    REFUGE validation).
- **The rest is split 80/20 into train and validation.**
- ORIGA's official Set A/B split is reported as a secondary analysis.

**Leakage rules:**
- The external sets are never used for training, tuning, threshold choice or model selection.
- The classification threshold and calibration temperature are fixed on the validation set.
- Every preprocessing parameter is fitted on the training data only.

**Known limitations:**
- No patient IDs exist, so a split that keeps each patient's eyes together can't be guaranteed.
- The diagnostic labels were defined differently in each dataset.
- The REFUGE2 Kaggle mirror the paper cites has 1200 images with masks, but **no label files**.
  Only its 400 training images can be labelled, from their `g`/`n` filenames (40 glaucoma). Its
  train and test images are byte-identical to REFUGE (2018) train and test; its val images are new.
  So the paper's "960 train / 240 validation / 240 test" REFUGE2 classification split can't be
  labelled from its stated source. We say this in the paper's critique, and add the 400 labelled
  REFUGE (2018) validation images to ours.
- Masks use 0 = cup, 128 = rim, 255 = background, so disc = value < 255 and cup = value 0.
- ORIGA uses the full-resolution `arnavjain1` mirror; the paper cites a low-resolution mirror.

## 3. Preprocessing

1. **Disc localisation (automatic, no ground truth at test time).** A small U-Net runs on the full
   image downsampled to 256 px and predicts an optic-disc heatmap. It is trained on ORIGA and
   REFUGE full images. This replaces the paper's centre crop + brightness threshold, which fails
   when the disc is off-centre or the image has bright artefacts.
2. **Crop at a consistent scale.** Take a square crop centred on the predicted disc, with side 1.4×
   the disc diameter, so the disc fills ~40% of every crop. Our reproduction showed that
   inconsistent disc scale alone drops cup Dice from 0.71 to 0.50 across datasets.
   - **Pre-cropped dataset (RIM-ONE DL).** It is published as disc-centred crops, with the disc
     filling ~3/4 of the width (our window gives ~1/1.4 ≈ 0.71). So each image is used as provided,
     as the final window. This is a fixed rule for the whole dataset; it uses no per-image mask or
     label.
   - **Why the localiser isn't used on RIM-ONE.** RIM-ONE's close-up images look very different
     from full fundus photos, and the localiser often outlined the bright cup instead of the disc:
     only 63% of crops were usable. We found this by checking localisation on RIM-ONE against its
     *disc masks* (never its glaucoma labels).
   - **Sensitivity analysis.** Results using the localiser's crops on RIM-ONE are reported
     separately.
   - **Localiser accuracy (v2).** Validation disc Dice 0.954 on whole images and 0.952 on
     simulated 1.05× tight crops. On ORIGA, REFUGE and DRISHTI-GS, 98–100% of crops were usable
     (median centre error 1.4–2.2% of the disc diameter).
3. **Resize to 224×224 RGB.** Colour is kept, because pallor in the cup is a colour cue; the paper
   converts to grayscale (tested in ablation A7).
4. **Contrast enhancement.** CLAHE on the L channel of LAB colour space, then per-channel
   normalisation using training-set statistics. A median filter (3×3) is kept only if ablation
   shows it helps.
5. **Training augmentation:**
   - flips; rotation ±20°; disc-centre jitter ±10%; scale jitter 0.9–1.1
   - colour, brightness and gamma jitter; mild blur and JPEG compression, to simulate other cameras
   - a class-balanced sampler

## 4. Proposed model: BT-ViT

A lightweight hybrid ViT. Clinical biomarkers enter the transformer as **tokens** at the middle of
the network, instead of being concatenated after it as in the paper.

```
ROI 224x224 RGB
  └─ Conv stem: 3 stride-2 conv blocks (32→64→128) + 2x2 projection → 14x14 = 196 patch tokens, D=192
       └─ [CLS] + patch tokens + 2D positional embedding
            └─ Transformer blocks 1–4 (D=192, 3 heads, MLP 768)
                 ├─ Segmentation decoder (tokens + stem skip features) → background / disc / cup masks
                 │     └─ Differentiable biomarkers from the soft masks:
                 │          cup-disc group: CDR (area), vCDR, hCDR
                 │          rim group:      ISNT rim fractions (I,S,N,T), rim ratio (NRR), ISNT-violation score
                 │          size group:     disc area, disc ellipticity
                 │          edge group:     gradient strength along disc and cup boundaries (the paper's Eq. 11)
                 │     └─ each group → MLP → 1 clinical token (+ type embedding) ⇒ 4 clinical tokens
                 └─ Transformer blocks 5–8 over [CLS | 196 patches | 4 clinical tokens]
                      with a rim-guided attention bias: CLS→patch logits += α·log(rim mask, pooled to 14x14)
                           └─ Head: concat(CLS, mean of clinical tokens) → MLP → 2 logits (softmax)
```

**Design rationale, mapped to the gaps in the paper:**

| Gap in the paper | How BT-ViT addresses it |
|---|---|
| Late concatenation; the ViT never sees anatomy | Clinical tokens join the attention in blocks 5–8; rim-guided attention bias |
| "Transformers miss local features" | Convolutional stem gives a local bias, and 196 tokens instead of 784 |
| Separate CNN and ViT pipelines | One shared backbone does segmentation and classification (multi-task) |
| ISNT rule discussed but not used | ISNT rim fractions, rim ratio and an ISNT-violation token |
| "Lightweight" not quantified | ~4–5M parameters; params, FLOPs and latency reported on a GTX 1650 Ti and on CPU |
| Attention maps only, never validated | Attention measured against the masks; token attribution |
| No uncertainty | Temperature scaling + MC-dropout / ensemble; uncertain cases flagged for a clinician |

**Initialisation:** transformer blocks from ImageNet DeiT-Tiny weights (first 8 blocks, via timm).
The stem, clinical tokens and decoder start from random weights. Ablation A6 trains everything
from scratch.

**Loss:**

L = L_cls + λ_seg·L_seg + λ_bio·L_bio

- **L_cls:** class-weighted focal loss (γ = 2).
- **L_seg:** Dice + cross-entropy over the three classes, on all 2,250 development images with masks
  (650 ORIGA + 1,200 REFUGE2 + 400 REFUGE validation).
- **L_bio:** smooth-L1 between the predicted biomarkers and the same biomarkers computed from the
  expert masks. This trains the clinical tokens directly.
- Start with λ_seg = 1 and λ_bio = 0.5, then tune both on the validation set.

**Optimisation:**
- AdamW, weight decay 0.05.
- Learning rate 1e-4 for the pretrained blocks and 3e-4 for everything else.
- 5-epoch warm-up, then cosine decay; 50 epochs (the same budget as every baseline); fp32, because fp16 produces NaN on the GTX 1650 Ti. Batch 16, with gradient accumulation to an
  effective 32 if memory runs short.
- Early stopping on validation AUC.

## 5. Baselines (all re-trained under the same splits and evaluation)

| ID | Model | Notes |
|---|---|---|
| **B1** | **The paper's model, re-implemented** | Grayscale + median filter + CLAHE. A 4-level CNN encoder–decoder with deep supervision segments disc and cup. Biomarkers: vCDR, cup area, disc area, edge gradient statistics. ViT on 224 px images with 8×8 patches and 6 layers. Concatenation → fully connected layers → softmax. Adam, learning rate 1e-4, batch 16. Wherever the paper leaves a setting unspecified (embedding size 192, 3 heads), we write down what we chose. |
| B2 | Pannu et al. 2025 (ResUNet + CDR/NRR + SVM) | Our earlier reproduction |
| B3 | ResNet50, DenseNet121 (ImageNet), fine-tuned | CNN baselines from the paper's Table 6 |
| B4 | DeiT-Tiny and ViT-S/16, fine-tuned | A plain ViT, to separate the effect of the architecture |
| B5 | Biomarkers only (logistic regression on vCDR, CDR, rim ratio, ISNT) | A clinical-rule reference |

## 6. Evaluation

- **Classification:**
  - AUC is the primary metric: it doesn't depend on a threshold, and the class balance differs
    across datasets.
  - Also reported: accuracy, balanced accuracy, sensitivity, specificity, precision and F1, all at
    the threshold fixed on validation; sensitivity at 95% specificity; confusion matrices.
- **Calibration:** ECE, Brier score and reliability diagrams.
- **Referral:** accuracy and AUC after deferring the 10% and 20% most uncertain cases.
- **Segmentation:**
  - Foreground-only per-image Dice and IoU for disc and cup.
  - The paper-style pooled / Keras metric is also reported, purely to show how it inflates scores.
- **Statistics:**
  - 5 seeds, reported as mean ± SD.
  - 95% bootstrap confidence intervals (2,000 resamples).
  - DeLong test for paired AUC comparisons, with Holm correction across comparisons.
- **Explainability:**
  - attention rollout maps
  - the share of attention that falls in the disc and the rim
  - a deletion test
  - each clinical token's contribution: CLS→token attention and integrated gradients
- **Efficiency:** parameters, GFLOPs, and images per second on the GTX 1650 Ti and on CPU, for
  BT-ViT and B1.

## 7. Ablations (run on the validation set and the external sets)

| ID | Change |
|---|---|
| A1 | Clinical tokens → the paper's late concatenation |
| A2 | No segmentation branch (and so no clinical tokens) |
| A3 | No biomarker supervision (λ_bio = 0) |
| A4 | No rim-guided attention bias |
| A5 | Conv stem → linear patch embedding (16×16) |
| A6 | Trained from scratch instead of DeiT-Tiny init |
| A7 | Grayscale input (as in the paper) instead of RGB |
| A8 | Paper-style centre crop + threshold instead of the localiser + consistent-scale crop |
| A9 | Remove one biomarker group at a time (cup-disc, rim, size, edge) |

## 8. Decision-support output (per image)

- Glaucoma probability (calibrated), plus a risk band.
- Uncertainty, with a "refer to clinician" flag when it is high.
- Biomarkers with reference ranges:
  - vCDR, with severity band: ≤0.5 normal, 0.5–0.8 moderate, >0.8 severe
  - CDR, rim ratio
  - ISNT check and the most-thinned sector
  - disc size
- Overlays: disc and cup masks, rim, and the attention map.
- A small local app (Gradio): upload a fundus image and get the report.

## 9. Work plan

| Phase | Deliverable |
|---|---|
| P0 | Data pipeline: full-image extraction, disc localiser, consistent-scale crops, splits saved to disk |
| P1 | Baselines B1–B5 trained; first external results |
| P2 | BT-ViT implemented and trained; compared with B1 (H1–H3) |
| P3 | Ablations A1–A9, 5 seeds, statistics, explainability analysis |
| P4 | Decision-support report + Gradio app; figures and tables for the manuscript |

## 10. Risks

- **The external sets differ a lot from the training data** (RIM-ONE: Spanish hospitals, different
  cameras, pre-cropped). This is why we use consistent-scale cropping and augmentation that
  simulates other cameras.
- **Few glaucoma cases to train on:** 248 in total, and only 80 from REFUGE. We use class-balanced
  sampling and focal loss, and AUC as the primary metric.
- **Label definitions differ by dataset**, so some error on the external sets will come from labels.
  This goes in the discussion.
- **DRISHTI-GS is small (101 images)**, so its confidence intervals will be wide. We report them.
- **Hardware is a 4 GB GPU**, so the model stays Tiny-sized. It trains in fp32, because fp16 is numerically broken on this
  GPU. Larger
  variants are out of scope.
