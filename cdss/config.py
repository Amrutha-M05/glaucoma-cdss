"""Central experiment settings shared by every model, so all comparisons use the same protocol."""

SEEDS = [2026, 2027, 2028, 2029, 2030]  # five seeds for the final comparison

BACKBONES = {
    "tiny": "deit_tiny_patch16_224.fb_in1k",
    "small": "deit_small_patch16_224.fb_in1k",
}

# Training protocol (METHODOLOGY.md section 4); identical for BT-ViT and the pure-ViT baseline.
TRAIN = {
    "preprocessing": "rgb_clahe",  # RGB + CLAHE on the LAB L channel, ImageNet normalisation
    "epochs": 50,
    "batch": 16,
    "lr": 1e-4,                    # pretrained transformer blocks
    "lr_new": 3e-4,                # newly initialised layers
    "weight_decay": 0.05,          # AdamW
    "patience": 15,                # early stopping on validation AUC
}

# BT-ViT specifics: loss weights and the share of segmentation-only images per batch.
BTVIT = {
    "lambda_seg": 1.0,
    "lambda_bio": 0.5,             # the selected final model uses 1.0 (--lambda-bio 1.0)
    "unlabelled_fraction": 1 / 3,
}

# Final configuration chosen on validation AUC (see README, "V2 training-recipe search").
FINAL_BTVIT_ARGS = ["--arch", "v2", "--lambda-bio", "1.0", "--tta", "--tag", "v2_bio1"]
FINAL_PUREVIT_ARGS = ["--model", "purevit", "--tta", "--tag", "purevit_tiny"]
