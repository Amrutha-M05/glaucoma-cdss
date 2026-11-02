"""P3 - Final 5-seed comparison: BT-ViT (V2 + lambda_bio = 1) vs pure ViT, identical protocol.

Reports per-seed AUCs, mean +/- SD, the 5-seed ensembles (averaged probabilities), bootstrap 95% CIs
and DeLong paired tests between the two ensembles on each test set. Writes runs/p3/final_comparison.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdss.paths import RUNS_DIR  # noqa: E402
from cdss.stats import bootstrap_auc_ci, delong_test  # noqa: E402
from cdss.trainer import summarize  # noqa: E402

SEEDS = [2026, 2027, 2028, 2029, 2030]
MODELS = {"PureViT": "PureViT_purevit_tiny", "BT-ViT": "BTViT_v2_bio1"}
GROUPS = ["test", "RIM-ONE", "DRISHTI-GS"]
KEYS = ["dataset", "image_id", "split", "label"]


def group_rows(df, g):
    return df[df.split == "test"] if g == "test" else df[(df.split == "external") & (df.dataset == g)]


def main():
    out_dir = RUNS_DIR / "p3"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, ens = {"seeds": SEEDS, "per_seed": {}, "mean_sd": {}, "ensemble": {}, "delong": {}}, {}
    for name, tag in MODELS.items():
        per = []
        for s in SEEDS:
            m = json.loads((RUNS_DIR / "p2" / tag / f"seed{s}" / "metrics.json").read_text())["metrics"]
            per.append({g: m[g]["auc"] for g in GROUPS} | {"test_acc": None})
        result["per_seed"][name] = per
        arr = np.array([[p[g] for g in GROUPS] for p in per])
        result["mean_sd"][name] = {g: (float(arr[:, i].mean()), float(arr[:, i].std(ddof=1))) for i, g in enumerate(GROUPS)}
        frames = [pd.read_csv(RUNS_DIR / "p2" / tag / f"seed{s}" / "predictions.csv").set_index(KEYS)["prob"] for s in SEEDS]
        e = pd.concat(frames, axis=1).mean(axis=1).rename("prob").reset_index()
        ens_dir = out_dir / f"ensemble_{name}"
        ens_dir.mkdir(exist_ok=True)
        e.to_csv(ens_dir / "predictions.csv", index=False)
        result["ensemble"][name] = summarize(e, f"{name} ensemble (5 seeds)", ens_dir)
        ens[name] = e.set_index(KEYS)["prob"]

    for g in GROUPS:
        both = pd.concat([ens["BT-ViT"].rename("bt"), ens["PureViT"].rename("pv")], axis=1).reset_index()
        rows = group_rows(both, g)
        a_bt, a_pv, z, p = delong_test(rows.label.values, rows.bt.values, rows.pv.values)
        result["delong"][g] = {"auc_btvit": a_bt, "auc_purevit": a_pv, "diff": a_bt - a_pv, "z": z, "p": p,
                               "ci_btvit": bootstrap_auc_ci(rows.label.values, rows.bt.values),
                               "ci_purevit": bootstrap_auc_ci(rows.label.values, rows.pv.values)}

    (out_dir / "final_comparison.json").write_text(json.dumps(result, indent=2, default=float))
    print("\nAUC mean +/- SD over 5 seeds")
    for name in MODELS:
        print(f"  {name:8s} " + "  ".join(f"{g}: {m:.3f}+/-{s:.3f}" for g, (m, s) in result["mean_sd"][name].items()))
    print("\n5-seed ensembles, DeLong paired test (BT-ViT vs PureViT)")
    for g, d in result["delong"].items():
        print(f"  {g:10s} BT-ViT {d['auc_btvit']:.3f} [{d['ci_btvit'][0]:.3f}, {d['ci_btvit'][1]:.3f}]  "
              f"PureViT {d['auc_purevit']:.3f} [{d['ci_purevit'][0]:.3f}, {d['ci_purevit'][1]:.3f}]  "
              f"diff {d['diff']:+.3f}  z={d['z']:.2f}  p={d['p']:.4f}")


if __name__ == "__main__":
    main()
