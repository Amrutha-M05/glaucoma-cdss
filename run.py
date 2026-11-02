"""BT-ViT glaucoma CDSS - single entry point.

    python run.py status                      what has been run, what is missing
    python run.py prepare                     P0: datasets -> splits -> disc localiser -> ROI crops
    python run.py train  [--model btvit|purevit] [--seeds 2026 2027 ...]   final configuration
    python run.py compare                     P3: 5-seed BT-ViT vs pure ViT (DeLong, bootstrap CIs)
    python run.py results                     print every results table and write RESULTS.md
    python run.py figures                     regenerate the architecture figure and LaTeX tables

Each command prints numbered stages with [done] / [FAIL] status and timing; training prints one
line per epoch with the best-so-far marker and ETA. Individual steps can also be run directly from
scripts/ (see README.md).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cdss import console  # noqa: E402
from cdss.config import FINAL_BTVIT_ARGS, FINAL_PUREVIT_ARGS, SEEDS  # noqa: E402
from cdss.paths import DATA_DIR, MANIFEST, ROI_MANIFEST, RUNS_DIR, SPLITS  # noqa: E402

GROUPS = [("test", "Internal test"), ("RIM-ONE", "RIM-ONE (ext.)"), ("DRISHTI-GS", "DRISHTI-GS (ext.)")]
VARIANTS = [("V0: Pure ViT", "p2/PureViT_purevit_tiny/seed2026/metrics.json", "5.5 M"),
            ("V1: BT-ViT v1", "p2/BTViT_full/seed2026/metrics_tta.json", "4.9 M"),
            ("V2: BT-ViT v2", "p2/BTViT_v2/seed2026/metrics.json", "6.7 M")]
RECIPES = [("V2 (reference)", "v2"), ("+ EMA 0.999", "v2_ema"), ("+ label smoothing 0.1", "v2_ls01"),
           ("+ lambda_bio = 1.0", "v2_bio1"), ("+ open gates", "v2_opengates")]


def load(rel):
    p = RUNS_DIR / rel
    return json.loads(p.read_text()) if p.exists() else None


def run_script(args):
    """Run a script with live output; raise if it fails so the Stage prints [FAIL]."""
    cmd = [sys.executable, "-u", *args]
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise RuntimeError(f"{' '.join(args)} exited with an error (see output above)")


# --------------------------------------------------------------------------- commands
def cmd_status(_):
    console.banner("BT-ViT glaucoma CDSS - status")
    console.section("P0 data pipeline")
    checks = [("unified datasets", MANIFEST), ("train/val/test splits", SPLITS),
              ("disc localiser", RUNS_DIR / "localizer" / "disc_unet.pth"), ("disc-centred ROI crops", ROI_MANIFEST)]
    console.table(["step", "status", "file"],
                  [[n, console.OK if p.exists() else console.TODO, str(p.relative_to(p.parents[2]))] for n, p in checks],
                  left=3)

    console.section("Architecture variants and V2 recipe search (seed 2026)")
    rows = [[n, console.OK if load(f) else console.TODO] for n, f, _ in VARIANTS]
    rows += [[f"V2 {n}" if n.startswith("+") else n, console.OK if load(f"p2/BTViT_{t}/seed2026/metrics.json") else console.TODO]
             for n, t in RECIPES[1:]]
    console.table(["experiment", "status"], rows, left=2)

    console.section("Final comparison (5 seeds each)")
    seed_rows = []
    for name, tag in (("BT-ViT (V2 + lambda_bio=1)", "BTViT_v2_bio1"), ("Pure ViT", "PureViT_purevit_tiny")):
        marks = ["x" if (RUNS_DIR / "p2" / tag / f"seed{s}" / "metrics.json").exists() else "." for s in SEEDS]
        seed_rows.append([name, " ".join(marks), f"{marks.count('x')}/{len(SEEDS)}"])
    console.table(["model", "seeds " + " ".join(str(s)[-2:] for s in SEEDS), "done"], seed_rows)
    final = RUNS_DIR / "p3" / "final_comparison.json"
    print(f"\n  final comparison: {console.OK if final.exists() else console.TODO}  {final.relative_to(ROOT)}")
    print("  legend: x = trained, . = missing", flush=True)


def cmd_prepare(_):
    steps = [("Unify datasets (images <= 1024 px, label maps 0/1/2)", "scripts/p0_extract.py"),
             ("Fixed train / validation / test splits", "scripts/p0_splits.py"),
             ("Train the optic-disc localiser", "scripts/p0_train_localizer.py"),
             ("Localise discs and write disc-centred ROI crops", "scripts/p0_make_rois.py")]
    console.banner("P0 - data pipeline", f"output -> {DATA_DIR}")
    for i, (label, script) in enumerate(steps, 1):
        with console.Stage(i, len(steps), label):
            run_script([script])


def cmd_train(a):
    base = FINAL_BTVIT_ARGS if a.model == "btvit" else FINAL_PUREVIT_ARGS
    console.banner(f"Training {'BT-ViT (V2 + lambda_bio = 1)' if a.model == 'btvit' else 'pure ViT (DeiT-Tiny)'}",
                   f"seeds: {', '.join(map(str, a.seeds))}")
    for i, s in enumerate(a.seeds, 1):
        with console.Stage(i, len(a.seeds), f"seed {s}"):
            run_script(["scripts/p2_train_btvit.py", *base, "--seed", str(s)])


def cmd_compare(_):
    console.banner("P3 - final comparison: BT-ViT vs pure ViT (5 seeds each)")
    with console.Stage(1, 1, "per-seed AUCs, ensembles, bootstrap CIs, DeLong tests"):
        run_script(["scripts/p3_final_compare.py"])


def cmd_figures(_):
    console.banner("Figures and LaTeX tables")
    with console.Stage(1, 2, "architecture figure (figures/btvit_architecture.*)"):
        run_script(["figures/make_architecture_figure.py"])
    with console.Stage(2, 2, "LaTeX tables (figures/architecture_variants.tex)"):
        run_script(["figures/make_latex_tables.py"])


def cmd_results(_):
    md = ["# Results\n", "_Generated by `python run.py results`. All thresholds fitted on validation only;"
          " RIM-ONE and DRISHTI-GS were never seen during training._\n"]

    def emit(title, headers, rows, note=None):
        console.section(title)
        console.table(headers, rows)
        if note:
            print(f"  {note}", flush=True)
        md.extend([f"\n## {title}\n", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
        md.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        if note:
            md.append(f"\n{note}")

    console.banner("BT-ViT glaucoma CDSS - results")
    rows = []
    for name, f, params in VARIANTS:
        m = load(f)
        rows.append([name, params] + ([f"{m['metrics'][g]['auc']:.3f}" for g, _ in GROUPS]
                                      + [f"{m['metrics_accthr']['test']['accuracy']:.3f}"] if m else ["-"] * 4))
    emit("1. Architecture variants (seed 2026, flip TTA, AUC)",
         ["variant", "params"] + [lab for _, lab in GROUPS] + ["test acc"], rows)

    rows = []
    for name, tag in RECIPES:
        m = load(f"p2/BTViT_{tag}/seed2026/metrics.json")
        rows.append([name] + ([f"{m['metrics']['val']['auc']:.3f}"] + [f"{m['metrics'][g]['auc']:.3f}" for g, _ in GROUPS]
                              if m else ["-"] * 4))
    best = max((r for r in rows if r[1] != "-"), key=lambda r: float(r[1]), default=None)
    emit("2. V2 training-recipe search (seed 2026, AUC; selected on validation)",
         ["variant", "val (selection)"] + [lab for _, lab in GROUPS], rows,
         f"selected: {best[0]} (highest validation AUC)" if best else None)

    final = load("p3/final_comparison.json")
    if final:
        rows = [[n] + [f"{final['mean_sd'][n][g][0]:.3f} +/- {final['mean_sd'][n][g][1]:.3f}" for g, _ in GROUPS]
                for n in ("PureViT", "BT-ViT")]
        emit("3. Final comparison: AUC mean +/- SD over 5 seeds", ["model"] + [lab for _, lab in GROUPS], rows)
        rows = []
        for n in ("PureViT", "BT-ViT"):
            e = final["ensemble"][n]
            rows.append([f"{n} ensemble"] + [f"{e['metrics'][g]['auc']:.3f}" for g, _ in GROUPS]
                        + [f"{e['metrics_accthr'][g]['sensitivity']:.2f}" for g in ("RIM-ONE", "DRISHTI-GS")])
        rows.append(["DeLong p (BT-ViT vs pure)"] + [f"{final['delong'][g]['p']:.3f}" for g, _ in GROUPS] + ["", ""])
        emit("4. Final comparison: 5-seed ensembles",
             ["model"] + [f"{lab} AUC" for _, lab in GROUPS] + ["RIM-ONE sens", "DRISHTI sens"], rows,
             "Sensitivity at the max-accuracy threshold fitted on validation.")
    else:
        print(f"\n  {console.TODO} final comparison not run yet (python run.py compare)")

    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written: {out.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in (("status", cmd_status), ("prepare", cmd_prepare), ("compare", cmd_compare),
                     ("results", cmd_results), ("figures", cmd_figures)):
        sub.add_parser(name).set_defaults(fn=fn)
    t = sub.add_parser("train", help="train the final BT-ViT or the pure-ViT baseline")
    t.add_argument("--model", choices=["btvit", "purevit"], default="btvit")
    t.add_argument("--seeds", type=int, nargs="+", default=[SEEDS[0]])
    t.set_defaults(fn=cmd_train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
