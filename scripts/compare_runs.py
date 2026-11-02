"""Print a side-by-side comparison of runs (metrics json path per run)."""
import json, sys
for spec in sys.argv[1:]:
    name, f = spec.split("=", 1)
    m = json.load(open(f))
    row = [f"{name:16s} ep={str(m.get('best_epoch','-')):>3s}"]
    for g in ["val", "test", "RIM-ONE", "DRISHTI-GS"]:
        a, b = m["metrics"][g], m["metrics_accthr"][g]
        row.append(f"{g}: AUC {a['auc']:.3f} acc {a['accuracy']:.3f}/{b['accuracy']:.3f}")
    print(" | ".join(row))
