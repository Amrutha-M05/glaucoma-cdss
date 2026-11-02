"""Classification metrics with a threshold fixed on the validation set."""
import numpy as np
from sklearn import metrics


def youden_threshold(labels, probs):
    fpr, tpr, thr = metrics.roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])


def accuracy_threshold(labels, probs):
    """Threshold maximising accuracy (on the set it is fitted to - validation only)."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    cands = np.unique(probs)
    accs = [((probs >= t) == labels).mean() for t in cands]
    return float(cands[int(np.argmax(accs))])


def sens_at_spec(labels, probs, spec=0.95):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def ece(labels, probs, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    return float(sum(abs(probs[idx == b].mean() - labels[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def classification_report(labels, probs, threshold):
    labels, probs = np.asarray(labels), np.asarray(probs, np.float64)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "auc": float(metrics.roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else float("nan"),
        "accuracy": float((tp + tn) / len(labels)), "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens), "specificity": float(spec), "precision": float(prec),
        "f1": float(2 * prec * sens / max(prec + sens, 1e-12)),
        "sens_at_95spec": sens_at_spec(labels, probs), "brier": float(np.mean((probs - labels) ** 2)),
        "ece": ece(labels, probs), "threshold": float(threshold), "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
