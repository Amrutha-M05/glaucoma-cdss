"""Statistics for comparing classifiers on the same images: DeLong's paired AUC test and bootstrap CIs."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(labels, prob_a, prob_b):
    """Two-sided DeLong test for two correlated ROC AUCs. Returns (auc_a, auc_b, z, p)."""
    labels = np.asarray(labels).astype(bool)
    preds = np.vstack([prob_a, prob_b])
    pos, neg = preds[:, labels], preds[:, ~labels]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_auc_ci(labels, probs, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the AUC (stratified resampling)."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
