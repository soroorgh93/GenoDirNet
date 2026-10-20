"""Evaluation metrics of the article (Methods: Evaluation metrics and statistics).

Coding transcripts are the positive class. AUROC, AUPRC (average precision), accuracy, F1 and the
Matthews correlation coefficient at a threshold of 0.5, sensitivity at a false positive rate of 5 %,
the expected calibration error in ten equal-width bins, bootstrap confidence intervals over the test
transcripts and the paired DeLong test.
"""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score)


def auroc(y, p):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, dtype=float)))


def auprc(y, p):
    return float(average_precision_score(np.asarray(y).astype(int), np.asarray(p, dtype=float)))


def metrics(y, p):
    """AUROC, AUPRC, accuracy, MCC and F1 (threshold 0.5) of coding probabilities p for labels y."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                accuracy=float(accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                f1=float(f1_score(y, pred)), n=int(len(y)))


def sensitivity_at_fpr(y, p, fpr=0.05):
    """Share of coding inputs scored above the threshold that calls a share `fpr` of the noncoding inputs coding."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    t = np.quantile(p[y == 0], 1 - fpr)
    return float((p[y == 1] > t).mean())


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins of the predicted probability."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if sel.any():
            e += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(e)


def reliability(y, p, n_bins=10, min_count=20):
    """Mean predicted probability and observed coding fraction per bin (bins with fewer than min_count inputs left out)."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if sel.sum() >= min_count:
            out.append((float(p[sel].mean()), float(y[sel].mean())))
    return np.array(out)


_RESAMPLES = {}


def resample_counts(n, n_boot=1000, seed=0):
    """Counts of every test input in each of n_boot bootstrap resamples (the same resamples for every model)."""
    key = (n, n_boot, seed)
    if key not in _RESAMPLES:
        rng = np.random.default_rng(seed)
        _RESAMPLES[key] = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]).astype(np.float64)
    return _RESAMPLES[key]


def bootstrap_auroc(y, p, n_boot=1000, seed=0):
    """AUROC of p on every bootstrap resample of the inputs (ties count one half)."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    wc = resample_counts(len(y), n_boot, seed)
    _, g = np.unique(p, return_inverse=True); n_groups = g.max() + 1
    pos = np.stack([np.bincount(g, w * (y == 1), n_groups) for w in wc])
    neg = np.stack([np.bincount(g, w * (y == 0), n_groups) for w in wc])
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def auroc_ci(y, runs, n_boot=1000, seed=0):
    """Mean AUROC over training runs (list of probability arrays) with its 95 % bootstrap interval."""
    b = np.mean([bootstrap_auroc(y, p, n_boot, seed) for p in runs], axis=0)
    m = np.mean([roc_auc_score(y, p) for p in runs])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(m), float(lo), float(hi)


def auroc_difference(y, runs_a, runs_b, n_boot=1000, seed=0):
    """Difference of the mean AUROC of two models (each a list of runs) with its paired 95 % bootstrap interval."""
    ba = np.mean([bootstrap_auroc(y, p, n_boot, seed) for p in runs_a], axis=0)
    bb = np.mean([bootstrap_auroc(y, p, n_boot, seed) for p in runs_b], axis=0)
    d = np.mean([roc_auc_score(y, p) for p in runs_a]) - np.mean([roc_auc_score(y, p) for p in runs_b])
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def delong_test(y, score_a, score_b):
    """Paired DeLong test of two AUROCs on the same inputs: difference (a - b) and two-sided P value."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([rankdata(r) for r in pos]); ty = np.array([rankdata(r) for r in neg])
    tz = np.array([rankdata(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]; var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return float(d), float(2 * norm.sf(abs(d / np.sqrt(var))))


def delong_variance(y, p):
    """AUROC and the variance of the empirical AUROC (DeLong et al. 1988)."""
    y = np.asarray(y).astype(int); s = np.asarray(p, dtype=float)
    sp, sn = s[y == 1], s[y == 0]; pos, neg = np.sort(sp), np.sort(sn); m, n = len(sp), len(sn)
    v10 = (np.searchsorted(neg, sp, 'left') + 0.5 * (np.searchsorted(neg, sp, 'right') - np.searchsorted(neg, sp, 'left'))) / n
    v01 = (m - np.searchsorted(pos, sn, 'right') + 0.5 * (np.searchsorted(pos, sn, 'right') - np.searchsorted(pos, sn, 'left'))) / m
    var = (np.var(v10, ddof=1) / m if m > 1 else 0.0) + (np.var(v01, ddof=1) / n if n > 1 else 0.0)
    return float(v10.mean()), float(var)


def bfdr_calls(p, q0):
    """Bayesian false discovery rate rule (Newton et al. 2004; Mueller et al. 2007): inputs are called coding in
    decreasing order of their posterior probability p while the mean of 1 - p over the calls stays below q0, so that
    the posterior expected false discovery rate of the calls is below q0. Returns a boolean array of calls."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p); fdr = np.cumsum(1 - p[order]) / np.arange(1, len(p) + 1)
    n = int((fdr < q0).sum())
    calls = np.zeros(len(p), dtype=bool); calls[order[:n]] = True
    return calls
