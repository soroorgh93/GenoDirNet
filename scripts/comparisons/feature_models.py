"""Feature-based models trained under the protocol of the article.

    python scripts/comparisons/feature_models.py --data data/extraction1 --out results/genodirnet \
        --sets data/extraction1/sets/test_w*_seqs.txt [--no-fat | --windows absolute] [--models orf_lr wordfreq_lr]

Models: length_lr (logistic regression on the logarithm of the length), wordfreq_lr and wordfreq_mlp (frequencies of
the words of length 1 to 6 and the log length; logistic regression, and a perceptron with 256 hidden units),
framemlp_k4 and framemlp_k5 (perceptron on the log frame-resolved counts of 4- or 5-letter words and the log
length) and orf_lr (logistic regression on the five ORF features, a CPAT-style model fitted on the training
chromosomes). Fragment-aware training of a feature model: every training transcript is used once complete and
once as one random window (covering 50-100 % of its length, or of 100-1,000 nt with --windows absolute).

Writes <out>/<name>[_nofat|_absolute]_{val,test,<set>}_probs.npy and <out>/<name>[...].json.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet import orf
from genodirnet.data import read_set, set_key, split
from genodirnet.fragments import fat_window
from genodirnet.frcgr import cgr_counts, frame_counts
from genodirnet.metrics import metrics


def matrix(seqs, row):
    """float32 matrix with one row of features per sequence."""
    first = row(seqs[0]); X = np.empty((len(seqs), first.size), dtype=np.float32); X[0] = first
    for i in range(1, len(seqs)):
        X[i] = row(seqs[i])
    return X


def feat_length(seqs):
    return matrix(seqs, lambda s: np.array([np.log(len(s))]))


def feat_words(seqs, ks=(1, 2, 3, 4, 5, 6)):
    return matrix(seqs, lambda s: np.concatenate([cgr_counts(s, k).ravel() / max(1.0, len(s) - k + 1) for k in ks] + [[np.log(len(s))]]))


def feat_frames(seqs, k):
    return matrix(seqs, lambda s: np.concatenate([np.log1p(frame_counts(s, k).reshape(3, -1).astype(np.float32)).ravel(), [np.log(len(s))]]))


FEATURES = {'length_lr': feat_length, 'wordfreq_lr': feat_words, 'wordfreq_mlp': feat_words,
            'framemlp_k4': lambda s: feat_frames(s, 4), 'framemlp_k5': lambda s: feat_frames(s, 5)}


def make_model(name):
    if name.endswith('_lr'):
        return make_pipeline(StandardScaler(copy=False), LogisticRegression(C=0.5, max_iter=3000))
    return make_pipeline(StandardScaler(copy=False), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=0))


def weights_of(clf, n_features):
    last = clf.steps[-1][1]
    if isinstance(last, MLPClassifier):
        sizes = [n_features] + list(last.hidden_layer_sizes) + [1]
        return int(sum(a * b for a, b in zip(sizes[:-1], sizes[1:])))
    return int(n_features)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--sets', nargs='*', default=[])
    ap.add_argument('--models', nargs='+', default=['length_lr', 'wordfreq_lr', 'wordfreq_mlp', 'framemlp_k4', 'framemlp_k5', 'orf_lr'])
    ap.add_argument('--no-fat', action='store_true'); ap.add_argument('--windows', choices=['relative', 'absolute'], default='relative')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    tag = '_nofat' if a.no_fat else '_absolute' if a.windows == 'absolute' else ''
    table = orf.HexamerTable([s for s, y in zip(str_, ytr) if y == 1], [s for s, y in zip(str_, ytr) if y == 0])
    if not a.no_fat:
        rng = np.random.default_rng(25520)
        draw = (lambda s: fat_window(s, _Always(rng), 'relative')) if a.windows == 'relative' else (lambda s: fat_window(s, _Always(rng), 'absolute'))
        str_ = str_ + [draw(s) for s in str_]; ytr = np.concatenate([ytr, ytr])
    sets = {set_key(p): read_set(p)[1] for p in a.sets}
    for name in a.models:
        feats = (lambda s: orf.features(s, table)) if name == 'orf_lr' else FEATURES[name]
        clf = make_model(name).fit(feats(str_), ytr)
        run = name + tag; res = dict(model=name, run=run, fat=not a.no_fat, windows=None if a.no_fat else a.windows)
        n_feat = 5 if name == 'orf_lr' else feats(ste[:1]).shape[1]
        res['params'] = res['macs'] = weights_of(clf, n_feat)
        for split_name, seqs, y in (('val', sva, yva), ('test', ste, yte)):
            p = clf.predict_proba(feats(seqs))[:, 1]; res[split_name] = metrics(y, p)
            np.save(os.path.join(a.out, f'{run}_{split_name}_probs.npy'), p)
        for key, seqs in sets.items():
            np.save(os.path.join(a.out, f'{run}_{key}_probs.npy'), clf.predict_proba(feats(seqs))[:, 1])
        json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
        print(run, f"test AUROC {100 * res['test']['auroc']:.2f}", flush=True)


class _Always:
    """A random generator for which every transcript is windowed (fat_window windows with probability 0.8 otherwise):
    the feature models see each transcript once complete and once as a window."""

    def __init__(self, rng):
        self.rng = rng

    def random(self):
        return 0.0

    def uniform(self, lo, hi):
        return self.rng.uniform(lo, hi)

    def integers(self, lo, hi):
        return self.rng.integers(lo, hi)


if __name__ == '__main__':
    main()
