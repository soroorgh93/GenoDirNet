"""Loading of the coding probabilities written by the scripts, the logistic fusion of GenoDirNet and quantities
shared by the tables and the figures."""
import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from genodirnet.data import read_seqs
from genodirnet.frcgr import frame_counts, partition_counts

SEED_TAGS = ('', '_seed1', '_seed2')
GDN = ('seq_orf_net', 'frcgr_k5')                              # GenoDirNet, PyTorch runs of the standard schedule
GDN_P = ('seq_orf_net_np', 'frcgr_k5_np')                      # GenoDirNet, NumPy runs, relative fragment-aware windows
GDN_P0 = ('seq_orf_net_np_nofat', 'frcgr_k5_np_nofat')         # the same, complete transcripts only
GDN_A = ('seq_orf_net_np_absolute', 'frcgr_k5_np_absolute')    # the same, absolute fragment-aware windows


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Runs:
    """Coding probabilities in one results folder. A run <name> stores <name>_<split>_probs.npy for the splits
    val, test (complete test transcripts), test_w<L> (windows), test_w<L>i<n> (windows with insertions and
    deletions) and refseq_* (RefSeq sets); further training runs of a network are <name>_seed1 and <name>_seed2."""

    def __init__(self, folder, data):
        self.folder, self.data = folder, data
        self.y = read_seqs(os.path.join(data, 'test_seqs.txt'))[0]
        self.yv = read_seqs(os.path.join(data, 'val_seqs.txt'))[0]
        self._labels = {}

    def labels(self, split):
        if split == 'val':
            return self.yv
        if split.startswith('test'):
            return self.y
        if split not in self._labels:
            f = os.path.join(self.data, 'sets', f'{split}_seqs.txt')
            if not os.path.exists(f):
                f = os.path.join(self.data, 'sets', re.sub(r'_w\d+$', '', split) + '_seqs.txt')
            self._labels[split] = read_seqs(f)[0]
        return self._labels[split]

    def path(self, run, split):
        return os.path.join(self.folder, f'{run}_{split}_probs.npy')

    def load(self, run, split='test'):
        f = self.path(run, split)
        return np.load(f) if os.path.exists(f) else None

    def tags(self, name, split='test'):
        return [t for t in SEED_TAGS if os.path.exists(self.path(name + t, split))]

    def runs(self, name, split='test'):
        """Probabilities of every training run of a network that has scored this split."""
        return [self.load(name + t, split) for t in self.tags(name, split)]

    def record(self, run):
        f = os.path.join(self.folder, f'{run}.json')
        return json.load(open(f)) if os.path.exists(f) else None

    def fused(self, names, split='test'):
        """GenoDirNet: a logistic regression on the logits of the networks, fitted on the validation chromosomes
        separately for every training run (matched seeds) and applied to one split."""
        tags = [t for t in SEED_TAGS if all(os.path.exists(self.path(n + t, 'val')) and os.path.exists(self.path(n + t, split)) for n in names)]
        out = []
        for t in tags:
            xv = np.column_stack([logit(self.load(n + t, 'val')) for n in names])
            xt = np.column_stack([logit(self.load(n + t, split)) for n in names])
            out.append(LogisticRegression(C=1.0, max_iter=2000).fit(xv, self.yv).predict_proba(xt)[:, 1])
        return out


def get(R, spec, split='test'):
    """Runs of a network (name) or of GenoDirNet (tuple of the two fused networks) on one split."""
    return R.fused(spec, split) if isinstance(spec, tuple) else R.runs(spec, split)


def auroc_mean(y, runs):
    return float(np.mean([roc_auc_score(y, p) for p in runs])) if runs else None


def _js3(c):
    c = c.reshape(3, -1).astype(np.float64) + 0.5
    p = c / c.sum(1, keepdims=True); m = p.mean(0)
    h = lambda q: -(q * np.log(q)).sum(-1)
    return float(h(m) - h(p).mean())


def frame_signal(seqs, k=3):
    """Jensen-Shannon divergence between the three FR-CGR channels of 3-letter words, minus the same divergence for
    channels formed by a random partition of the start positions (Methods)."""
    return np.array([_js3(frame_counts(s, k)) - _js3(partition_counts(s, k)) for s in seqs])


LENGTH_CLASSES = [(200, 500, '200-500'), (500, 1000, '500-1k'), (1000, 2000, '1k-2k'), (2000, 5000, '2k-5k'), (5000, 20001, '5k-20k')]
LNC_CLASSES = ('unnamed', 'intergenic', 'antisense', 'divergent', 'other named')


def lncrna_class(name):
    """Class of a lncRNA from its gene symbol: antisense (-AS), divergent (-DT), overlapping (-IT, -OT), intergenic (LINC,
    LOC), unnamed or other named. The few overlapping lncRNAs are not one of LNC_CLASSES and are left out of the
    class-wise error rates."""
    if re.search(r'-AS\d*$', name):
        return 'antisense'
    if re.search(r'-DT$', name):
        return 'divergent'
    if re.search(r'-IT\d*$', name) or re.search(r'-OT\d*$', name):
        return 'overlapping'
    if name.startswith('LINC') or re.match(r'^LOC\d+', name):
        return 'intergenic'
    if name == '' or name.startswith('ENSG'):
        return 'unnamed'
    return 'other named'
