"""Periodic Markov chain of order 5 (Methods: Compared methods).

    python scripts/comparisons/markov_chain.py --data data/extraction1 --out results/genodirnet \
        --sets data/extraction1/sets/test_w*_seqs.txt

The coding chain has three transition tables, one per codon phase, fitted by hard expectation maximisation over the
unknown phase of each coding training transcript; the noncoding chain is homogeneous and fitted by maximum
likelihood (pseudocount 1). A logistic regression combines the per-nucleotide log-likelihood ratio of the best
alignment, the difference between the best and the mean over the three alignments, and the log length.
Writes <out>/markov5_{val,test,<set>}_probs.npy and <out>/markov5.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import read_set, set_key, split
from genodirnet.frcgr import encode
from genodirnet.metrics import metrics


def word_index(codes, k):
    """Index (base 4) of every word of length k; -1 where a word has an ambiguous base."""
    n = len(codes) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.zeros(n, dtype=np.int64); ok = np.ones(n, dtype=bool)
    for j in range(k):
        c = codes[j:j + n]; idx = idx * 4 + np.maximum(c, 0); ok &= c >= 0
    idx[~ok] = -1
    return idx


class PeriodicChain:
    """Three transition tables P_f(x_t | the m preceding nucleotides), or one shared table (homogeneous chain)."""

    def __init__(self, order, homogeneous=False):
        self.m, self.homogeneous, self.logp = order, homogeneous, None

    def fit(self, seqs, shifts=None):
        k = self.m + 1; counts = np.zeros((3, 4 ** k))
        shifts = np.zeros(len(seqs), dtype=int) if shifts is None else shifts
        for s, sh in zip(seqs, shifts):
            idx = word_index(encode(s), k)
            for f in range(3):
                w = idx[(f - sh) % 3::3]; w = w[w >= 0]
                counts[f] += np.bincount(w, minlength=4 ** k)
        if self.homogeneous:
            counts[:] = counts.sum(0, keepdims=True)
        c = counts.reshape(3, 4 ** self.m, 4) + 1.0
        self.logp = np.log(c / c.sum(2, keepdims=True)).reshape(3, 4 ** k).astype(np.float32)
        return self

    def fit_aligned(self, seqs, iterations=4):
        """Hard expectation maximisation over the phase of every coding transcript."""
        shifts = np.zeros(len(seqs), dtype=int)
        self.fit(seqs, shifts)
        for _ in range(iterations):
            shifts = np.array([int(np.argmax(self.loglik(encode(s)))) for s in seqs])
            self.fit(seqs, shifts)
        return self

    def loglik(self, codes):
        """Log-likelihood for the three alignments (phase of the first position 0, 1, 2)."""
        k = self.m + 1; idx = word_index(codes, k); out = np.zeros(3)
        if len(idx) == 0:
            return out
        ok = idx >= 0; pos = np.arange(len(idx))
        for shift in range(3):
            out[shift] = self.logp[((pos + shift) % 3)[ok], idx[ok]].sum()
        return out


def scores(chains, seqs):
    cod, non = chains; out = np.zeros((len(seqs), 3))
    for i, s in enumerate(seqs):
        c = encode(s); n = max(len(s), 1); lc = cod.loglik(c); ln = non.loglik(c)
        out[i] = ((lc.max() - ln.max()) / n, (lc.max() - lc.mean()) / n, np.log(n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--sets', nargs='*', default=[])
    ap.add_argument('--order', type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    m = a.order; name = f'markov{m}'
    chains = (PeriodicChain(m).fit_aligned([s for s, y in zip(str_, ytr) if y == 1]),
              PeriodicChain(m, homogeneous=True).fit([s for s, y in zip(str_, ytr) if y == 0]))
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(scores(chains, str_), ytr)
    res = dict(model=name, order=m, params=int(4 * 4 ** (m + 1)), macs=int(4 * 4 ** (m + 1)))
    for split_name, seqs, y in (('val', sva, yva), ('test', ste, yte)):
        p = lr.predict_proba(scores(chains, seqs))[:, 1]; res[split_name] = metrics(y, p)
        np.save(os.path.join(a.out, f'{name}_{split_name}_probs.npy'), p)
    for path in a.sets:
        np.save(os.path.join(a.out, f'{name}_{set_key(path)}_probs.npy'), lr.predict_proba(scores(chains, read_set(path)[1]))[:, 1])
    json.dump(res, open(os.path.join(a.out, f'{name}.json'), 'w'), indent=1)
    print(name, f"test AUROC {100 * res['test']['auroc']:.2f}")


if __name__ == '__main__':
    main()
