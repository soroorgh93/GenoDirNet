"""The five ORF features of the sequence-ORF network and of the ORF-feature regression:
log(1 + length of the longest ORF), its coverage of the input, the Fickett score of the input, the hexamer score of
the longest ORF (0 without an ORF) and the logarithm of the input length.

The longest ORF runs from ATG to a stop codon in one of the three forward frames. The hexamer score is the
mean log ratio of the frequency of the in-frame hexamers of the ORF in the coding training transcripts (read in
frame along their longest ORF) and in the lncRNA training transcripts (read in steps of three from the first
nucleotide), with a pseudocount of one.
"""
from itertools import product

import numpy as np

STOPS = {'TAA', 'TAG', 'TGA'}


def longest_orf(seq):
    """Start and length (nucleotides) of the longest ATG-to-stop ORF on the forward strand ((0, 0) if none)."""
    s = seq.upper(); best = (0, 0); n = len(s)
    for frame in range(3):
        start = None
        for i in range(frame, n - 2, 3):
            codon = s[i:i + 3]
            if start is None and codon == 'ATG':
                start = i
            elif start is not None and codon in STOPS:
                if i + 3 - start > best[1]:
                    best = (start, i + 3 - start)
                start = None
    return best


# Fickett TESTCODE statistic (Fickett 1982): position and content probabilities and weights per nucleotide.
_POS_PROB = {'A': [0.94, 0.68, 0.84, 0.93, 0.58, 0.68, 0.45, 0.34, 0.20, 0.22], 'C': [0.80, 0.70, 0.70, 0.81, 0.66, 0.48, 0.51, 0.33, 0.30, 0.23],
             'G': [0.90, 0.88, 0.74, 0.64, 0.53, 0.48, 0.27, 0.16, 0.08, 0.08], 'T': [0.97, 0.97, 0.91, 0.68, 0.69, 0.44, 0.54, 0.20, 0.09, 0.09]}
_POS_W = {'A': 0.26, 'C': 0.18, 'G': 0.31, 'T': 0.33}
_POS_PARA = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
_CON_PROB = {'A': [0.28, 0.49, 0.44, 0.55, 0.62, 0.49, 0.67, 0.65, 0.81, 0.21], 'C': [0.82, 0.64, 0.51, 0.64, 0.59, 0.59, 0.58, 0.44, 0.51, 0.24],
             'G': [0.40, 0.54, 0.47, 0.64, 0.64, 0.73, 0.41, 0.41, 0.33, 0.30], 'T': [0.28, 0.24, 0.39, 0.40, 0.55, 0.75, 0.56, 0.69, 0.51, 0.58]}
_CON_W = {'A': 0.11, 'C': 0.12, 'G': 0.15, 'T': 0.14}
_CON_PARA = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.17, 0.0]


def fickett(seq):
    s = seq.upper(); n = max(1, len(s)); score = 0.0
    for b in 'ACGT':
        counts = [s[i::3].count(b) for i in range(3)]
        value = max(counts) / (min(counts) + 1.0)
        for i, th in enumerate(_POS_PARA):
            if value >= th:
                score += _POS_PROB[b][i] * _POS_W[b]; break
        frac = s.count(b) / n
        for i, th in enumerate(_CON_PARA):
            if frac >= th:
                score += _CON_PROB[b][i] * _CON_W[b]; break
    return score


def hexamers(seq, start, length):
    s = seq.upper()
    return [s[i:i + 6] for i in range(start, start + length - 5, 3)]


class HexamerTable:
    """Log ratio of in-frame hexamer frequencies in coding ORFs and in lncRNAs of the training chromosomes."""

    def __init__(self, coding_seqs, noncoding_seqs):
        words = [''.join(w) for w in product('ACGT', repeat=6)]
        c = dict.fromkeys(words, 1.0); nc = dict.fromkeys(words, 1.0)
        for s in coding_seqs:
            st, ln = longest_orf(s)
            for h in hexamers(s, st, ln):
                if h in c:
                    c[h] += 1
        for s in noncoding_seqs:
            for h in hexamers(s, 0, len(s)):
                if h in nc:
                    nc[h] += 1
        tc, tn = sum(c.values()), sum(nc.values())
        self.table = {w: np.log((c[w] / tc) / (nc[w] / tn)) for w in words}

    def score(self, seq, start, length):
        hs = [self.table[h] for h in hexamers(seq, start, length) if h in self.table]
        return float(np.mean(hs)) if hs else 0.0


def save_fit(path, fit):
    """Write the hexamer table and the feature standardisation returned by fit() to a JSON file."""
    import json
    table, mean, std = fit
    json.dump({'hexamer_log_ratio': table.table, 'mean': [float(v) for v in mean], 'std': [float(v) for v in std]}, open(path, 'w'))


def load_fit(path):
    import json
    d = json.load(open(path))
    table = HexamerTable.__new__(HexamerTable); table.table = d['hexamer_log_ratio']
    return table, np.array(d['mean'], dtype=np.float32), np.array(d['std'], dtype=np.float32)


def features(seqs, table):
    """The five ORF features of every sequence (n x 5, float32)."""
    out = []
    for s in seqs:
        st, ln = longest_orf(s)
        out.append([np.log1p(ln), ln / len(s), fickett(s), table.score(s, st, ln), np.log(len(s))])
    return np.asarray(out, dtype=np.float32)


def fit(train_seqs, train_labels):
    """Hexamer table and feature standardisation from the training transcripts: (table, mean, standard deviation)."""
    table = HexamerTable([s for s, y in zip(train_seqs, train_labels) if y == 1], [s for s, y in zip(train_seqs, train_labels) if y == 0])
    f = features(train_seqs, table)
    return table, f.mean(0), f.std(0) + 1e-6
