"""One-hot input and the five ORF features of the extended schedule.

The features are the length of the longest ORF (at least 30 nucleotides, ATG to stop in one of the three forward
frames), its coverage of the input, the Fickett TESTCODE score of the input, the hexamer score of the longest ORF
and the logarithm of the input length. The hexamer table compares in-frame hexamers of the annotated coding
sequences with the hexamers at every position of the lncRNAs of the training chromosomes (pseudocount 1). An input
without an ORF has length, coverage and hexamer score 0. The features are standardised by data.feature_prep.
"""
import math
from itertools import product

import numpy as np

ALPH = 'ACGT'
IDX = {c: i for i, c in enumerate(ALPH)}
STOPS = {'TAA', 'TAG', 'TGA'}


def one_hot(seq, length=8192):
    """4 x length float32 array (cut or zero-padded; ambiguous bases all zero) and the mask of real positions."""
    x = np.zeros((4, length), dtype=np.float32)
    s = seq[:length]
    arr = np.frombuffer(s.encode(), dtype=np.uint8)
    for c, i in IDX.items():
        x[i, :len(s)][arr == ord(c)] = 1.0
    mask = np.zeros(length, dtype=np.float32); mask[:len(s)] = 1.0
    return x, mask


def longest_orf(seq, min_len=30):
    """Start, end (exclusive) and length of the longest ATG-to-stop ORF on the forward strand, or (None, None, 0)."""
    best = (None, None, 0)
    for frame in range(3):
        start = None
        for i in range(frame, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if start is None and codon == 'ATG':
                start = i
            elif codon in STOPS and start is not None:
                if i + 3 - start > best[2]:
                    best = (start, i + 3, i + 3 - start)
                start = None
    if best[2] < min_len:
        return (None, None, 0)
    return best


def fit_hexamer_table(coding_cds, noncoding, pseudo=1.0):
    """Log ratio of hexamer frequencies: in frame in the coding sequences, at every position in the lncRNAs."""
    hexamers = [''.join(p) for p in product(ALPH, repeat=6)]
    cod = {h: pseudo for h in hexamers}
    non = {h: pseudo for h in hexamers}
    for s in coding_cds:
        for i in range(0, len(s) - 5, 3):
            h = s[i:i + 6]
            if h in cod:
                cod[h] += 1
    for s in noncoding:
        for i in range(0, len(s) - 5):
            h = s[i:i + 6]
            if h in non:
                non[h] += 1
    zc, zn = sum(cod.values()), sum(non.values())
    return {h: math.log((cod[h] / zc) / (non[h] / zn)) for h in hexamers}


def hexamer_score(orf_seq, table):
    """Mean log ratio of the in-frame hexamers of an ORF (0 for an ORF shorter than six nucleotides)."""
    n, tot = 0, 0.0
    for i in range(0, len(orf_seq) - 5, 3):
        v = table.get(orf_seq[i:i + 6])
        if v is not None:
            tot += v; n += 1
    return tot / n if n else 0.0


# Lookup tables of the TESTCODE statistic (Fickett 1982) with which the networks of the extended schedule were trained;
# some content entries differ from the tables distributed with CPAT and from genodirnet.orf.
_POS_PROB = {'A': [0.94, 0.68, 0.84, 0.93, 0.58, 0.68, 0.45, 0.34, 0.20, 0.22],
             'C': [0.80, 0.70, 0.70, 0.81, 0.66, 0.48, 0.51, 0.33, 0.30, 0.23],
             'G': [0.90, 0.88, 0.74, 0.64, 0.53, 0.48, 0.27, 0.16, 0.08, 0.08],
             'T': [0.97, 0.97, 0.91, 0.68, 0.69, 0.44, 0.54, 0.20, 0.09, 0.09]}
_POS_WEIGHT = {'A': 0.26, 'C': 0.18, 'G': 0.31, 'T': 0.33}
_POS_PARAM = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
_CONT_PROB = {'A': [0.28, 0.49, 0.44, 0.55, 0.62, 0.49, 0.54, 0.52, 0.36, 0.22],
              'C': [0.82, 0.64, 0.51, 0.64, 0.59, 0.59, 0.41, 0.45, 0.31, 0.11],
              'G': [0.40, 0.54, 0.47, 0.64, 0.64, 0.73, 0.41, 0.41, 0.33, 0.29],
              'T': [0.28, 0.24, 0.39, 0.40, 0.55, 0.75, 0.56, 0.69, 0.51, 0.58]}
_CONT_WEIGHT = {'A': 0.11, 'C': 0.12, 'G': 0.15, 'T': 0.14}
_CONT_PARAM = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.17, 0.17, 0.0]


def fickett_score(seq):
    """Fickett TESTCODE score of a sequence."""
    L = len(seq)
    if L < 3:
        return 0.0
    score = 0.0
    for base in ALPH:
        cnt = [seq[i::3].count(base) for i in range(3)]
        pos_val = max(cnt) / (min(cnt) + 1.0)
        for j, thr in enumerate(_POS_PARAM):
            if pos_val >= thr:
                score += _POS_PROB[base][j] * _POS_WEIGHT[base]; break
        content = seq.count(base) / L
        for j, thr in enumerate(_CONT_PARAM):
            if content >= thr:
                score += _CONT_PROB[base][j] * _CONT_WEIGHT[base]; break
    return score


def orf_features(seq, hexamer_table):
    """[longest ORF length, ORF coverage, Fickett score, hexamer score, log length] (float32)."""
    s, e, ln = longest_orf(seq)
    orf = seq[s:e] if ln else ''
    return np.array([ln, ln / max(len(seq), 1), fickett_score(seq), hexamer_score(orf, hexamer_table),
                     math.log(max(len(seq), 1))], dtype=np.float32)
