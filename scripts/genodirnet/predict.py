"""Coding probability of every sequence of a FASTA file with GenoDirNet (NumPy networks; no GPU needed).

    python scripts/genodirnet/predict.py --fasta transcripts.fa --models models/genodirnet --out predictions.tsv \
        [--prior 0.2] [--fusion genodirnet_absolute] [--fdr 0.05]

Scores the FR-CGR network and the sequence-ORF network named in <models>/<fusion>_fusion.json (written by fuse.py),
fuses their logits with the stored logistic regression and writes one line per sequence: identifier and
coding probability. --prior gives the coding fraction expected in the input; the calibrated probability is then
shifted from the class ratio of the validation chromosomes to this prior in closed form. --fdr adds a column that calls
sequences coding at a nominal false discovery rate (Bayesian FDR rule on the calibrated probabilities: sequences are
called in decreasing order of probability while the mean of 1 - p over the calls stays below the level). The hexamer table and the
standardisation of the ORF features are read from <models>/orf_features.json (written by fuse.py). --fusion
genodirnet_absolute selects the networks trained with absolute windows, which are more accurate on fragments of a few
hundred nucleotides.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genodirnet import orf
from genodirnet.data import read_fasta
from genodirnet.metrics import bfdr_calls
from genodirnet.numpy_nets.models import load, predict
from score_numpy import BATCH, batches, encoder


def shift_prior(p, pi, pi_new):
    """Posterior for a new class ratio (Saerens et al. 2002)."""
    a = pi_new * p / pi; b = (1 - pi_new) * (1 - p) / (1 - pi)
    return a / (a + b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fasta', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--prior', type=float, default=None)
    ap.add_argument('--fusion', default='genodirnet', help='genodirnet (relative windows) or genodirnet_absolute')
    ap.add_argument('--fdr', type=float, default=None, help='nominal false discovery rate for a coding/noncoding call column')
    a = ap.parse_args()
    fusion = json.load(open(os.path.join(a.models, f'{a.fusion}_fusion.json')))
    ids, seqs = read_fasta(a.fasta)
    orf_fit = orf.load_fit(os.path.join(a.models, 'orf_features.json'))
    z = []
    for run in fusion['networks']:
        model, name = load(os.path.join(a.models, f'{run}_best.npz')); enc = encoder(name, orf_fit)
        p = predict(model, (batches([enc(s) for s in seqs[i:i + BATCH]]) for i in range(0, len(seqs), BATCH)))
        z.append(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)))
    p = 1 / (1 + np.exp(-(fusion['intercept'] + np.column_stack(z) @ np.array(fusion['coef']))))
    if a.prior is not None:
        p = shift_prior(p, fusion['prior'], a.prior)
    calls = bfdr_calls(p, a.fdr) if a.fdr is not None else None
    with open(a.out, 'w') as fh:
        fh.write('id\tcoding_probability' + ('\tcall\n' if calls is not None else '\n'))
        for j, (i, q) in enumerate(zip(ids, p)):
            fh.write(f'{i}\t{q:.6f}' + (f'\t{"coding" if calls[j] else "noncoding"}\n' if calls is not None else '\n'))
    print(f'{len(ids)} sequences scored; output in {a.out}')


if __name__ == '__main__':
    main()
