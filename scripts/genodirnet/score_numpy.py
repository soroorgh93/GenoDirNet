"""Score sets of sequences with trained networks of the NumPy implementation (no GPU and no PyTorch needed).

    python scripts/genodirnet/score_numpy.py --data data/extraction1 --models models/genodirnet --out results/genodirnet \
        --runs seq_orf_net_np frcgr_k5_np --sets data/extraction1/sets/*_seqs.txt data/real/wtc11.fa

A set is a file in the split format (label, tab, sequence), a TSV file of the extended schedule (id, label, sequence)
or a FASTA file. The coding probabilities of run <run> on set <key> (--tag followed by the file name without
_seqs.txt, .tsv or .fa) are written to <out>/<run>_<key>_probs.npy; for a FASTA or TSV file, the sequence identifiers
are written to <out>/<key>_ids.txt.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet import orf
from genodirnet.data import one_hot, read_set, set_key, split
from genodirnet.frcgr import network_input
from genodirnet.numpy_nets import set_threads
from genodirnet.numpy_nets.models import FRCGR_INPUTS, load, predict

BATCH = 64


def encoder(name, orf_fit):
    if name in FRCGR_INPUTS:
        k, control = FRCGR_INPUTS[name]
        return lambda s: network_input(s, k, control).transpose(1, 2, 0)
    if name == 'seq_net':
        return lambda s: one_hot(s).T[:, None, :]
    table, mean, std = orf_fit
    return lambda s: (one_hot(s).T[:, None, :], ((orf.features([s], table)[0] - mean) / std).astype(np.float32))


def batches(items):
    if isinstance(items[0], tuple):
        return np.ascontiguousarray(np.stack([a for a, _ in items])), np.ascontiguousarray(np.stack([b for _, b in items]))
    return np.ascontiguousarray(np.stack(items))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data folder with the training split (for the ORF features)')
    ap.add_argument('--models', required=True, help='folder with <run>_best.npz')
    ap.add_argument('--out', required=True); ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--sets', nargs='+', required=True); ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--tag', default='', help='prefix of the set keys (extraction2_ for the sets of the extended schedule)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); set_threads(a.threads)
    orf_fit = None
    for run in a.runs:
        model, name = load(os.path.join(a.models, f'{run}_best.npz'))
        if name == 'seq_orf_net' and orf_fit is None:
            ytr, str_ = split(a.data, 'train'); orf_fit = orf.fit(str_, ytr)
        enc = encoder(name, orf_fit)
        for path in a.sets:
            key = a.tag + set_key(path); f = os.path.join(a.out, f'{run}_{key}_probs.npy')
            if os.path.exists(f):
                continue
            _, seqs, ids = read_set(path)
            if ids is not None:
                with open(os.path.join(a.out, f'{key}_ids.txt'), 'w') as fh:
                    fh.write('\n'.join(ids) + '\n')
            t0 = time.time()
            p = predict(model, (batches([enc(s) for s in seqs[i:i + BATCH]]) for i in range(0, len(seqs), BATCH)))
            np.save(f, p); print(f'{run} {key}: {len(seqs)} sequences, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
