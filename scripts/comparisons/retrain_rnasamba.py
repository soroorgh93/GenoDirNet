"""RNAsamba retrained with its own software on the training transcripts of the article.

    python scripts/comparisons/retrain_rnasamba.py --data data/extraction1 --models models/rnasamba [--no-fat]

Fragment-aware variant (default): every training transcript is given once complete and twice as a random window
covering 50-100 % of its length (at least 200 nt, seed 7), sampled once before training, because RNAsamba trains on
a fixed set of sequences; --no-fat gives every transcript once complete. Training: rnasamba train with at most 12
epochs and early stopping after 3 epochs without improvement on the 10 % of the training sequences that RNAsamba
sets aside for validation. Writes <models>/rnasamba_retrained[_nofat].hdf5; score it with score_rnasamba.py.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import split
from genodirnet.fragments import retraining_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--models', required=True); ap.add_argument('--no-fat', action='store_true')
    ap.add_argument('--rnasamba', default='rnasamba'); ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--patience', type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.models, exist_ok=True)
    name = 'rnasamba_retrained' + ('_nofat' if a.no_fat else ''); copies = 0 if a.no_fat else 2
    ytr, str_ = split(a.data, 'train'); files = {1: os.path.join(a.models, f'{name}_coding.fa'), 0: os.path.join(a.models, f'{name}_noncoding.fa')}
    handles = {lab: open(f, 'w') for lab, f in files.items()}
    for lab, n, s in retraining_set(ytr, str_, copies=copies):
        handles[lab].write(f'>{n}\n{s}\n')
    for h in handles.values():
        h.close()
    subprocess.run([a.rnasamba, 'train', '-s', str(a.patience), '-e', str(a.epochs), '-v', '3',
                    os.path.join(a.models, f'{name}.hdf5'), files[1], files[0]], check=True)


if __name__ == '__main__':
    main()
