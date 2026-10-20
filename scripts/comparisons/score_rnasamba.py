"""Score sets of sequences with RNAsamba weight files (released or retrained) through rnasamba_numpy.py.

    python scripts/comparisons/score_rnasamba.py --out results/genodirnet \
        --model rnasamba=full_length_weights.hdf5 rnasamba_partial=partial_length_weights.hdf5 \
                rnasamba_retrained=rnasamba_retrained.hdf5 rnasamba_retrained_nofat=rnasamba_retrained_nofat.hdf5 \
        --sets data/extraction1/sets/*_seqs.txt data/real/wtc11.fa

Writes <out>/<model>_<key>_probs.npy for every model and set (key: file name without _seqs.txt or .fa).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..')); sys.path.insert(0, HERE)
from genodirnet.data import read_set, set_key
from rnasamba_numpy import RNAsamba


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True); ap.add_argument('--model', nargs='+', required=True, help='NAME=WEIGHT_FILE')
    ap.add_argument('--sets', nargs='+', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for item in a.model:
        name, path = item.split('=', 1); model = RNAsamba(path)
        for s in a.sets:
            f = os.path.join(a.out, f'{name}_{set_key(s)}_probs.npy')
            if os.path.exists(f):
                continue
            t0 = time.time(); _, seqs, _ = read_set(s)
            np.save(f, model.predict(seqs)); print(f'{name} {set_key(s)}: {len(seqs)} sequences, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
