"""Evaluation sets of the standard schedule, written to <data>/sets/<key>_seqs.txt (label, tab, sequence).

    python scripts/data/make_sets.py --data data/extraction1 [--draws 10]

test_w<L>       one random window of L = 100, 200, 300, 500 and 1,000 nt per test transcript (seed 0;
                genodirnet.fragments.fixed_windows)
test_w<L>i<n>   the 300- and 1,000-nt windows with n = 1 or 2 insertions or deletions per 100 nt
test_w300sh     the 300-nt windows after a dinucleotide-preserving shuffle (control for compositional signal)
test_w<L>_s<d>  further independent draws of the windows with seeds d = 1, ..., draws - 1
refseq_*        the RefSeq sets written by scripts/data/prepare_refseq.py are scored in the same way
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import split, write_seqs
from genodirnet.fragments import WINDOW_LENGTHS, fixed_windows, indel_windows, shuffled_windows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--draws', type=int, default=10)
    a = ap.parse_args()
    out = os.path.join(a.data, 'sets'); os.makedirs(out, exist_ok=True)
    y, seqs = split(a.data, 'test')
    sets = {f'test_w{L}': fixed_windows(seqs, L) for L in WINDOW_LENGTHS}
    for L in (300, 1000):
        for tag in ('i1', 'i2'):
            sets[f'test_w{L}{tag}'] = indel_windows(seqs, L, tag)
    sets['test_w300sh'] = shuffled_windows(sets['test_w300'])
    for d in range(1, a.draws):
        for L in WINDOW_LENGTHS:
            sets[f'test_w{L}_s{d}'] = fixed_windows(seqs, L, seed=d)
    for key, ws in sets.items():
        write_seqs(os.path.join(out, f'{key}_seqs.txt'), y, ws)
    print(f'{len(sets)} sets written to {out}')


if __name__ == '__main__':
    main()
