"""Test transcripts and real assembled transcript models with a close relative among the training transcripts.

    python scripts/data/homology_check.py --data data [--minimap2 minimap2]

Aligns the queries to the training transcripts with minimap2 2.31 (-x asm20 -c -N 20). The identity of a query and
a training transcript is the number of identical aligned bases divided by the length of the shorter of the two, the
global identity of CD-HIT. Queries with an identity of at least 0.8 to some training transcript are written to
  <data>/extraction1/test_close_relatives.txt   test transcripts, against the training transcripts of extraction 1
  <data>/real/<set>_close_relatives.txt         WTC11 and CHESS models, against the training transcripts of both
                                                extractions
analysis/make_tables.py reports how much removing them changes the AUROCs.
"""
import argparse
import collections
import csv
import os
import shutil
import subprocess
import tempfile


def write_split_fasta(data, split, fh):
    meta = list(csv.DictReader(open(os.path.join(data, f'{split}_meta.csv'))))
    with open(os.path.join(data, f'{split}_seqs.txt')) as fs:
        for r, line in zip(meta, fs):
            fh.write(f">{r['transcript']}\n{line.rstrip().split(chr(9))[1]}\n")


def close_relatives(minimap2, target_fa, query_fa, identity=0.8):
    paf = subprocess.run([minimap2, '-x', 'asm20', '-c', '-N', '20', '-t', str(os.cpu_count() or 1), target_fa, query_fa],
                         check=True, capture_output=True, text=True).stdout
    pair = collections.defaultdict(float)
    for line in paf.splitlines():
        f = line.split('\t')
        pair[(f[0], f[5])] += int(f[9]) / min(int(f[1]), int(f[6]))
    best = collections.defaultdict(float)
    for (q, _), v in pair.items():
        best[q] = max(best[q], v)
    return sorted(q for q, v in best.items() if v >= identity)


def write_ids(path, ids, what):
    with open(path, 'w') as fh:
        fh.write(''.join(i + '\n' for i in ids))
    print(what, len(ids), 'with at least 80 % identity to a training transcript')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--minimap2', default='minimap2')
    a = ap.parse_args()
    exe = shutil.which(a.minimap2)
    if exe is None:
        raise SystemExit('minimap2 not found (version 2.31)')
    e1 = os.path.join(a.data, 'extraction1')
    with tempfile.TemporaryDirectory() as tmp:
        train1, test1, both = (os.path.join(tmp, f) for f in ('train1.fa', 'test1.fa', 'train_both.fa'))
        for split, path in (('train', train1), ('test', test1)):
            with open(path, 'w') as fh:
                write_split_fasta(e1, split, fh)
        with open(both, 'w') as fh:
            for path in (train1, os.path.join(a.data, 'extraction2', 'train_transcripts.fa')):
                fh.write(open(path).read())
        write_ids(os.path.join(e1, 'test_close_relatives.txt'), close_relatives(exe, train1, test1), 'test transcripts:')
        for key in ('wtc11', 'chess'):
            write_ids(os.path.join(a.data, 'real', f'{key}_close_relatives.txt'),
                      close_relatives(exe, both, os.path.join(a.data, 'real', f'{key}.fa')), f'{key} models:')


if __name__ == '__main__':
    main()
