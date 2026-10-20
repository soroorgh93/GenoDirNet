"""Test transcripts without a close relative among the training transcripts (CD-HIT-EST-2D, 80 % identity).

    python scripts/data/homology_screen.py --data data/extraction1

Runs cd-hit-est-2d (CD-HIT 4.8.1; word length 5) with the training transcripts as the first and the test
transcripts as the second database and writes <data>/test_homology_free.txt, the identifiers of the test transcripts
that share no cluster with a training transcript.
"""
import argparse
import csv
import os
import shutil
import subprocess
import tempfile


def write_fasta(data, split, path):
    meta = list(csv.DictReader(open(os.path.join(data, f'{split}_meta.csv'))))
    with open(os.path.join(data, f'{split}_seqs.txt')) as fs, open(path, 'w') as fh:
        for r, line in zip(meta, fs):
            fh.write(f">{r['transcript']}\n{line.rstrip().split(chr(9))[1]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--identity', type=float, default=0.8)
    a = ap.parse_args()
    exe = shutil.which('cd-hit-est-2d')
    if exe is None:
        raise SystemExit('cd-hit-est-2d not found (CD-HIT 4.8.1)')
    with tempfile.TemporaryDirectory() as tmp:
        for split in ('train', 'test'):
            write_fasta(a.data, split, os.path.join(tmp, f'{split}.fa'))
        out = os.path.join(tmp, 'test_vs_train')
        subprocess.run([exe, '-i', os.path.join(tmp, 'train.fa'), '-i2', os.path.join(tmp, 'test.fa'), '-o', out,
                        '-c', str(a.identity), '-n', '5', '-M', '0', '-T', '0', '-d', '0'], check=True, capture_output=True)
        ids = [line[1:].strip() for line in open(out) if line.startswith('>')]
    with open(os.path.join(a.data, 'test_homology_free.txt'), 'w') as fh:
        fh.write('\n'.join(ids) + '\n')
    print(len(ids), 'test transcripts without a training relative at', a.identity, 'identity')


if __name__ == '__main__':
    main()
