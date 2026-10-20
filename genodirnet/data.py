"""Reading the split files written by scripts/data/prepare_gencode.py and the evaluation sets.

A split or set file has one line per transcript: label (1 coding, 0 lncRNA), a tab, the sequence.
The metadata file <split>_meta.csv has the columns gene, transcript, chrom, name, length, label.
"""
import csv
import gzip
import os

import numpy as np

SEQ_LEN = 8192
_LUT = np.full(256, 4, dtype=np.int64)
for _i, _b in enumerate('ACGT'):
    _LUT[ord(_b)] = _i; _LUT[ord(_b.lower())] = _i


def one_hot(seq, length=SEQ_LEN):
    """4 x length one-hot array; a longer sequence is cut, a shorter one zero-padded, ambiguous bases are all zero."""
    codes = _LUT[np.frombuffer(seq[:length].encode('ascii', 'replace'), dtype=np.uint8)]
    x = np.zeros((4, length), dtype=np.float32)
    idx = np.flatnonzero(codes < 4); x[codes[idx], idx] = 1.0
    return x


def read_seqs(path):
    """Labels (array) and sequences (list) of a split or set file."""
    ys, ss = [], []
    with open(path) as fh:
        for line in fh:
            y, s = line.rstrip('\n').split('\t'); ys.append(int(y)); ss.append(s)
    return np.array(ys), ss


def read_fasta(path):
    """Identifiers and sequences of a (optionally gzipped) FASTA file."""
    ids, seqs, buf = [], [], []
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as fh:
        for line in fh:
            if line.startswith('>'):
                if buf or ids:
                    seqs.append(''.join(buf).upper())
                ids.append(line[1:].split()[0]); buf = []
            else:
                buf.append(line.strip())
    if ids:
        seqs.append(''.join(buf).upper())
    return ids, seqs


def read_set(path):
    """Sequences of an evaluation set: a split-format file (labels known), a TSV file of the extended schedule with the
    columns id, label and sequence, or a FASTA file (labels None)."""
    if path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')):
        ids, seqs = read_fasta(path)
        return None, seqs, ids
    if path.endswith('.tsv'):
        with open(path) as fh:
            rows = list(csv.DictReader(fh, delimiter='\t'))
        return np.array([int(r['label']) for r in rows]), [r['sequence'].upper() for r in rows], [r['id'] for r in rows]
    y, seqs = read_seqs(path)
    return y, seqs, None


def set_key(path):
    b = os.path.basename(path)
    for suf in ('_seqs.txt', '.fa.gz', '.fasta.gz', '.fa', '.fasta', '.tsv'):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def write_seqs(path, labels, seqs):
    with open(path, 'w') as fh:
        for y, s in zip(labels, seqs):
            fh.write(f'{int(y)}\t{s}\n')


def read_meta(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def split(data, name):
    """Labels and sequences of the train, val or test split of a data folder."""
    return read_seqs(os.path.join(data, f'{name}_seqs.txt'))


def sets(data, keys=None):
    """Evaluation sets of data/sets: {key: (labels, sequences)}."""
    d = os.path.join(data, 'sets'); out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith('_seqs.txt'):
            k = f[:-len('_seqs.txt')]
            if keys is None or k in keys:
                out[k] = read_seqs(os.path.join(d, f))
    return out
