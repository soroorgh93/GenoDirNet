"""Split files, data set and fixed test windows of the extended schedule.

A split file (written by scripts/data/prepare_gencode.py) is a TSV with a header and the columns
    id  label  chrom  sequence  cds_start  cds_end
(label 1 coding, 0 lncRNA; cds_start and cds_end 1-based and inclusive, empty for lncRNAs).
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import fit_hexamer_table, one_hot, orf_features
from .sampling import ErrorAwareNoise, FragmentSampler, length_bin

TEST_WINDOW_SETS = ('w100', 'w200', 'w300', 'w500', 'w1000', 'half', 'w300_indel2')


def read_split(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            r['label'] = int(r['label'])
            rows.append(r)
    return rows


def write_rows(rows, path, cols=('id', 'label', 'chrom', 'sequence')):
    with open(path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')


def worker_init(worker_id):
    """DataLoader worker_init_fn: every worker, in every epoch, gets its own random stream."""
    info = torch.utils.data.get_worker_info()
    if hasattr(info.dataset, 'rng'):
        info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


def fit_feature_standardiser(rows, hexamer_table, n_max=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n_max]
    F = np.stack([orf_features(rows[i]['sequence'].upper(), hexamer_table) for i in idx])
    return F.mean(0), F.std(0) + 1e-6


def feature_prep(train_rows, seed=0):
    """Hexamer table (annotated coding sequences against lncRNAs of the training split) and the mean and
    standard deviation of the five ORF features on up to 20,000 training transcripts (drawn with `seed`)."""
    cds = [r['sequence'][int(r['cds_start']) - 1:int(r['cds_end'])].upper() for r in train_rows
           if r['label'] == 1 and r.get('cds_start')]
    lnc = [r['sequence'].upper() for r in train_rows if r['label'] == 0]
    table = fit_hexamer_table(cds, lnc)
    mean, std = fit_feature_standardiser(train_rows, table, seed=seed)
    return table, mean.astype(np.float32), std.astype(np.float32)


class TranscriptDataset(Dataset):
    """One-hot input, mask, standardised ORF features, label and window-length bin of each transcript, with
    fragment-aware windows and error-aware noise applied during training."""

    def __init__(self, rows, hexamer_table, feat_mean, feat_std, seq_len=8192, fragment_aware=True, error_aware=False, seed=0):
        self.rows, self.table, self.mean, self.std, self.seq_len = rows, hexamer_table, feat_mean, feat_std, seq_len
        self.sampler = FragmentSampler() if fragment_aware else None
        self.noise = ErrorAwareNoise() if error_aware else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        s = r['sequence'].upper()
        if self.sampler is not None:
            s = self.sampler(s, self.rng)
        if self.noise is not None:
            s = self.noise(s, self.rng)
        x, m = one_hot(s, self.seq_len)
        f = (orf_features(s, self.table) - self.mean) / self.std
        return (torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(f),
                torch.tensor(r['label'], dtype=torch.long), torch.tensor(length_bin(len(s)), dtype=torch.long))


def make_test_windows(rows, lengths=(100, 200, 300, 500, 1000), half=True, seed=0, min_half=200):
    """One random window per transcript and length (a transcript shorter than the window is kept whole) and,
    with half=True, one window of half the length (at least min_half nucleotides)."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in lengths:
        win = []
        for r in rows:
            s = r['sequence']
            if len(s) <= L:
                w = s
            else:
                a = rng.integers(0, len(s) - L + 1); w = s[a:a + L]
            win.append({**r, 'sequence': w, 'parent_len': len(s)})
        out[f'w{L}'] = win
    if half:
        win = []
        for r in rows:
            s = r['sequence']; L = max(min_half, len(s) // 2)
            if len(s) <= L:
                w = s
            else:
                a = rng.integers(0, len(s) - L + 1); w = s[a:a + L]
            win.append({**r, 'sequence': w, 'parent_len': len(s)})
        out['half'] = win
    return out


def add_indels(rows, rate, seed=0):
    """Windows with single-nucleotide deletions and insertions, each at rate / 2 per nucleotide."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        s = r['sequence']; res = []
        for c in s:
            u = rng.random()
            if u < rate / 2:
                continue
            if u < rate:
                res.append('ACGT'[rng.integers(0, 4)])
            res.append(c)
        out.append({**r, 'sequence': ''.join(res) if res else s})
    return out


def write_test_windows(test_rows, folder):
    """The fixed test windows of the extended schedule (identical for every model): seed 12345 for the windows,
    seed 777 for the insertions and deletions of the 300-nucleotide windows."""
    os.makedirs(folder, exist_ok=True)
    wins = make_test_windows(test_rows, seed=12345)
    for k, rows in wins.items():
        write_rows(rows, os.path.join(folder, f'{k}.tsv'))
    write_rows(add_indels(wins['w300'], 0.02, seed=777), os.path.join(folder, 'w300_indel2.tsv'))
