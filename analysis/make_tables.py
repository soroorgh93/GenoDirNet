"""Every table of the article and every number quoted in its text, from the run outputs.

    python analysis/make_tables.py --results results --data data --out paper

Reads the coding probabilities and run records written by the scripts (results/genodirnet, results/genodirnet2,
results/real) and writes paper/tables/<name>.tsv, one file per table, and paper/numbers.tsv, every number quoted in
the text with a short description.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analysis.common import (GDN, GDN_A, GDN_P, GDN_P0, LENGTH_CLASSES, LNC_CLASSES, Runs, auroc_mean, frame_signal,
                             get, lncrna_class, logit)
from genodirnet.data import read_meta, read_seqs
from genodirnet.fragments import WINDOW_LENGTHS, fat_window, window_starts
from genodirnet.metrics import (auroc_ci, auroc_difference, bfdr_calls, bootstrap_auroc, delong_test, ece, metrics,
                                sensitivity_at_fpr)
from genodirnet.orf import longest_orf

WKEYS = [f'test_w{L}' for L in WINDOW_LENGTHS] + ['test']
WNAMES = [f'{L:,}' for L in WINDOW_LENGTHS] + ['Complete']
DRAWS = [''] + [f'_s{d}' for d in range(1, 10)]
NT_TOKENS = -(-8192 // 6)                 # 6-mer tokens of an input of 8,192 nucleotides


class Out:
    def __init__(self, folder):
        self.folder = folder; os.makedirs(os.path.join(folder, 'tables'), exist_ok=True); self.numbers = []

    def table(self, name, rows, columns):
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(os.path.join(self.folder, 'tables', f'{name}.tsv'), sep='\t', index=False)
        print(f'\n== {name}'); print(df.to_string(index=False))

    def num(self, key, value, text=''):
        self.numbers.append((key, value, text))

    def close(self):
        with open(os.path.join(self.folder, 'numbers.tsv'), 'w') as fh:
            fh.write('key\tvalue\tdescription\n')
            for k, v, t in self.numbers:
                fh.write(f'{k}\t{v}\t{t}\n')
        print(f'\n{len(self.numbers)} numbers written to {self.folder}/numbers.tsv')


def f1(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    t = f'{v:.1f}'
    return '0.0' if t == '-0.0' else t


def pct(v):
    return None if v is None else 100 * v


def ci_text(m, lo, hi):
    return f'{f1(100 * m)} (CI {f1(100 * lo)} to {f1(100 * hi)})'


def adiff(y, runs_a, runs_b):
    """Difference of two models as the difference of their rounded AUROCs (as in the tables) with its paired bootstrap CI."""
    _, lo, hi = auroc_difference(y, runs_a, runs_b)
    return rd(auroc_mean(y, runs_a), auroc_mean(y, runs_b)) / 100, lo, hi


def rd(a, b):
    """Difference of two AUROCs after rounding each to the decimal of the tables (points), so that text and tables agree."""
    return round(100 * a, 1) - round(100 * b, 1)


def up(x):
    """Round up to one decimal, for bounds stated as 'within x points'."""
    return float(np.ceil(x * 10 - 1e-9) / 10)


def mean_metric(R, spec, split, fn):
    runs = get(R, spec, split)
    return None if not runs else float(np.mean([fn(R.labels(split), p) for p in runs]))


def fmt_p(p):
    if p < 1e-15:
        return '<1e-15'
    return f'{p:.1e}' if p < 1e-3 else f'{p:.2g}'


def qtext(q):
    """A P or q value for the text, in TeX."""
    t = fmt_p(q)
    m = re.match(r'^([\d.]+)e-0*(\d+)$', t)
    return f'${m.group(1)} \\times 10^{{-{m.group(2)}}}$' if m else ('$<10^{-15}$' if t == '<1e-15' else t)


def bh(p):
    """Benjamini-Hochberg adjusted P values."""
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); adj[i] = prev
    return adj


# ------------------------------------------------------------------------------------------- data
def data_numbers(out, data, R):
    e1 = os.path.join(data, 'extraction1')
    meta = {s: read_meta(os.path.join(e1, f'{s}_meta.csv')) for s in ('train', 'val', 'test')}
    for s, m in meta.items():
        lab = np.array([int(r['label']) for r in m]); ln = np.array([int(r['length']) for r in m])
        out.num(f'data_{s}', f'{len(m):,}', f'{s} transcripts (first extraction)')
        out.num(f'data_{s}_coding', f'{int(lab.sum()):,}', f'{s} coding transcripts')
        out.num(f'data_{s}_lncrna', f'{int((lab == 0).sum()):,}', f'{s} lncRNAs')
        if s == 'test':
            out.num('data_test_median_coding', f'{int(np.median(ln[lab == 1])):,}', 'median length of coding test transcripts (nt)')
            out.num('data_test_median_lncrna', f'{int(np.median(ln[lab == 0])):,}', 'median length of test lncRNAs (nt)')
        if s == 'val':
            out.num('data_val_coding_share', f'{100 * lab.mean():.0f}', 'coding share of the validation chromosomes (%)')
    free = {line.strip() for line in open(os.path.join(e1, 'test_homology_free.txt')) if line.strip()}
    keep = np.array([r['transcript'] in free for r in meta['test']])
    out.num('homology_removed', f'{int((~keep).sum())}', 'test transcripts with >= 80 % identity to a training transcript')
    shift = []
    for _, spec in T1:
        for p in get(R, spec):
            shift.append(abs(roc_auc_score(R.y[keep], p[keep]) - roc_auc_score(R.y, p)))
    out.num('homology_max_shift', f'{100 * max(shift):.1f}', 'largest change of the AUROC of a model of the complete-transcript table without them (points)')
    close = {line.strip() for line in open(os.path.join(e1, 'test_close_relatives.txt')) if line.strip()}
    keep = np.array([r['transcript'] not in close for r in meta['test']])
    out.num('homology2_removed', f'{int((~keep).sum())}', 'test transcripts with >= 80 % identity to a training transcript (minimap2)')
    for split, models in (('test', [s for _, s in T1]), ('test_w300', [s for _, _, s in CONTROLLED + RELEASED])):
        shift = [abs(roc_auc_score(R.y[keep], p[keep]) - roc_auc_score(R.y, p)) for spec in models for p in get(R, spec, split)]
        out.num(f'homology2_max_shift_{split}', f'{100 * max(shift):.1f}', f'largest change of an AUROC without them, {split} (points)')
    e2 = os.path.join(data, 'extraction2')
    for s in ('train', 'val', 'test'):
        f = os.path.join(e2, f'{s}.tsv')
        if os.path.exists(f):
            out.num(f'data2_{s}', f'{sum(1 for _ in open(f)) - 1:,}', f'{s} transcripts (second extraction)')
    _, test_seqs = read_seqs(os.path.join(e1, 'test_seqs.txt'))
    ln = np.array([len(s) for s in test_seqs])
    for L in WINDOW_LENGTHS:
        out.num(f'whole_w{L}', f'{100 * (ln <= L).mean():.1f}', f'test transcripts scored whole at {L} nt (%)')
    out.num('whole_w1000_lncrna', f'{100 * (ln[R.y == 0] <= 1000).mean():.1f}', 'test lncRNAs scored whole at 1,000 nt (%)')
    out.num('whole_w1000_coding', f'{100 * (ln[R.y == 1] <= 1000).mean():.1f}', 'coding test transcripts scored whole at 1,000 nt (%)')
    _, train_seqs = read_seqs(os.path.join(e1, 'train_seqs.txt'))
    for mode in ('relative', 'absolute'):
        rng = np.random.default_rng(0)
        lens = np.array([len(fat_window(s, rng, mode)) for _ in range(20) for s in train_seqs])
        for t in (300, 1000):
            out.num(f'fat_{mode}_le{t}', f'{100 * (lens <= t).mean():.1f}', f'{mode} windows: training inputs of {t} nt or shorter (%)')
        out.num(f'fat_{mode}_min', f'{lens.min()}', f'{mode} windows: shortest training input (nt)')
        out.num(f'fat_{mode}_median', f'{int(np.median(lens)):,}', f'{mode} windows: median training input (nt)')
    for name in ('refseq_human', 'refseq_mouse'):
        y = R.labels(name)
        out.num(f'{name}_n', f'{len(y):,}', f'{name} transcripts'); out.num(f'{name}_coding', f'{int(y.sum()):,}', 'coding')
        out.num(f'{name}_lncrna', f'{int((y == 0).sum()):,}', 'lncRNA')


# ------------------------------------------------------------------------------- complete transcripts
T1 = [('Length-only LR', 'length_lr'), ('Periodic Markov chain (order 5)', 'markov5'), ('Word-frequency LR (k = 1 to 6)', 'wordfreq_lr'),
      ('Word-frequency MLP (k = 1 to 6)', 'wordfreq_mlp'), ('Frame-resolved MLP (k = 4)', 'framemlp_k4'), ('Frame-resolved MLP (k = 5)', 'framemlp_k5'),
      ('ORF-feature LR', 'orf_lr'), ('CPC2, retrained', 'cpc2_retrained'), ('CPAT, retrained', 'cpat_retrained'),
      ('RNAsamba, retrained', 'rnasamba_retrained'), ('ResNet18 on the FR-CGR (k = 7)', 'resnet18_k7'), ('FR-CGR network (k = 7)', 'frcgr_k7'),
      ('FR-CGR network (k = 6)', 'frcgr_k6'), ('FR-CGR network (k = 5)', 'frcgr_k5'), ('Sequence network', 'seq_net'),
      ('Sequence-ORF network', 'seq_orf_net'), ('GenoDirNet', GDN), ('Nucleotide Transformer', 'nt'), ('CPC2', 'cpc2'), ('CPAT', 'cpat'), ('RNAsamba', 'rnasamba'),
      ('RNAsamba, partial-length model', 'rnasamba_partial')]


def size_of(R, spec):
    """Parameters and multiply-accumulate operations per transcript of a model (None for the released tools)."""
    if isinstance(spec, tuple):
        parts = [size_of(R, s) for s in spec]
        return sum(p[0] for p in parts), sum(p[1] for p in parts)
    rec = R.record(spec) or {}
    if spec == 'nt':                      # lower bound: parameters x 6-mer tokens of an 8,192-nt input
        return rec['params'], rec['params'] * NT_TOKENS
    if 'macs' not in rec:
        return None, None
    return rec.get('params', rec['macs']), rec['macs']


def table_complete(out, R):
    rows = []
    for label, spec in T1:
        runs = get(R, spec)
        m = [metrics(R.y, p) for p in runs]
        params, macs = size_of(R, spec)
        fp = lambda v: 'n/a' if v is None else ('<0.001' if v < 1e3 else (f'{v / 1e6:.3f}' if v < 1e5 else f'{v / 1e6:.2f}'))
        fm = lambda v: 'n/a' if v is None else ('<0.001' if v < 1e3 else (f'{v / 1e6:.3f}' if v < 1e5 else
                                                                    (f'{v / 1e6:.1f}' if v < 1e10 else f'{v / 1e6:,.0f}')))
        rows.append([label, fp(params), fm(macs), len(runs)] + [f'{100 * np.mean([r[k] for r in m]):.1f}' for k in ('auroc', 'auprc', 'accuracy')] +
                    [f'{np.mean([r["mcc"] for r in m]):.3f}', f'{100 * np.mean([r["f1"] for r in m]):.1f}'])
    out.table('complete', rows, ['Model', 'Params (M)', 'MMAC', 'Runs', 'AUROC', 'AUPRC', 'Acc.', 'MCC', 'F1'])
    g = get(R, GDN)
    out.num('gdn_complete', ci_text(*auroc_ci(R.y, g)), 'GenoDirNet, complete test transcripts, mean of runs (95 % CI)')
    m, lo, hi = auroc_ci(R.y, g)
    out.num('gdn_complete_ci', f'{f1(100 * lo)} to {f1(100 * hi)}', 'GenoDirNet, complete test transcripts: 95 % CI')
    for t in ('cpc2', 'cpat', 'rnasamba', 'cpc2_retrained', 'cpat_retrained', 'rnasamba_retrained'):
        out.num(f'gdn_minus_{t}', ci_text(*adiff(R.y, g, get(R, t))), f'GenoDirNet minus {t}, complete transcripts')
    out.num('gdn_minus_seq_orf_net', ci_text(*adiff(R.y, g, get(R, 'seq_orf_net'))), 'GenoDirNet minus its sequence-ORF network, complete transcripts')
    for t in ('nt', 'nt_nofat'):
        out.num(f'gdn_minus_{t}', ci_text(*adiff(R.y, g, get(R, t))), f'GenoDirNet minus {t}, complete transcripts')
    out.num('impl_diff_complete', f'{abs(rd(auroc_mean(R.y, get(R, GDN)), auroc_mean(R.y, get(R, GDN_P)))):.1f}',
            'GenoDirNet: difference between the two implementations on complete transcripts (points)')
    out.num('mac_ratio_seqorf_frcgr', f'{size_of(R, "seq_orf_net")[1] / size_of(R, "frcgr_k5")[1]:.0f}', 'MACs of the sequence-ORF network / FR-CGR network')
    spread_c, spread_w = [], []
    for name in ('frcgr_k7', 'frcgr_k6', 'frcgr_k5', 'resnet18_k7', 'seq_net', 'seq_orf_net'):
        a = [roc_auc_score(R.y, p) for p in R.runs(name)]; b = [roc_auc_score(R.y, p) for p in R.runs(name, 'test_w300')]
        if len(a) > 1:
            spread_c.append(max(a) - min(a)); spread_w.append(max(b) - min(b))
    out.num('run_spread_complete', f'{up(100 * max(spread_c)):.1f}', 'largest difference between two runs of a network, standard schedule, complete (points)')
    out.num('run_spread_w300', f'{up(100 * max(spread_w)):.1f}', 'the same on 300-nt windows (points)')


# ------------------------------------------------------------------------------------------ fragments
CONTROLLED = [('GenoDirNet', 'yes', GDN_P), ('GenoDirNet', 'no', GDN_P0), ('RNAsamba, retrained', 'yes', 'rnasamba_retrained'),
              ('RNAsamba, retrained', 'no', 'rnasamba_retrained_nofat'), ('CPAT, retrained', 'yes', 'cpat_retrained'),
              ('CPAT, retrained', 'no', 'cpat_retrained_nofat'), ('CPC2, retrained', 'yes', 'cpc2_retrained'),
              ('CPC2, retrained', 'no', 'cpc2_retrained_nofat'), ('Nucleotide Transformer', 'yes', 'nt'), ('Nucleotide Transformer', 'no', 'nt_nofat'),
              ('Periodic Markov chain (order 5)', 'no', 'markov5'), ('Periodic Markov chain (order 4)', 'no', 'markov4')]
RELEASED = [('RNAsamba', '--', 'rnasamba'), ('RNAsamba, partial-length model', '--', 'rnasamba_partial'), ('CPAT', '--', 'cpat'), ('CPC2', '--', 'cpc2')]


def table_fragments(out, R):
    au, se = {}, {}
    for label, fat, spec in CONTROLLED + RELEASED:
        au[(label, fat)] = [mean_metric(R, spec, k, roc_auc_score) for k in WKEYS]
        se[(label, fat)] = [mean_metric(R, spec, k, sensitivity_at_fpr) for k in WKEYS]
    rows = [['Trained under the protocol of this study', '', '', '', '', '', '', '', '']]
    rows += [[lab, fat, len(get(R, spec))] + [f1(pct(v)) for v in au[(lab, fat)]] for lab, fat, spec in CONTROLLED]
    rows += [['Released models', '', '', '', '', '', '', '', '']]
    rows += [[lab, fat, 1] + [f1(pct(v)) for v in au[(lab, fat)]] for lab, fat, spec in RELEASED]
    out.table('fragments', rows, ['Model', 'FAT', 'Runs'] + WNAMES)
    srows = [[lab, fat] + [f1(pct(v)) for v in se[(lab, fat)]] for lab, fat, _ in CONTROLLED + RELEASED]
    out.table('S_sensitivity', srows, ['Model', 'FAT'] + WNAMES)
    g = lambda k: get(R, GDN_P, k)
    for t in ('rnasamba', 'rnasamba_retrained', 'cpat_retrained', 'cpc2_retrained', 'nt'):
        for L in WINDOW_LENGTHS:
            if get(R, t, f'test_w{L}'):
                out.num(f'gdn_minus_{t}_w{L}', ci_text(*adiff(R.y, g(f'test_w{L}'), get(R, t, f'test_w{L}'))), f'GenoDirNet minus {t}, {L} nt')
    for spec, name in ((GDN_P, 'gdn'), ('rnasamba', 'rnasamba'), ('rnasamba_retrained', 'rnasamba_retrained')):
        for L in (100, 200, 300):
            out.num(f'sens_{name}_w{L}', f'{100 * mean_metric(R, spec, f"test_w{L}", sensitivity_at_fpr):.0f}', f'{name}: sensitivity at 5 % FPR, {L} nt (%)')
    short = [au[(lab, '--')][i] for lab in ('CPAT', 'CPC2') for i in range(3)]
    out.num('cpat_cpc2_short_range', f'{100 * min(short):.1f} to {100 * max(short):.1f}', 'released CPAT and CPC2 on 100- to 300-nt windows')
    gd = au[('GenoDirNet', 'yes')]
    rr = [rd(gd[i], au[('RNAsamba, retrained', 'yes')][i]) for i in range(3)]
    out.num('rnasamba_retrained_margin_range', f'{min(rr):.1f} to {max(rr):.1f}', 'GenoDirNet minus retrained RNAsamba, 100- to 300-nt windows (points)')
    margins = [rd(gd[i], au[key][i]) for key in (('CPAT', '--'), ('CPC2', '--'), ('CPAT, retrained', 'yes'), ('CPC2, retrained', 'yes')) for i in range(3)]
    out.num('cpat_cpc2_margin_range', f'{min(margins):.0f} to {max(margins):.0f}', 'GenoDirNet minus CPAT and CPC2 (released and retrained), 100- to 300-nt windows (points)')
    margins = [rd(gd[-1], au[key][-1]) for key in (('CPAT', '--'), ('CPC2', '--'), ('CPAT, retrained', 'yes'), ('CPC2, retrained', 'yes'))]
    out.num('cpat_cpc2_complete_margin_range', f'{min(margins):.1f} to {max(margins):.1f}', 'GenoDirNet minus CPAT and CPC2 (released and retrained), complete transcripts (points)')
    nt, gd = au[('Nucleotide Transformer', 'yes')], au[('GenoDirNet', 'yes')]
    for i, L in enumerate(WINDOW_LENGTHS):
        out.num(f'nt_minus_gdn_w{L}', f1(rd(nt[i], gd[i])), f'Nucleotide Transformer minus GenoDirNet, {L} nt')
        k = f'test_w{L}'
        out.num(f'gdn_minus_seq_orf_np_w{L}', ci_text(*adiff(R.y, g(k), get(R, 'seq_orf_net_np', k))), f'GenoDirNet minus its sequence-ORF network, {L} nt')
        out.num(f'gdn_minus_markov_w{L}', ci_text(*adiff(R.y, g(k), get(R, 'markov5', k))), f'GenoDirNet minus the periodic Markov chain, {L} nt')
        if get(R, GDN_A, k):
            out.num(f'gdna_minus_markov_w{L}', ci_text(*adiff(R.y, get(R, GDN_A, k), get(R, 'markov5', k))),
                    f'GenoDirNet (absolute windows) minus the periodic Markov chain, {L} nt')
            out.num(f'gdna_minus_seq_orf_np_absolute_w{L}', ci_text(*adiff(R.y, get(R, GDN_A, k), get(R, 'seq_orf_net_np_absolute', k))),
                    f'GenoDirNet (absolute windows) minus its sequence-ORF network, {L} nt')
    out.num('markov_minus_frcgr_np_w300', ci_text(*adiff(R.y, get(R, 'markov5', 'test_w300'), get(R, 'frcgr_k5_np', 'test_w300'))),
            'periodic Markov chain minus the FR-CGR network (relative windows), 300 nt')
    # training runs of GenoDirNet (matched seeds of its two networks): per-run AUROC and the spread between runs
    rows, spread = [], {}
    for i, _ in enumerate(g('test')):
        cells = [f'Run {i + 1}']
        for k, name in zip(['test'] + [f'test_w{L}' for L in WINDOW_LENGTHS], ['complete'] + [f'w{L}' for L in WINDOW_LENGTHS]):
            a = [100 * roc_auc_score(R.y, p) for p in g(k)]
            cells.append(f'{a[i]:.2f}' if i < len(a) else '-')
            spread[name] = up(max(a) - min(a)) if len(a) > 1 else None
        rows.append(cells)
    out.table('S_gdn_runs', rows, ['Run', 'Complete'] + WNAMES[:len(WINDOW_LENGTHS)])
    out.num('gdnp_runs', str(len(rows)), 'training runs of GenoDirNet (NumPy networks, relative windows)')
    for name, v in spread.items():
        if v is not None:
            out.num(f'gdnp_run_spread_{name}', f'{v:.1f}', f'GenoDirNet: largest difference between its training runs, {name} (points)')
    if len(rows) > 1:
        out.num('gdnp_run_spread_max_windows', f'{max(v for n, v in spread.items() if n != "complete" and v is not None):.1f}',
                'GenoDirNet: largest difference between its training runs on any window length (points)')
    for lab in ('CPAT, retrained', 'CPC2, retrained', 'RNAsamba, retrained'):
        rel = {'CPAT, retrained': 'CPAT', 'CPC2, retrained': 'CPC2', 'RNAsamba, retrained': 'RNAsamba'}[lab]
        d = [100 * (a - b) for a, b in zip(au[(lab, 'yes')], au[(rel, '--')])]
        out.num(f'retrained_vs_released_{rel.lower()}', ', '.join(f'{L}: {x:+.1f}' for L, x in zip(WNAMES, d)), f'{lab} (FAT) minus released {rel}')
    return au


# ------------------------------------------------------------------------------------- repeated draws
DRAW_MODELS = [('GenoDirNet', GDN_P), ('GenoDirNet, no FAT', GDN_P0), ('RNAsamba, retrained', 'rnasamba_retrained'),
               ('RNAsamba, retrained, no FAT', 'rnasamba_retrained_nofat'), ('CPAT, retrained', 'cpat_retrained'),
               ('CPC2, retrained', 'cpc2_retrained'), ('Periodic Markov chain (order 5)', 'markov5'), ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial'),
               ('CPAT', 'cpat'), ('CPC2', 'cpc2')]


def draw_aurocs(R, spec, L):
    """AUROC of the first training run of a model on each of the ten window draws (None if a draw is missing)."""
    out = []
    for d in DRAWS:
        runs = get(R, spec, f'test_w{L}{d}')
        out.append(roc_auc_score(R.y, runs[0]) if runs else None)
    return None if any(v is None for v in out) else np.array(out)


def table_draws(out, R):
    rows, prow, pvals = [], [], []
    ref = {L: draw_aurocs(R, GDN_P, L) for L in WINDOW_LENGTHS}
    for label, spec in DRAW_MODELS:
        cells = [label]
        for L in WINDOW_LENGTHS:
            a = draw_aurocs(R, spec, L)
            cells.append('--' if a is None else f'{100 * a.mean():.1f} ({100 * a.std(ddof=1):.1f})')
        rows.append(cells)
        if spec is GDN_P:
            continue
        cells = [label]
        for L in WINDOW_LENGTHS:
            a = draw_aurocs(R, spec, L)
            if a is None:
                cells.append('--'); continue
            d = 100 * (ref[L] - a); wins = int((d > 0).sum())
            p = binomtest(wins, len(d), 0.5).pvalue
            pvals.append(p); cells.append([d.mean(), d.min(), d.max(), wins, p])
        prow.append(cells)
    adj = iter(bh(pvals))
    diff_rows, all_ten, other = [], [], []
    for cells in prow:
        row = [cells[0]]
        for L, c in zip(WINDOW_LENGTHS, cells[1:]):
            if c == '--':
                row.append('--'); continue
            m, lo, hi, wins, p = c; q = next(adj)
            row.append(f'{m:+.1f} ({lo:+.1f} to {hi:+.1f}); {wins}/10; q = {fmt_p(q)}')
            out.num(f'draw_diff_{cells[0]}_w{L}', f'{abs(m):.1f}', f'GenoDirNet against {cells[0]}, {L} nt: mean absolute difference over the draws')
            out.num(f'draw_wins_{cells[0]}_w{L}', f'{wins}', f'GenoDirNet against {cells[0]}, {L} nt: draws won by GenoDirNet')
            out.num(f'draw_losses_{cells[0]}_w{L}', f'{10 - wins}', f'GenoDirNet against {cells[0]}, {L} nt: draws lost by GenoDirNet')
            if wins < 10:
                out.num(f'draws_q_{cells[0]}_w{L}', fmt_p(q), 'adjusted q of a comparison with fewer than ten wins')
            (all_ten if wins == 10 else other).append((cells[0], L, wins, q))
        diff_rows.append(row)
    out.num('draws_all_ten', f'{len(all_ten)} of {len(all_ten) + len(other)}', 'comparisons in which GenoDirNet was ahead in all ten draws')
    rr = np.concatenate([100 * (ref[L] - draw_aurocs(R, 'rnasamba_retrained', L)) for L in (100, 200, 300)])
    out.num('rnasamba_retrained_draw_margin_range', f'{rr.min():.1f} to {rr.max():.1f}',
            'GenoDirNet minus retrained RNAsamba over the ten draws, 100- to 300-nt windows (smallest to largest)')
    mk = 100 * (draw_aurocs(R, 'markov5', 100) - ref[100])
    out.num('markov_draw_margin_w100', f'{mk.mean():.1f}', 'periodic Markov chain minus GenoDirNet, 100 nt, mean over the ten draws')
    out.num('draws_all_ten_q', fmt_p(max(q for *_, q in all_ten)) if all_ten else '--', 'largest BH-adjusted q among them')
    out.num('draws_other', '; '.join(f'{lab}, {L} nt: {w}/10' for lab, L, w, _ in other) or 'none', 'comparisons with fewer than ten wins')
    out.table('draws', rows, ['Model'] + WNAMES[:-1])
    out.table('draws_differences', diff_rows, ['Comparison with GenoDirNet'] + WNAMES[:-1])
    sd = [float(np.std(draw_aurocs(R, spec, L), ddof=1)) for _, spec in DRAW_MODELS for L in WINDOW_LENGTHS if draw_aurocs(R, spec, L) is not None]
    out.num('draw_sd_range', f'{100 * min(sd):.1f} to {100 * max(sd):.1f}', 'standard deviation of the AUROC over the ten window draws (points)')
    out.num('draw_tests', f'{len(pvals)}', 'sign tests in the draws table (Benjamini-Hochberg adjusted together)')


# ------------------------------------------------------------------------- fragment-aware training
T_FAT = [('Periodic Markov chain (order 5)', None, 'markov5', None), ('ORF-feature LR', 'orf_lr', 'orf_lr_nofat', 'orf_lr_absolute'),
         ('Word-frequency LR (k = 1 to 6)', 'wordfreq_lr', 'wordfreq_lr_nofat', 'wordfreq_lr_absolute'),
         ('Frame-resolved MLP (k = 5)', 'framemlp_k5', 'framemlp_k5_nofat', 'framemlp_k5_absolute'),
         ('FR-CGR network (k = 5)', 'frcgr_k5_np', 'frcgr_k5_np_nofat', 'frcgr_k5_np_absolute'), ('Sequence network', 'seq_net_np', 'seq_net_np_nofat', None),
         ('Sequence-ORF network', 'seq_orf_net_np', 'seq_orf_net_np_nofat', 'seq_orf_net_np_absolute'), ('GenoDirNet', GDN_P, GDN_P0, GDN_A),
         ('RNAsamba, retrained', 'rnasamba_retrained', 'rnasamba_retrained_nofat', None), ('CPAT, retrained', 'cpat_retrained', 'cpat_retrained_nofat', None),
         ('CPC2, retrained', 'cpc2_retrained', 'cpc2_retrained_nofat', None), ('Nucleotide Transformer', 'nt', 'nt_nofat', None)]
FAT_KEYS = ('test', 'test_w300', 'test_w100')


def table_fat(out, R):
    rows, gains, abs_cost = [], [], []
    for label, rel, none, absolute in T_FAT:
        v = {'none': [mean_metric(R, none, k, roc_auc_score) for k in FAT_KEYS],
             'rel': [mean_metric(R, rel, k, roc_auc_score) if rel is not None else None for k in FAT_KEYS],
             'abs': [mean_metric(R, absolute, k, roc_auc_score) if absolute is not None else None for k in FAT_KEYS]}
        runs = [len(get(R, s)) if s is not None else 0 for s in (none, rel, absolute)]
        rows.append([label] + [f1(pct(x)) for t in ('none', 'rel', 'abs') for x in v[t]] + ['/'.join(str(r) if r else '-' for r in runs)])
        if rel is not None:
            g_abs = rd(v['abs'][1], v['none'][1]) if v['abs'][1] is not None else None
            gains.append((label, rd(v['rel'][1], v['none'][1]), rd(v['rel'][0], v['none'][0]), g_abs))
        if absolute is not None and v['abs'][0] is not None:
            d = [rd(a, b) for a, b in zip(v['abs'], v['rel'])]
            abs_cost.append(-d[0])
            out.num(f'absolute_minus_relative_{label.split(" (")[0].replace(" ", "_").lower()}', ', '.join(f'{x:+.1f}' for x in d),
                    f'{label}: absolute minus relative windows (complete, 300 nt, 100 nt)')
    out.num('absolute_cost_complete_range', f'{min(abs_cost):.1f} to {max(abs_cost):.1f}', 'loss on complete transcripts from absolute instead of relative windows (points)')
    out.num('absolute_families', {5: 'five', 6: 'six', 7: 'seven'}.get(len(abs_cost), str(len(abs_cost))), 'families trained with absolute windows')
    out.table('fat', rows, ['Model'] + [f'{t} {k}' for t in ('None', 'Relative', 'Absolute') for k in ('Complete', '300', '100')] + ['Runs (none/relative/absolute)'])
    out.table('fig_fat_gains', [[lab, f'{a:.4f}', '' if c is None else f'{c:.4f}'] for lab, a, _, c in gains],
              ['Model', 'Relative windows, 300 nt', 'Absolute windows, 300 nt'])
    d = {lab: a for lab, a, _, _ in gains}
    out.num('fat_gain_orf_lr', f'{d["ORF-feature LR"]:.1f}', 'FAT gain, ORF-feature LR, 300 nt')
    wf = [d['Word-frequency LR (k = 1 to 6)'], d['Frame-resolved MLP (k = 5)']]
    out.num('fat_gain_wordfreq_range', f'{min(wf):.1f} to {max(wf):.1f}', 'FAT gain, word-frequency models, 300 nt')
    cn = [d[k] for k in ('FR-CGR network (k = 5)', 'Sequence network', 'Sequence-ORF network', 'GenoDirNet')]
    out.num('fat_gain_networks_range', f'{min(cn):.1f} to {max(cn):.1f}', 'FAT gain, convolutional networks and GenoDirNet, 300 nt')
    rt = [d[k] for k in ('RNAsamba, retrained', 'CPAT, retrained', 'CPC2, retrained')]
    out.num('fat_gain_retrained_range', f'{min(rt):.1f} to {max(rt):.1f}', 'FAT gain, retrained released tools, 300 nt')
    for k in ('RNAsamba, retrained', 'CPAT, retrained', 'CPC2, retrained', 'Nucleotide Transformer'):
        out.num(f'fat_gain_{k.split(",")[0].replace(" ", "_").lower()}', f'{d[k]:.1f}', f'FAT gain, {k}, 300 nt')
    c = [b for _, _, b, _ in gains]
    out.num('fat_change_complete_range', f'{min(c):+.1f} to {max(c):+.1f}', 'change on complete transcripts from FAT (points)')
    for spec_w, spec_o, name in ((GDN_P, GDN_P0, 'gdn'), ('rnasamba_retrained', 'rnasamba_retrained_nofat', 'rnasamba'), ('nt', 'nt_nofat', 'nt'),
                                 ('cpat_retrained', 'cpat_retrained_nofat', 'cpat'), ('cpc2_retrained', 'cpc2_retrained_nofat', 'cpc2')):
        for L in WINDOW_LENGTHS:
            k = f'test_w{L}'
            if get(R, spec_w, k) and get(R, spec_o, k):
                out.num(f'fat_gain_{name}_w{L}', ci_text(*adiff(R.y, get(R, spec_w, k), get(R, spec_o, k))), f'FAT gain of {name}, {L} nt')
    for L in WINDOW_LENGTHS:
        k = f'test_w{L}'
        if get(R, GDN_A, k):
            out.num(f'gdn_absolute_minus_relative_w{L}', ci_text(*adiff(R.y, get(R, GDN_A, k), get(R, GDN_P, k))),
                    f'GenoDirNet, absolute minus relative windows, {L} nt')
    if get(R, GDN_A, 'test'):
        out.num('gdn_absolute_minus_relative_complete', ci_text(*adiff(R.y, get(R, GDN_A, 'test'), get(R, GDN_P, 'test'))),
                'GenoDirNet, absolute minus relative windows, complete transcripts')
        d = {k: rd(auroc_mean(R.y, get(R, GDN_A, k)), auroc_mean(R.y, get(R, GDN_P, k))) for k in ('test_w100', 'test_w300', 'test')}
        lo_, hi_ = f'{min(d["test_w100"], d["test_w300"]):.1f}', f'{max(d["test_w100"], d["test_w300"]):.1f}'
        out.num('gdna_gain_short_range', lo_ if lo_ == hi_ else f'{lo_} to {hi_}',
                'GenoDirNet, absolute minus relative windows, 100 and 300 nt (points, range)')
        out.num('gdna_cost_complete', f'{-d["test"]:.1f}', 'GenoDirNet, loss on complete transcripts from absolute instead of relative windows (points)')
        out.table('S_absolute', [[lab] + [f1(pct(mean_metric(R, s, k, roc_auc_score))) for k in WKEYS] for lab, s in
                                 (('GenoDirNet, relative windows', GDN_P), ('GenoDirNet, absolute windows', GDN_A), ('GenoDirNet, no FAT', GDN_P0))],
                  ['Model'] + WNAMES)


# ----------------------------------------------------------------------------------------- controls
T_CONTROLS = [('FR-CGR (reference)', 'frcgr_k5_np'), ('Channels summed (FCGR)', 'frcgr_k5_summed_np'), ('Random partition of starts', 'frcgr_k5_partition_np'),
              ('FR-CGR with permuted cells', 'frcgr_k5_permuted_np'), ('One kernel per depthwise layer', 'frcgr_k5_onekernel_np'),
              ('No neighbourhood blocks', 'frcgr_k5_noneighbour_np'), ('Plain 3 x 3 convolutions', 'plaincnn_k5_np')]


def matched_loss(R, ref, other, split):
    """Mean AUROC loss of `other` against `ref` over matched training runs, with its paired 95 % bootstrap interval."""
    tags = [t for t in R.tags(ref, split) if t in R.tags(other, split)]
    y = R.labels(split)
    b = np.mean([bootstrap_auroc(y, R.load(ref + t, split)) - bootstrap_auroc(y, R.load(other + t, split)) for t in tags], axis=0)
    m = np.mean([roc_auc_score(y, R.load(ref + t, split)) - roc_auc_score(y, R.load(other + t, split)) for t in tags])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return 100 * m, 100 * lo, 100 * hi


def table_controls(out, R, data):
    rows = []
    for label, name in T_CONTROLS:
        c, w = mean_metric(R, name, 'test', roc_auc_score), mean_metric(R, name, 'test_w300', roc_auc_score)
        row = [label, len(R.runs(name)), f1(pct(c)), f1(pct(w))]
        if name != 'frcgr_k5_np':
            for split in ('test', 'test_w300'):
                m, lo, hi = matched_loss(R, 'frcgr_k5_np', name, split)
                row.append(f'{f1(m)} ({f1(lo)} to {f1(hi)})')
        else:
            row += ['', '']
        rows.append(row)
    out.table('controls', rows, ['Input or network', 'Runs', 'Complete', '300 nt', 'Loss, complete (CI)', 'Loss, 300 nt (CI)'])
    _, seqs = read_seqs(os.path.join(data, 'extraction1', 'test_seqs.txt'))
    out.num('frame_signal_auroc', f'{100 * roc_auc_score(R.y, frame_signal(seqs)):.1f}', 'frame signal as a score, complete test transcripts')
    out.num('markov_w300', f'{100 * mean_metric(R, "markov5", "test_w300", roc_auc_score):.1f}', 'periodic Markov chain, 300 nt')
    out.num('frcgr_np_w300', f'{100 * mean_metric(R, "frcgr_k5_np", "test_w300", roc_auc_score):.1f}', 'FR-CGR network (NumPy series), 300 nt')
    out.num('frcgr_minus_markov_complete', f'{100 * (mean_metric(R, "frcgr_k5_np", "test", roc_auc_score) - mean_metric(R, "markov5", "test", roc_auc_score)):.1f}',
            'FR-CGR network minus periodic Markov chain, complete transcripts')
    out.num('markov_params', f'{R.record("markov5")["macs"]:,}', 'parameters of the periodic Markov chain')
    if R.record('markov4'):                       # the chain for which the 5-letter FR-CGR is a sufficient statistic
        for k, name in (('test', 'complete'), ('test_w300', 'w300'), ('test_w100', 'w100')):
            out.num(f'markov4_{name}', f'{100 * mean_metric(R, "markov4", k, roc_auc_score):.1f}', f'periodic Markov chain of order 4, {name}')


# ------------------------------------------------------------------------- position of the windows
def table_position(out, R, data):
    e1 = os.path.join(data, 'extraction1')
    meta = read_meta(os.path.join(e1, 'test_meta.csv')); _, seqs = read_seqs(os.path.join(e1, 'test_seqs.txt'))
    cds = {r['transcript']: int(r['cds_length']) for r in read_meta(os.path.join(e1, 'test_cds.csv'))}
    starts = window_starts(seqs, 300)
    frac = np.full(len(seqs), np.nan)
    for i, (m, s) in enumerate(zip(meta, seqs)):
        if R.y[i] != 1 or m['transcript'] not in cds:
            continue
        st, ln = longest_orf(s)
        if ln == 0 or abs(ln - cds[m['transcript']]) > 3:
            continue
        w0, w1 = starts[i], starts[i] + min(len(s), 300)
        frac[i] = max(0, min(w1, st + ln) - max(w0, st)) / (w1 - w0)
    located = int(np.isfinite(frac).sum())
    groups = [('Inside CDS', frac == 1), ('Partial overlap', (frac > 0) & (frac < 1)), ('UTR only', frac == 0)]
    ref_rows = []
    for L in (100, 200, 300):                     # reference AUROC 100 - u/2 for the windows of Table 2
        st_L, kind = window_starts(seqs, L), []
        for i, s in enumerate(seqs):
            if np.isfinite(frac[i]):
                st, ln = longest_orf(s); w0, w1 = st_L[i], st_L[i] + min(len(s), L)
                kind.append('utr' if min(w1, st + ln) <= max(w0, st) else 'in' if (w0 >= st and w1 <= st + ln) else 'part')
        n = {k: kind.count(k) for k in ('in', 'part', 'utr')}; tot = len(kind); u = 100 * n['utr'] / tot
        ref_rows.append(['First', L, f"{100 * n['in'] / tot:.1f}", f"{100 * n['part'] / tot:.1f}", f'{u:.1f}', f'{100 - u / 2:.1f}'])
        out.num(f'reference1_auroc_w{L}', f'{100 - u / 2:.1f}', f'reference AUROC 100 - u/2, {L}-nt windows of the first extraction')
    out.num('position_located', f'{located:,} ({100 * located / int(R.y.sum()):.0f} %)', 'coding test transcripts whose longest ORF gives the CDS position')
    for g, sel in groups:
        out.num(f'position_n_{g.split()[0].lower()}', f'{int(sel.sum()):,} ({100 * sel.sum() / located:.1f} %)', f'300-nt windows: {g}')
    neg = R.y == 0
    rows = []
    utr = []
    for label, spec in (('GenoDirNet', GDN_P), ('GenoDirNet, absolute windows', GDN_A), ('Sequence-ORF network', 'seq_orf_net_np'),
                        ('FR-CGR network (k = 5)', 'frcgr_k5_np'), ('RNAsamba, retrained', 'rnasamba_retrained'), ('RNAsamba', 'rnasamba'),
                        ('ORF-feature LR', 'orf_lr'), ('CPC2', 'cpc2')):
        runs = get(R, spec, 'test_w300'); cells = [label]
        if not runs:
            continue
        for _, sel in groups:
            yy = np.r_[np.ones(sel.sum()), np.zeros(neg.sum())]
            a = np.mean([roc_auc_score(yy, np.r_[p[sel], p[neg]]) for p in runs])
            s = np.mean([(p[sel] > np.quantile(p[neg], 0.95)).mean() for p in runs])
            cells += [f'{100 * a:.1f}', f'{100 * s:.1f}']
        utr.append(100 * a)
        rows.append(cells)
    out.num('position_utr_max', f'{max(utr):.1f}', 'highest AUROC of a model on untranslated windows (%)')
    out.table('S_position', rows, ['Model'] + [f'{g} {m}' for g, _ in groups for m in ('AUROC', 'Sensitivity')])
    # control: the same windows after a dinucleotide-preserving shuffle, and GC content alone as a score
    _, w300 = read_seqs(os.path.join(e1, 'sets', 'test_w300_seqs.txt'))
    gc = np.array([(s.count('G') + s.count('C')) / max(1, len(s)) for s in w300])
    ctrl, pairs = [], [('Inside CDS', frac == 1), ('UTR only', frac == 0)]
    def _auroc(p, sel):
        return 100 * roc_auc_score(np.r_[np.ones(sel.sum()), np.zeros(neg.sum())], np.r_[p[sel], p[neg]])
    for label, spec in (('GenoDirNet, relative windows', GDN_P), ('GenoDirNet, absolute windows', GDN_A)):
        orig, shuf = get(R, spec, 'test_w300'), get(R, spec, 'test_w300sh')
        if not orig or not shuf:
            continue
        cells = [label]
        for g, sel in pairs:
            a_o = np.mean([_auroc(p, sel) for p in orig]); a_s = np.mean([_auroc(p, sel) for p in shuf])
            cells += [f'{a_o:.1f}', f'{a_s:.1f}']
            key = 'cds' if g == 'Inside CDS' else 'utr'
            tag = 'gdnp' if spec is GDN_P else 'gdna'
            out.num(f'ctrl_{tag}_{key}', f'{a_o:.1f}', f'{label}, {g} 300-nt windows vs lncRNA windows, AUROC (%)')
            out.num(f'ctrl_{tag}_{key}_shuffled', f'{a_s:.1f}', f'{label}, the same after a dinucleotide shuffle (%)')
        ctrl.append(cells)
    cells = ['GC content']
    for g, sel in pairs:
        a = _auroc(gc, sel); cells += [f'{a:.1f}', f'{a:.1f}']
        out.num(f'ctrl_gc_{"cds" if g == "Inside CDS" else "utr"}', f'{a:.1f}', f'GC content as a score, {g} windows vs lncRNA windows, AUROC (%)')
    ctrl.append(cells)
    out.table('S_utr_control', ctrl, ['Model', 'Inside CDS', 'Inside CDS, shuffled', 'UTR only', 'UTR only, shuffled'])
    e2 = os.path.join(data, 'extraction2')
    test = pd.read_csv(os.path.join(e2, 'test.tsv'), sep='\t')
    cod = test[(test.label == 1) & test.cds_start.notna()].reset_index(drop=True)
    rows = []
    for L in (100, 200, 300):
        win = pd.read_csv(os.path.join(e2, 'test_windows', f'w{L}.tsv'), sep='\t').set_index('id')['sequence']
        n = {'in': 0, 'part': 0, 'utr': 0}
        for _, r in cod.iterrows():
            w = win[r['id']]; st = r['sequence'].find(w); en = st + len(w); a, b = int(r.cds_start) - 1, int(r.cds_end)
            n['in' if (st >= a and en <= b) else 'utr' if (en <= a or st >= b) else 'part'] += 1
        tot = sum(n.values()); u = 100 * n['utr'] / tot
        ref_rows.append(['Second', L, f"{100 * n['in'] / tot:.1f}", f"{100 * n['part'] / tot:.1f}", f'{u:.1f}', f'{100 - u / 2:.1f}'])
        out.num(f'reference_auroc_w{L}', f'{100 - u / 2:.1f}', f'reference AUROC 100 - u/2, {L}-nt windows of the second extraction')
    out.num('position_coding_second_extraction', f'{len(cod):,}', 'coding test transcripts of the second extraction')
    out.table('S_reference', ref_rows, ['Extraction', 'Window (nt)', 'Inside CDS (%)', 'Partial (%)', 'UTR (%)', 'Reference AUROC (%)'])


# -------------------------------------------------------------------- the FR-CGR network added to other models
ADDON = [('CPAT, retrained', 'cpat_retrained'), ('CPC2, retrained', 'cpc2_retrained'), ('RNAsamba, retrained', 'rnasamba_retrained'),
         ('Word-frequency MLP (k = 1 to 6)', 'wordfreq_mlp'), ('Periodic Markov chain (order 5)', 'markov5'), ('Nucleotide Transformer', 'nt'),
         ('Sequence-ORF network', 'seq_orf_net_np')]
ADDON_KEYS = ('test_w100', 'test_w300', 'test')


def addon_auroc(R, model, fr, key):
    """AUROC of a logistic regression on the logits of a model and of the FR-CGR network, fitted on the validation
    chromosomes (runs of the model paired with runs of the FR-CGR network by position, cycling the shorter list)."""
    mv, fv = R.runs(model, 'val'), R.runs(fr, 'val'); mt, ft = R.runs(model, key), R.runs(fr, key)
    if not (mv and fv and mt and ft):
        return None
    n = max(len(mv), len(fv)); n = min(n, len(mt) if len(mt) > 1 else n, len(ft) if len(ft) > 1 else n)
    out = []
    for i in range(n):
        a, b = mv[i % len(mv)], fv[i % len(fv)]; at, bt = mt[i % len(mt)], ft[i % len(ft)]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([logit(a), logit(b)]), R.yv)
        out.append(roc_auc_score(R.y, lr.predict_proba(np.column_stack([logit(at), logit(bt)]))[:, 1]))
    return float(np.mean(out))


def table_addon(out, R):
    rows = []
    for label, m in ADDON:
        alone = [mean_metric(R, m, k, roc_auc_score) for k in ADDON_KEYS]
        if None in alone:
            continue
        cells = [label]
        for fr, tag in (('frcgr_k5_np', 'rel'), ('frcgr_k5_np_absolute', 'abs')):
            for k, a0 in zip(ADDON_KEYS, alone):
                a1 = addon_auroc(R, m, fr, k)
                cells.append('--' if a1 is None else f'{100 * a1:.1f}')
                if a1 is not None:
                    short = m.replace('_retrained', '').replace('_np', '')
                    out.num(f'addon_{tag}_{short}_{k.replace("test_", "") or "complete"}', f'{rd(a1, a0):.1f}',
                            f'{label} with the FR-CGR network ({tag}) minus alone, {k} (points)')
        rows.append([label] + [f'{100 * a:.1f}' for a in alone] + cells[1:])
    cnn = [abs(rd(addon_auroc(R, m, 'frcgr_k5_np', k), mean_metric(R, m, k, roc_auc_score))) for m in ('seq_orf_net_np', 'nt') for k in ADDON_KEYS
           if addon_auroc(R, m, 'frcgr_k5_np', k) is not None]
    out.num('addon_rel_max_cnn', f'{max(cnn):.1f}', 'largest change of the sequence-ORF network or the Nucleotide Transformer from the FR-CGR network (relative windows), points')
    out.table('addon', rows, ['Model', 'Alone 100', 'Alone 300', 'Alone complete', 'With FR-CGR (relative) 100', 'With FR-CGR (relative) 300',
                              'With FR-CGR (relative) complete', 'With FR-CGR (absolute) 100', 'With FR-CGR (absolute) 300', 'With FR-CGR (absolute) complete'])


# ---------------------------------------------------------------------------------- errors, calibration
ERR_MODELS = [('FR-CGR network', 'frcgr_k5'), ('Sequence network', 'seq_net'), ('Sequence-ORF network', 'seq_orf_net'), ('GenoDirNet', GDN),
              ('ORF-feature LR', 'orf_lr'), ('CPC2', 'cpc2'), ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial')]


def errors(out, R, data):
    meta = read_meta(os.path.join(data, 'extraction1', 'test_meta.csv'))
    lens = np.array([int(m['length']) for m in meta]); cls = np.array([lncrna_class(m['name']) for m in meta])
    rows_a, rows_b = [], []
    for label, spec in ERR_MODELS:
        runs = get(R, spec)
        rows_a.append([label] + [f1(pct(np.mean([roc_auc_score(R.y[(lens >= lo) & (lens < hi)], p[(lens >= lo) & (lens < hi)]) for p in runs])))
                                 for lo, hi, _ in LENGTH_CLASSES])
        rows_b.append([label] + [f1(pct(np.mean([(p[(R.y == 0) & (cls == c)] > np.quantile(p[R.y == 1], 0.05)).mean() for p in runs])))
                                 for c in LNC_CLASSES])
    out.table('fig_errors_length', rows_a, ['Model'] + [n for _, _, n in LENGTH_CLASSES])
    out.table('fig_errors_lncrna_class', rows_b, ['Model'] + list(LNC_CLASSES))
    sizes = [int(((lens >= lo) & (lens < hi)).sum()) for lo, hi, _ in LENGTH_CLASSES]
    out.num('length_class_sizes', ', '.join(f'{n}: {s}' for (_, _, n), s in zip(LENGTH_CLASSES, sizes)), 'test transcripts per length class')
    out.num('length_share_500_5000', f'{100 * sum(sizes[1:4]) / len(lens):.0f}', 'test transcripts of 500 to 5,000 nt (%)')
    out.num('lncrna_class_sizes', ', '.join(f'{c}: {int(((R.y == 0) & (cls == c)).sum())}' for c in LNC_CLASSES + ('overlapping',)), 'test lncRNAs per class')


def platt(pv, yv, p):
    lg = lambda q: logit(q).reshape(-1, 1)
    return LogisticRegression(C=1e4, max_iter=1000).fit(lg(pv), yv).predict_proba(lg(p))[:, 1]


def calibration(out, R):
    rows = []
    for label, spec in (('GenoDirNet', GDN), ('Sequence-ORF network', 'seq_orf_net'), ('Sequence network', 'seq_net'),
                        ('FR-CGR network (k = 5)', 'frcgr_k5'), ('CPC2', 'cpc2'), ('CPAT', 'cpat'), ('RNAsamba', 'rnasamba')):
        runs, vals = get(R, spec), get(R, spec, 'val')
        m0 = [metrics(R.y, p) for p in runs]
        cells = [label, f1(100 * np.mean([ece(R.y, p) for p in runs]))] + [f1(100 * np.mean([m[k] for m in m0])) for k in ('accuracy', 'f1')]
        cells.append(f'{np.mean([m["mcc"] for m in m0]):.3f}')
        if not isinstance(spec, tuple) and len(vals) == len(runs):
            rc = [platt(v, R.yv, p) for v, p in zip(vals, runs)]; m1 = [metrics(R.y, p) for p in rc]
            cells += [f1(100 * np.mean([ece(R.y, p) for p in rc]))] + [f1(100 * np.mean([m[k] for m in m1])) for k in ('accuracy', 'f1')]
            cells.append(f'{np.mean([m["mcc"] for m in m1]):.3f}')
        else:
            cells += ['--', '--', '--', '--']
        rows.append(cells)
    out.table('S_calibration', rows, ['Model', 'ECE', 'Accuracy', 'F1', 'MCC', 'ECE, recalibrated', 'Accuracy, recalibrated', 'F1, recalibrated',
                                      'MCC, recalibrated'])
    fdr_calls(out, R)


FDR_LEVELS = (0.01, 0.05, 0.10)


def fdr_calls(out, R):
    """Realised false discovery rate and sensitivity of the Bayesian FDR rule on the complete test transcripts and on the
    300-nt windows, for calibrated and uncalibrated outputs."""
    rows = []
    specs = [('GenoDirNet', GDN, False), ('Sequence-ORF network', 'seq_orf_net', False), ('Sequence-ORF network, recalibrated', 'seq_orf_net', True),
             ('RNAsamba', 'rnasamba', False), ('RNAsamba, recalibrated', 'rnasamba', True), ('CPC2', 'cpc2', False), ('CPC2, recalibrated', 'cpc2', True)]
    for label, spec, recal in specs:
        cells = [label]
        for key in ('test', 'test_w300'):
            runs = get(R, spec, key); vals = get(R, spec, 'val')
            if recal:
                runs = [platt(v, R.yv, p) for v, p in zip(vals, runs)]
            for q0 in FDR_LEVELS:
                fdr, sens = [], []
                for p in runs:
                    c = bfdr_calls(np.asarray(p, dtype=float), q0)
                    fdr.append((c & (R.y == 0)).sum() / max(1, c.sum())); sens.append((c & (R.y == 1)).sum() / (R.y == 1).sum())
                cells += [f'{100 * np.mean(fdr):.1f}', f'{100 * np.mean(sens):.1f}']
                tag = label.lower().replace(' ', '_').replace(',', '').replace('-', '_')
                out.num(f'fdr_{tag}_{key}_q{int(100 * q0)}', f'{100 * np.mean(fdr):.1f}', f'{label}, {key}: realised FDR at nominal {100 * q0:.0f} % (%)')
                out.num(f'fdrsens_{tag}_{key}_q{int(100 * q0)}', f'{100 * np.mean(sens):.1f}', f'{label}, {key}: sensitivity at nominal FDR {100 * q0:.0f} % (%)')
        rows.append(cells)
    out.table('S_fdr', rows, ['Model'] + [f'{set_} {m} q{int(100 * q)}' for set_ in ('complete', '300 nt') for q in FDR_LEVELS for m in ('FDR', 'sens.')])


def compute(out, R):
    nt = R.record('nt')['params']; gdn_p, gdn_m = size_of(R, GDN)
    tokens = int(np.ceil(8192 / 6))
    out.num('nt_params', f'{nt / 1e6:.1f} million', 'parameters of the Nucleotide Transformer')
    out.num('param_ratio_nt_gdn', f'{nt / gdn_p:.0f}', 'parameters of the Nucleotide Transformer / GenoDirNet')
    out.num('nt_tokens', f'{tokens:,}', '6-mer tokens of an 8,192-nt input')
    out.num('nt_macs_lower_bound', f'{nt * tokens / 1e9:.0f} billion', 'parameters x tokens of an 8,192-nt input (lower bound)')
    out.num('mac_ratio_nt_gdn', f'{nt * tokens / gdn_m:.0f}', 'MACs of the Nucleotide Transformer (lower bound) / GenoDirNet')
    out.num('gdn_params', f'{gdn_p / 1e6:.2f} million', 'GenoDirNet: parameters')
    out.num('gdn_mmac', f'{gdn_m / 1e6:.0f} million', 'GenoDirNet: MACs per input (every input is padded to 8,192 nt)')
    for L in (100, 300, 1000):
        out.num(f'mac_ratio_nt_gdn_w{L}', f'{nt * -(-L // 6) / gdn_m:.0f}', f'Nucleotide Transformer lower bound for a {L}-nt input / GenoDirNet')
    runtime(out, R)
    return nt * tokens, gdn_m


RUNTIME_NAMES = {'genodirnet': 'GenoDirNet (both networks)', 'seq_orf_net_np': 'Sequence-ORF network', 'frcgr_k5_np': 'FR-CGR network',
                 'cpc2': 'CPC2 standalone'}


def runtime(out, R):
    """Wall-clock time and peak memory of scoring on a CPU (scripts/comparisons/runtime.py)."""
    f = os.path.join(R.folder, 'runtime.json')
    if not os.path.exists(f):
        return
    rt = json.load(open(f))
    cpu = re.sub(r'\((R|TM)\)', '', rt['cpu']).replace(' Processor', ' processor').replace(' @ ', ' at ').replace('GHz', ' GHz')
    out.num('runtime_cpu', re.sub(r'\s+', ' ', cpu).strip(), 'processor of the runtime measurements')
    rows = []
    for method, label in RUNTIME_NAMES.items():
        recs = [r for r in rt['runs'] if r['method'] == method]
        if not recs:
            continue
        for th in sorted({r['threads'] for r in recs}):
            cells = [label]
            for key in ('complete', 'w300'):
                r = [x for x in recs if x['threads'] == th and x['set'] == key]
                if r:
                    r = r[0]; per10k = r['seconds'] * 10000 / r['n']
                    cells += [f'{per10k:.0f}', f'{r["peak_mb"]:,.0f}']
                    out.num(f'runtime_{method}_{key}_t{th}', f'{per10k:.0f}', f'{label}, {th} thread(s): seconds per 10,000 {key} inputs')
                    out.num(f'memory_{method}_{key}_t{th}', f'{r["peak_mb"]:,.0f}', f'{label}, {th} thread(s): peak memory (MB), {key} inputs')
                else:
                    cells += ['-', '-']
            if th == 1:
                rows.append(cells)
    for key in ('complete', 'w300'):
        r = [x for x in rt['runs'] if x['set'] == key][0]
        out.num(f'runtime_n_{key}', f'{r["n"]:,}', f'inputs scored in the runtime measurement ({key})')
        out.num(f'runtime_mean_length_{key}', f'{r["nt"] / r["n"]:,.0f}', f'mean input length (nt) in the runtime measurement ({key})')
    ov = []
    for key in ('complete', 'w300'):
        g = [x for x in rt['runs'] if x['set'] == key and x['method'] == 'genodirnet' and x['threads'] == 1]
        so = [x for x in rt['runs'] if x['set'] == key and x['method'] == 'seq_orf_net_np' and x['threads'] == 1]
        if g and so:
            ov.append(100 * (g[0]['seconds'] / so[0]['seconds'] - 1))
    if ov:
        out.num('runtime_fusion_overhead', f'{max(ov):.0f}', 'time of GenoDirNet over its sequence-ORF network alone (%, larger of the two sets)')
    two = [(r['method'], r['set']) for r in rt['runs'] if r['threads'] == 2]
    if two:
        ratio = []
        for method, key in two:
            a = [x for x in rt['runs'] if x['method'] == method and x['set'] == key]
            t1 = [x for x in a if x['threads'] == 1]; t2 = [x for x in a if x['threads'] == 2]
            if t1 and t2:
                ratio.append(t1[0]['seconds'] / t2[0]['seconds'])
        out.num('runtime_speedup_two_threads', f'{min(ratio):.1f} to {max(ratio):.1f}', 'speed-up from a second thread (range over methods and sets)')
    out.table('runtime', rows, ['Method', 'Complete: s per 10,000', 'Complete: peak MB', '300 nt: s per 10,000', '300 nt: peak MB'])


# -------------------------------------------------------------------------------- RefSeq and mouse
T_EXT = [('GenoDirNet', GDN_P), ('GenoDirNet, absolute windows', GDN_A), ('GenoDirNet, no FAT', GDN_P0), ('RNAsamba, retrained', 'rnasamba_retrained'),
         ('RNAsamba, retrained, no FAT', 'rnasamba_retrained_nofat'), ('CPAT, retrained', 'cpat_retrained'), ('CPC2, retrained', 'cpc2_retrained'),
         ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial'), ('CPAT', 'cpat'), ('CPC2', 'cpc2'),
         ('Nucleotide Transformer', 'nt'), ('Nucleotide Transformer, no FAT', 'nt_nofat')]


def table_external(out, R, data):
    meta = read_meta(os.path.join(data, 'extraction1', 'sets', 'refseq_human_meta.csv'))
    testchr = np.array([m['chrom'] in ('chr4', 'chr9', 'chr14', 'chr19', 'chr21') for m in meta])
    out.num('refseq_human_test_chromosomes', f'{int(testchr.sum()):,}', 'human RefSeq transcripts of genes on the test chromosomes')
    yh = R.labels('refseq_human')
    rows = []
    for label, spec in T_EXT:
        h = get(R, spec, 'refseq_human')
        cells = [label, f1(pct(auroc_mean(yh, h))), f1(pct(np.mean([roc_auc_score(yh[testchr], p[testchr]) for p in h]) if h else None))]
        for k in ('refseq_mouse', 'refseq_mouse_w1000', 'refseq_mouse_w300'):
            cells.append(f1(pct(auroc_mean(R.labels(k), get(R, spec, k)))))
        rows.append(cells)
    out.table('S_external', rows, ['Model', 'Human, all genes', 'Human, test chromosomes', 'Mouse, complete', 'Mouse, 1,000 nt', 'Mouse, 300 nt'])
    for t in ('cpat', 'cpc2', 'rnasamba', 'nt'):
        g = [p[testchr] for p in get(R, GDN_P, 'refseq_human')]; o = [p[testchr] for p in get(R, t, 'refseq_human')]
        out.num(f'refseq_gdn_minus_{t}', f1(rd(auroc_mean(yh[testchr], g), auroc_mean(yh[testchr], o))), f'human RefSeq, test chromosomes: GenoDirNet minus {t}')
    ym = R.labels('refseq_mouse_w300')
    for t in ('rnasamba', 'rnasamba_retrained', 'cpat_retrained', 'nt'):
        if get(R, t, 'refseq_mouse_w300'):
            out.num(f'mouse_w300_gdn_minus_{t}', ci_text(*adiff(ym, get(R, GDN_P, 'refseq_mouse_w300'), get(R, t, 'refseq_mouse_w300'))),
                    f'mouse, 300 nt: GenoDirNet minus {t}')


# ------------------------------------------------------------------------------------------- indels
T_INDEL = [('GenoDirNet', GDN_P), ('GenoDirNet, absolute windows', GDN_A), ('GenoDirNet, no FAT', GDN_P0), ('RNAsamba, retrained', 'rnasamba_retrained'),
           ('RNAsamba, retrained, no FAT', 'rnasamba_retrained_nofat'), ('CPAT, retrained', 'cpat_retrained'), ('CPC2, retrained', 'cpc2_retrained'),
           ('Nucleotide Transformer', 'nt'), ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial'), ('CPAT', 'cpat'), ('CPC2', 'cpc2')]
IKEYS = ['test_w300', 'test_w300i1', 'test_w300i2', 'test_w1000', 'test_w1000i1', 'test_w1000i2']


def table_indels(out, R):
    au, rows = {}, []
    for label, spec in T_INDEL:
        if isinstance(spec, tuple):
            vals = [auroc_mean(R.y, get(R, spec, k)) for k in IKEYS]
        else:
            tags = [t for t in R.tags(spec, 'test_w300i1') if all(os.path.exists(R.path(spec + t, k)) for k in IKEYS)]
            vals = [auroc_mean(R.y, [R.load(spec + t, k) for t in tags]) if tags else None for k in IKEYS]
        au[label] = vals
        rows.append([label] + [f1(pct(v)) for v in vals])
    out.table('S_indels', rows, ['Model', '300 nt, 0', '300 nt, 1', '300 nt, 2', '1,000 nt, 0', '1,000 nt, 1', '1,000 nt, 2'])
    g, r = au['GenoDirNet'], au['RNAsamba']
    out.num('indel_gdn_minus_rnasamba_one', f'{rd(g[1], r[1]):.1f} and {rd(g[4], r[4]):.1f}', 'one error per 100 nt: GenoDirNet minus released RNAsamba (300, 1,000 nt)')
    out.num('indel_gdn_minus_rnasamba_two', f'{rd(g[2], r[2]):.1f} and {rd(g[5], r[5]):.1f}', 'two errors per 100 nt: GenoDirNet minus released RNAsamba (300, 1,000 nt)')
    rr = au['RNAsamba, retrained']
    out.num('indel_gdn_minus_rnasamba_retrained_two', f'{rd(g[2], rr[2]):.1f} and {rd(g[5], rr[5]):.1f}',
            'two errors per 100 nt: GenoDirNet minus retrained RNAsamba (300, 1,000 nt)')
    out.num('indel_gdn_minus_rnasamba_retrained_one', f'{rd(g[1], rr[1]):.1f} and {rd(g[4], rr[4]):.1f}',
            'one error per 100 nt: GenoDirNet minus retrained RNAsamba (300, 1,000 nt)')
    if au.get('Nucleotide Transformer') and None not in au['Nucleotide Transformer']:
        n = au['Nucleotide Transformer']
        out.num('indel_gdn_minus_nt_two', f'{rd(g[2], n[2]):.1f} and {rd(g[5], n[5]):.1f}',
                'two errors per 100 nt: GenoDirNet minus the Nucleotide Transformer (300, 1,000 nt)')
    for label in ('GenoDirNet', 'RNAsamba, retrained', 'Nucleotide Transformer', 'CPAT, retrained', 'CPC2, retrained'):
        v = au[label]
        if None not in v:
            out.num(f'indel_loss_{label.split(",")[0].replace(" ", "_").lower()}', f'{rd(v[0], v[2]):.1f} and {rd(v[3], v[5]):.1f}',
                    f'{label}: loss from 0 to 2 errors per 100 nt (300, 1,000 nt)')


# ------------------------------------------------------------------------------------- GenoDirNet-2
G2KEYS = ['full', 'w1000', 'w500', 'w300', 'w200', 'w100', 'w300_indel2']


def g2runs(G, name):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(G, f'{name}_s[0-9].json')))]


def best_tau(G):
    """Temperature of the length-weighted loss with the highest mean validation objective."""
    taus = {}
    for f in glob.glob(os.path.join(G, 'lengthweighted_tau*_s[0-9].json')):
        t = re.search(r'tau([0-9.]+)_s', f).group(1); taus.setdefault(t, []).append(json.load(open(f))['val_auroc'])
    return max(taus, key=lambda t: np.mean(taus[t])), {t: float(np.mean(v)) for t, v in taus.items()}


def extraction2_labels(data, k):
    f = os.path.join(data, 'extraction2', 'test.tsv' if k == 'full' else os.path.join('test_windows', f'{k}.tsv'))
    return pd.read_csv(f, sep='\t', usecols=['label']).label.values


def table_genodirnet2(out, G, gdn_macs, nt_macs, R, data):
    tau, taus = best_tau(G)
    out.num('lengthweighted_tau', tau, 'temperature of the length-weighted loss chosen on the validation chromosomes')
    out.num('lengthweighted_tau_validation', ', '.join(f'{t}: {v:.2f}' for t, v in sorted(taus.items())), 'mean validation objective per temperature')
    spec = [('GenoDirNet-SO (reference)', 'so'), ('+ likelihood-ratio features', 'llr'), ('+ error-aware augmentation', 'erroraware'),
            ('+ pretraining (default network)', 'pretrain'), ('+ synthetic windows', 'synthetic'), ('+ deeper network (no pretraining)', 'deeper'),
            ('+ length-weighted loss', f'lengthweighted_tau{tau}'), ('GenoDirNet-2 (all additions)', 'gdn2'), ('Sequence network of the second search', 'compact')]
    lm = json.load(open(os.path.join(G, 'lm', 'genlr.json')))
    rows, s1, mean = [], [], {}
    for label, name in spec:
        rs = g2runs(G, name)
        v = [np.mean([r['test'][k]['auroc'] for r in rs]) for k in G2KEYS]; mean[name] = v
        rows.append([label, f'{rs[0]["params"] / 1e6:.2f}', len(rs)] + [f'{x:.1f}' for k, x in zip(G2KEYS, v) if k != 'w500'])
    for label, spec_ in (('GenoDirNet, relative windows', GDN_P), ('GenoDirNet, absolute windows', GDN_A)):
        keys = ['extraction2_' + ('test' if k == 'full' else k) for k in G2KEYS]   # standard schedule, scored on these windows
        v = [100 * auroc_mean(extraction2_labels(data, k), get(R, spec_, key)) for k, key in zip(G2KEYS, keys)]
        mean[label] = v
        rows.append([label, f'{size_of(R, spec_)[0] / 1e6:.2f}', len(get(R, spec_, keys[0]))] + [f'{x:.1f}' for k, x in zip(G2KEYS, v) if k != 'w500'])
        for k, x in zip(G2KEYS, v):
            out.num(f'{"gdna" if spec_ is GDN_A else "gdn"}_x2_{k}', f'{x:.1f}', f'{label} on the windows of the second extraction, {k}')
    i2 = G2KEYS.index('w300_indel2')
    out.num('gdn2_minus_so_indel', f'{mean["gdn2"][i2] - mean["so"][i2]:.1f}', 'GenoDirNet-2 minus GenoDirNet-SO, 300-nt windows with two errors per 100 nt (points)')
    out.num('gdn2_minus_gdna_indel', f'{mean["gdn2"][i2] - mean["GenoDirNet, absolute windows"][i2]:.1f}',
            'GenoDirNet-2 minus GenoDirNet with absolute windows, 300-nt windows with two errors per 100 nt (points)')
    lr = [lm['test'][k]['genlr_auroc'] for k in G2KEYS]; mean['lr'] = lr
    rows.append(['Likelihood ratio alone', f'2 x {lm["n_params_each"] / 1e6:.2f}', '--'] + [f'{x:.1f}' for k, x in zip(G2KEYS, lr) if k != 'w500'])
    out.table('genodirnet2', rows, ['Model', 'Params (M)', 'Runs', 'Complete', '1,000', '300', '200', '100', '300 + indels'])
    for label, name in spec[:7] + [('+ masked denoising pretraining', 'masked')] + spec[7:]:
        for i, r in enumerate(g2runs(G, name), 1):
            s1.append([label, i, f'{r["val_auroc"]:.2f}'] + [f'{r["test"][k]["auroc"]:.2f}' for k in G2KEYS])
    s1.append(['Likelihood ratio alone (no training)', '--', '--'] + [f'{x:.2f}' for x in lr])
    out.table('S_genodirnet2_runs', s1, ['Model', 'Run', 'Validation', 'Complete', '1,000 nt', '500 nt', '300 nt', '200 nt', '100 nt', '300 nt + indels'])
    b = lm['bits_per_base']
    out.num('lm_bits_own', f'{b["coding_model_on_coding"]:.2f} and {b["noncoding_model_on_noncoding"]:.2f}', 'code length of each model on its own class (bits per nt)')
    out.num('lm_bits_other', f'{b["coding_model_on_noncoding"]:.2f} and {b["noncoding_model_on_coding"]:.2f}', 'code length of each model on the other class')
    so = g2runs(G, 'so')
    for name in ('llr', 'erroraware', 'pretrain', 'synthetic', 'gdn2', 'compact'):
        rs = g2runs(G, name)
        for k in G2KEYS:
            above = min(r['test'][k]['auroc'] for r in rs) > max(r['test'][k]['auroc'] for r in so)
            below = max(r['test'][k]['auroc'] for r in rs) < min(r['test'][k]['auroc'] for r in so)
            out.num(f'{name}_vs_so_{k}', f'{mean["so"][G2KEYS.index(k)]:.1f} to {mean[name][G2KEYS.index(k)]:.1f}', f'{name} against GenoDirNet-SO (AUROC, %)')
            out.num(f'{name}_vs_so_{k}_runs', 'above' if above else 'below' if below else 'overlap', 'both runs above / below both reference runs, or overlap')
    for name in ('so', 'deeper', f'lengthweighted_tau{tau}', 'gdn2', 'compact', 'pretrain', 'llr'):
        rs = g2runs(G, name)
        out.num(f'run_spread_{name}', f'{up(max(abs(rs[0]["test"][k]["auroc"] - rs[1]["test"][k]["auroc"]) for k in G2KEYS)):.1f}',
                f'{name}: largest difference between its two runs (points)')
    best_single = {k: max(mean[n][G2KEYS.index(k)] for _, n in spec[1:7]) for k in G2KEYS}
    out.num('gdn2_minus_best_single', ', '.join(f'{k}: {mean["gdn2"][G2KEYS.index(k)] - best_single[k]:+.1f}' for k in G2KEYS),
            'GenoDirNet-2 minus the best single addition (points)')
    g2 = g2runs(G, 'gdn2')
    lm_macs = 2 * lm['macs_per_nucleotide_each'] * 8192
    g2_macs = lm_macs + g2[0]['reader_mmac'] * 1e6
    out.num('lm_params', f'{lm["n_params_each"] / 1e6:.2f} million', 'parameters of each autoregressive model')
    out.num('gdn2_macs', f'{g2_macs / 1e9:.0f} billion', 'GenoDirNet-2, MACs for an 8,192-nt input')
    out.num('lm_macs_per_nt', f'{2 * lm["macs_per_nucleotide_each"] / 1e6:.1f} million', 'both autoregressive models, MACs per nucleotide')
    out.num('gdn2_macs_vs_nt', f'{g2_macs / nt_macs:.2f}', 'GenoDirNet-2 / Nucleotide Transformer lower bound')
    out.num('gdn2_macs_vs_gdn', f'{round(g2_macs / 1e9) * 1e9 / (round(gdn_macs / 1e6) * 1e6):.0f}', 'GenoDirNet-2 / GenoDirNet (from the printed figures)')
    s = json.load(open(os.path.join(G, 'search1', 'summary.json')))
    out.num('search1_validated', f'{s["n_validated"]} ({s["n_validated_policy"]} sampler, {s["n_validated_random"]} random)', 'first search: validated candidates')
    out.num('search1_selected_arm', s['selected'].get('arm', ''), 'first search: arm of the selected network')
    c = g2runs(G, 'compact')[0]
    out.num('compact_size', f'{c["params"] / 1e6:.2f} M, {c["reader_mmac"]:.0f} MMAC', 'sequence-ORF network with the network of the second search')


# --------------------------------------------------------------------------------- real assemblies
REAL = [('GenoDirNet', 'genodirnet'), ('GenoDirNet, absolute windows', 'genodirnet_absolute'), ('GenoDirNet-2', 'gdn2'), ('RNAsamba, retrained', 'rnasamba_retrained'),
        ('CPAT, retrained', 'cpat_retrained'), ('CPC2, retrained', 'cpc2_retrained'), ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial'),
        ('CPAT', 'cpat'), ('CPC2', 'cpc2'), ('LncFinder', 'lncfinder'), ('LncDC', 'lncdc'), ('TransDecoder', 'transdecoder')]
OURS = ('genodirnet', 'genodirnet_absolute', 'gdn2')
EXCLUDED_FROM_TESTS = ()
SUBSETS = [('all', 'All models'), ('trunc', 'Truncated'), ('part', 'Truncated, part of the coding sequence'),
           ('whole', 'Truncated, all of the coding sequence'), ('none', 'Truncated, none of the coding sequence'), ('complete', 'Complete models')]


def subsets(df):
    t = df[df.truncated == 1]
    return {'all': df, 'trunc': t, 'complete': df[df.truncated == 0], 'whole': t[(t.label == 0) | (t.orf_status == 'complete')],
            'part': t[(t.label == 0) | (t.orf_status == 'partial')], 'none': t[(t.label == 0) | (t.orf_status == 'absent')]}


def table_real(out, real, data):
    rows, ci, counts = [], [], {}
    sc = {ds: subsets(pd.read_csv(os.path.join(real, f'scores_{ds}.tsv'), sep='\t')) for ds in ('wtc11', 'chess')}
    cols = [(ds, s) for ds in ('wtc11', 'chess') for s in ('all', 'trunc', 'part', 'none', 'complete')]
    for label, m in REAL:
        cells, cic = [label], [label]
        for ds, s in cols:
            d = sc[ds][s][[m, 'label']].dropna(); y = d.label.values.astype(int); p = d[m].values.astype(float)
            cells.append(f'{100 * roc_auc_score(y, p):.1f}')
            a, lo, hi = auroc_ci(y, [p]); cic.append(f'{100 * a:.1f} ({100 * lo:.1f}-{100 * hi:.1f})')
            if m == 'genodirnet':
                counts[(ds, s)] = (int(y.sum()), int((y == 0).sum()))
        rows.append(cells); ci.append(cic)
    names = [f'{ds} {s}' for ds, s in cols]
    out.table('real', rows, ['Model'] + names)
    shift = []
    for ds in ('wtc11', 'chess'):
        df = pd.read_csv(os.path.join(real, f'scores_{ds}.tsv'), sep='\t')
        close = {line.strip() for line in open(os.path.join(data, 'real', f'{ds}_close_relatives.txt')) if line.strip()}
        kept = subsets(df[~df.id.isin(close)])
        out.num(f'{ds}_close_relatives', f'{len(close):,}', f'{ds}: models with >= 80 % identity to a training transcript (minimap2)')
        for _, m in REAL:
            for s in ('all', 'trunc', 'part', 'none', 'complete'):
                a, b = sc[ds][s][[m, 'label']].dropna(), kept[s][[m, 'label']].dropna()
                shift.append(abs(roc_auc_score(b.label, b[m]) - roc_auc_score(a.label, a[m])))
    out.num('real_homology_max_shift', f'{100 * max(shift):.1f}', 'largest change of an AUROC of the real-assembly table without these models (points)')
    released = [m for _, m in REAL[REAL.index(('RNAsamba', 'rnasamba')):]]
    for ds in ('wtc11', 'chess'):
        for s in ('part', 'trunc'):
            d = sc[ds][s]
            v = [100 * roc_auc_score(d.dropna(subset=[m]).label, d.dropna(subset=[m])[m]) for m in released]
            out.num(f'{ds}_{s}_released_range', f'{min(v):.1f} to {max(v):.1f}', f'{ds}, {s}: range of the released tools')
        out.num(f'{ds}_none_n', f'{int(((sc[ds]["none"].label == 1)).sum()):,}', f'{ds}: truncated coding models without annotated CDS')
    out.table('S_real_ci', ci, ['Model'] + names)
    out.num('real_counts', '; '.join(f'{ds} {s}: {a:,}/{b:,}' for (ds, s), (a, b) in counts.items()), 'coding/noncoding models per column of the real-assembly table')
    tests = []
    for ds in ('wtc11', 'chess'):
        for s, sname in SUBSETS:
            for g in OURS:
                for label, m in REAL:
                    if m in OURS or m in EXCLUDED_FROM_TESTS:
                        continue
                    d = sc[ds][s][[g, m, 'label']].dropna()
                    diff, p = delong_test(d.label.values, d[g].values, d[m].values)
                    tests.append([ds, sname, g, label, diff, p])
            for i, g in enumerate(OURS):                 # the networks of this study against each other
                for h in OURS[i + 1:]:
                    d = sc[ds][s][[g, h, 'label']].dropna()
                    diff, p = delong_test(d.label.values, d[g].values, d[h].values)
                    tests.append([ds, sname, g, dict((m, lab) for lab, m in REAL)[h], diff, p])
    q = bh([t[-1] for t in tests])
    out.num('real_tests', f'{len(tests)}', 'paired DeLong tests on the real assemblies (Benjamini-Hochberg adjusted together)')
    dl = []
    for (ds, sname, g, label, diff, p), qq in zip(tests, q):
        dl.append([ds, sname, {'genodirnet': 'GenoDirNet', 'genodirnet_absolute': 'GenoDirNet, absolute windows', 'gdn2': 'GenoDirNet-2'}[g], label,
                   f'{100 * diff:+.1f}', fmt_p(p), fmt_p(qq)])
    out.table('S_real_delong', dl, ['Set', 'Subset', 'Model', 'Compared with', 'Difference', 'P', 'q (BH)'])
    key = {(t[0], t[1], t[2], t[3]): (100 * t[4], qq) for t, qq in zip(tests, q)}
    compared = [lab for lab, m in REAL if m not in OURS and m not in EXCLUDED_FROM_TESTS]
    for sname, tag in (('Truncated, part of the coding sequence', 'part'), ('Truncated', 'trunc'), ('Complete models', 'complete')):
        cells = [(ds, g, lab, *key[(ds, sname, g, lab)]) for ds in ('wtc11', 'chess') for g in OURS for lab in compared]
        ahead = [c for c in cells if c[3] > 0 and c[4] < 0.05]
        out.num(f'real_{tag}_ahead', f'{len(ahead)} of {len(cells)}', f'{sname}: tests in which a network of this study was ahead with q < 0.05')
        names = {'genodirnet': 'GenoDirNet', 'genodirnet_absolute': 'GenoDirNet with absolute windows', 'gdn2': 'GenoDirNet-2'}
        sets = {'wtc11': 'WTC11', 'chess': 'CHESS'}
        nice = {'RNAsamba, retrained': 'retrained RNAsamba', 'CPAT, retrained': 'retrained CPAT', 'CPC2, retrained': 'retrained CPC2',
                'RNAsamba': 'released RNAsamba', 'RNAsamba, partial-length model': 'the partial-length RNAsamba model'}
        out.num(f'real_{tag}_other', '; '.join(f'{names[g]} against {nice.get(lab, lab)} on {sets[ds]} ({d:+.1f} points, $q$ = {qtext(qq)})'
                                               for ds, g, lab, d, qq in cells if not (d > 0 and qq < 0.05)) or 'none', f'{sname}: the other tests')
    # single comparisons quoted in the text: GenoDirNet against retrained RNAsamba, and absolute against relative windows
    short = {'Truncated, part of the coding sequence': 'part', 'Truncated': 'trunc', 'All models': 'all', 'Complete models': 'complete'}
    for ds in ('wtc11', 'chess'):
        for sname, tag in short.items():
            for lab, name in (('RNAsamba, retrained', 'rnasamba_retrained'), ('RNAsamba', 'rnasamba'), ('RNAsamba, partial-length model', 'rnasamba_partial')):
                d, qq = key[(ds, sname, 'genodirnet', lab)]
                out.num(f'real_{ds}_{tag}_gdn_minus_{name}', f'{d:+.1f} points ($q$ = {qtext(qq)})',
                        f'{ds}, {sname}: GenoDirNet minus {lab} (DeLong, BH-adjusted q)')
            d, qq = key[(ds, sname, 'genodirnet', 'GenoDirNet, absolute windows')]
            out.num(f'real_{ds}_{tag}_gdna_minus_gdn', f'{-d:+.1f} points ($q$ = {qtext(qq)})',
                    f'{ds}, {sname}: GenoDirNet with absolute windows minus GenoDirNet with relative windows (DeLong, BH-adjusted q)')
    whole = [100 * roc_auc_score(sc[ds]['whole'].dropna(subset=[m]).label, sc[ds]['whole'].dropna(subset=[m])[m]) for ds in ('wtc11', 'chess') for _, m in REAL]
    out.num('real_whole_min_auroc', f'{min(whole):.1f}', 'lowest AUROC of any method on truncated models that contain the complete CDS (both sets)')
    # complete models: largest gap between a network of this study and the best RNAsamba model
    gap = []
    for ds in ('wtc11', 'chess'):
        d = sc[ds]['complete']
        best = max(roc_auc_score(d.label, d[m]) for m in ('rnasamba', 'rnasamba_partial', 'rnasamba_retrained'))
        gap += [100 * (best - roc_auc_score(d.label, d[g])) for g in OURS]
    out.num('real_complete_gap_max', f'{max(gap):.1f}', 'complete models: largest gap of a network of this study behind the best RNAsamba model (points)')
    # the fusion against its sequence-ORF network on the real sets, relative and absolute windows (per-network scores of results/genodirnet)
    R1 = Runs(os.path.join(os.path.dirname(real), 'genodirnet'), os.path.join(data, 'extraction1'))
    fgain = {'rel': [], 'abs': []}
    for ds in ('wtc11', 'chess'):
        df = pd.read_csv(os.path.join(real, f'scores_{ds}.tsv'), sep='\t')
        for tag, spec in (('rel', GDN_P), ('abs', GDN_A)):
            so = R1.load(spec[0], ds); fu = R1.fused(spec, ds)
            if so is None or not fu or len(so) != len(df):
                continue
            fu = fu[0]
            for sub in ('trunc', 'part', 'complete'):
                idx = subsets(df)[sub].index.values; y = df.label.values[idx]
                fgain[tag].append(100 * (roc_auc_score(y, fu[idx]) - roc_auc_score(y, so[idx])))
    for tag, name in (('rel', 'relative'), ('abs', 'absolute')):
        if fgain[tag]:
            out.num(f'real_fusion_gain_{tag}_range', f'{min(fgain[tag]):.1f} to {max(fgain[tag]):.1f}',
                    f'real sets: GenoDirNet ({name} windows) minus its sequence-ORF network over the truncated, Part CDS and complete subsets (points)')
    # lengths of the real truncated models
    for ds in ('wtc11', 'chess'):
        df = pd.read_csv(os.path.join(real, f'scores_{ds}.tsv'), sep='\t')
        seqs = pd.read_csv(os.path.join(data, 'real', f'{ds}.tsv'), sep='\t', usecols=['id', 'sequence']).set_index('id')['sequence']
        t = df[df.truncated == 1]; ln = seqs.loc[t.id].str.len()
        out.num(f'{ds}_trunc_median_length', f'{int(ln.median()):,}', f'{ds}: median length of the truncated models (nt)')
        out.num(f'{ds}_trunc_le300', f'{100 * (ln <= 300).mean():.0f}', f'{ds}: truncated models of at most 300 nt (%)')
    for ds in ('wtc11', 'chess'):
        for g in OURS:
            for sname in ('Truncated, part of the coding sequence', 'Truncated', 'All models', 'Complete models'):
                v = [(lab, *key[(ds, sname, g, lab)]) for lab, m in REAL if m not in OURS and m not in EXCLUDED_FROM_TESTS]
                out.num(f'{ds}_{g}_{sname.split(",")[-1].strip().replace(" ", "_").lower()}', '; '.join(f'{lab} {d:+.1f} (q {fmt_p(qq)})' for lab, d, qq in v),
                        f'{ds}, {g}, {sname}: difference and BH-adjusted q against each compared method')
    for ds in ('wtc11', 'chess'):
        df = pd.read_csv(os.path.join(real, f'scores_{ds}.tsv'), sep='\t'); t = df[df.truncated == 1]
        out.num(f'{ds}_models', f'{len(df):,} models ({int(df.label.sum()):,} coding), {len(t):,} of them truncated ({int(t.label.sum()):,} coding)',
                f'{ds}: models')
        tc = t[t.label == 1]
        for k in ('complete', 'partial', 'absent'):
            out.num(f'{ds}_cds_{k}', f'{int((tc.orf_status == k).sum()):,}', f'{ds}: truncated coding models with {k} CDS')
        out.num(f'{ds}_truncated_share', f'{100 * df.truncated.mean():.0f}', f'{ds}: truncated models (%)')
        out.num(f'{ds}_missing_scores', ', '.join(f'{m}: {int(df[m].isna().sum())}' for _, m in REAL if df[m].isna().any()) or 'none', f'{ds}: models without a score')
        for sub in ('none', 'whole'):
            d = sc[ds][sub]
            out.num(f'{ds}_{sub}_auroc', ', '.join(f'{lab}: {100 * roc_auc_score(d.dropna(subset=[m]).label, d.dropna(subset=[m])[m]):.1f}' for lab, m in REAL),
                    f'{ds}: truncated models with {"no" if sub == "none" else "all of the"} CDS against truncated noncoding models')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results'); ap.add_argument('--data', default='data'); ap.add_argument('--out', default='paper')
    a = ap.parse_args()
    out = Out(a.out)
    R = Runs(os.path.join(a.results, 'genodirnet'), os.path.join(a.data, 'extraction1'))
    data_numbers(out, a.data, R)
    table_complete(out, R)
    table_fragments(out, R)
    table_draws(out, R)
    table_fat(out, R)
    table_controls(out, R, a.data)
    table_position(out, R, a.data)
    table_addon(out, R)
    errors(out, R, a.data)
    calibration(out, R)
    nt_macs, gdn_macs = compute(out, R)
    table_external(out, R, a.data)
    table_indels(out, R)
    table_genodirnet2(out, os.path.join(a.results, 'genodirnet2'), gdn_macs, nt_macs, R, a.data)
    table_real(out, os.path.join(a.results, 'real'), a.data)
    out.close()


if __name__ == '__main__':
    main()
