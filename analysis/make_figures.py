"""The data figures of the article, drawn from the run outputs.

    python analysis/make_figures.py --results results --data data --out paper/figures

Writes fig_data, fig_encoding, fig_signal, fig_roc, fig_fragments, fig_errors, fig_reliability, fig_gdn2 and fig_real
as PDF and PNG files.
"""
import argparse
import itertools
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analysis.common import GDN, GDN_P, GDN_P0, LENGTH_CLASSES, LNC_CLASSES, Runs, frame_signal, get, lncrna_class
from analysis.make_tables import REAL, g2runs, subsets
from genodirnet.data import read_meta, read_seqs
from genodirnet.fragments import WINDOW_LENGTHS
from genodirnet.frcgr import frame_counts
from genodirnet.metrics import auroc_ci, sensitivity_at_fpr

BLUE, ORANGE, AQUA, YELLOW, VIOLET, MAGENTA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e87ba4', '#c0392b'
G1, G2, G3, G4 = '#2f2f2d', '#5c5b57', '#8a8984', '#b3b2ab'
NAVY = '#173f7a'
INK, MUTED, GRID = '#1b1b1a', '#52514e', '#e4e3de'
DOT, DASH = (0, (1, 1.4)), (0, (3, 1.6))
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.labelsize': 7.4, 'axes.titlesize': 7.6, 'legend.fontsize': 6.4,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})


def panel(ax, s, dx=-0.13, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')


def tidy(ax, axis='y'):
    for a in (('x', 'y') if axis == 'both' else (axis,)):
        getattr(ax, f'{a}axis').grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


class Figures:
    def __init__(self, results, data, out):
        self.R = Runs(os.path.join(results, 'genodirnet'), os.path.join(data, 'extraction1'))
        self.G, self.real, self.data, self.out = os.path.join(results, 'genodirnet2'), os.path.join(results, 'real'), data, out
        os.makedirs(out, exist_ok=True)

    def save(self, fig, name):
        fig.savefig(os.path.join(self.out, name + '.pdf'))
        fig.savefig(os.path.join(self.out, name + '.png'), dpi=220)
        plt.close(fig)

    def auroc(self, spec, split='test'):
        runs = get(self.R, spec, split)
        return 100 * float(np.mean([roc_auc_score(self.R.labels(split), p) for p in runs]))

    # ------------------------------------------------------------------------------- data set
    def fig_data(self):
        e1 = os.path.join(self.data, 'extraction1')
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={'wspace': 0.32})
        edges = np.logspace(np.log10(200), np.log10(20000), 36)
        for split, col in (('train', G2), ('test', BLUE)):
            rows = read_meta(os.path.join(e1, f'{split}_meta.csv'))
            for lab, ls in ((1, '-'), (0, DASH)):
                ln = np.array([int(r['length']) for r in rows if int(r['label']) == lab])
                axes[0].hist(ln, bins=edges, histtype='step', color=col, lw=1.0, ls=ls, weights=np.full(len(ln), 100.0 / len(ln)),
                             label=f"{'Training' if split == 'train' else 'Test'}, {'protein-coding' if lab else 'lncRNA'}")
        axes[0].set_xscale('log'); axes[0].set_xlabel('Transcript length (nucleotides)'); axes[0].set_ylabel('Transcripts per bin (%)')
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        tidy(axes[0]); panel(axes[0], 'A', -0.16)
        meta = read_meta(os.path.join(e1, 'test_meta.csv'))
        chroms = sorted({m['chrom'] for m in meta}, key=lambda c: (len(c), c)); x = np.arange(len(chroms))
        cod = [sum(1 for m in meta if m['chrom'] == c and m['label'] == '1') for c in chroms]
        non = [sum(1 for m in meta if m['chrom'] == c and m['label'] == '0') for c in chroms]
        axes[1].bar(x - 0.2, cod, 0.4, color=BLUE, label='Protein-coding'); axes[1].bar(x + 0.2, non, 0.4, color=G2, label='lncRNA')
        axes[1].set_xticks(x); axes[1].set_xticklabels([c.replace('chr', '') for c in chroms]); axes[1].set_xlabel('Test chromosome')
        axes[1].set_ylabel('Test transcripts'); axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        tidy(axes[1]); panel(axes[1], 'B', -0.16)
        self.save(fig, 'fig_data')

    # ------------------------------------------------------------------------------- FR-CGR
    def fig_encoding(self):
        e1 = os.path.join(self.data, 'extraction1')
        meta = read_meta(os.path.join(e1, 'test_meta.csv')); y, seqs = read_seqs(os.path.join(e1, 'test_seqs.txt'))
        cds = {r['transcript']: int(r['cds_length']) for r in read_meta(os.path.join(e1, 'test_cds.csv'))}
        idx = {m['transcript']: i for i, m in enumerate(meta)}
        fr = [RED, BLUE, AQUA]
        fig = plt.figure(figsize=(7.0, 4.9))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.38, wspace=0.28, left=0.03, right=0.99, top=0.94, bottom=0.03)

        def cell(word):
            r = sum(((1 if c in 'TG' else 0) << j) for j, c in enumerate(word)); c_ = sum(((1 if c in 'CG' else 0) << j) for j, c in enumerate(word))
            return r, c_

        def grid_panel(ax, k, title, highlight=()):
            n = 1 << k
            ax.set_xlim(-0.7, n + 0.7); ax.set_ylim(n + 0.7, -0.7); ax.set_aspect('equal'); ax.axis('off')
            for w in itertools.product('ACGT', repeat=k):
                word = ''.join(w); r, c = cell(word); fc = 'white'
                for suf, col in highlight:
                    if word.endswith(suf):
                        fc = col
                ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec=G4, lw=0.4))
                if k == 2:
                    ax.text(c + 0.5, r + 0.5, word, ha='center', va='center', fontsize=6.8, family='monospace')
            for xx, yy, t, ha, va in ((-0.15, -0.15, 'A', 'right', 'bottom'), (n + 0.15, -0.15, 'C', 'left', 'bottom'),
                                      (-0.15, n + 0.15, 'T', 'right', 'top'), (n + 0.15, n + 0.15, 'G', 'left', 'top')):
                ax.text(xx, yy, t, ha=ha, va=va, fontsize=7.6, weight='bold', color=ORANGE)
            ax.set_title(title, fontsize=7.2, loc='left', pad=9)

        grid_panel(fig.add_subplot(gs[0, 0]), 2, 'A   Cell of each word, k = 2')
        grid_panel(fig.add_subplot(gs[0, 1]), 3, 'B   k = 3: words ending in G (grey)\n      and in CG (orange)', ((('G', '#dcdbd6'), ('CG', '#f6b998'))))
        grid_panel(fig.add_subplot(gs[0, 2]), 4, 'C   k = 4: the same suffix blocks', ((('G', '#dcdbd6'), ('CG', '#f6b998'))))
        ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        ax.set_title('D   Codon-phase channels', fontsize=7.2, loc='left', pad=4)
        word = 'ATGGCTAAGCGT'
        for i, c in enumerate(word):
            ax.add_patch(Rectangle((0.5 + i * 0.75, 7.4), 0.75, 1.0, fc='white', ec=fr[i % 3], lw=1.0))
            ax.text(0.875 + i * 0.75, 7.9, c, ha='center', va='center', fontsize=6.2, family='monospace')
            ax.text(0.875 + i * 0.75, 6.85, str(i % 3), ha='center', va='center', fontsize=5.8, color=fr[i % 3])
        ax.text(0.45, 6.85, '(t − 1) mod 3', ha='right', va='center', fontsize=5.8, color=MUTED)
        for f in range(3):
            ax.add_patch(Rectangle((0.5 + f * 0.75, 5.55 - f * 0.85), 5 * 0.75, 0.5, fc=fr[f], ec='none'))
            ax.text(0.5 + f * 0.75 + 5 * 0.75 + 0.25, 5.8 - f * 0.85, f'word counted in channel f = {f}', fontsize=5.8, va='center', color=MUTED)
        ax.text(0.3, 1.2, 'Inside a coding sequence the three channels\ndiffer; in untranslated regions and lncRNAs\nthey sample one word distribution.',
                fontsize=5.9, color=MUTED, linespacing=1.25)
        coding = max((t for t in cds if t in idx and 3000 <= len(seqs[idx[t]]) <= 4000), key=lambda t: cds[t] / len(seqs[idx[t]]))
        lnc = next(m['transcript'] for m in meta if m['label'] == '0' and 1000 <= int(m['length']) <= 1500)
        sub = gs[1, 1:].subgridspec(2, 3, wspace=0.06, hspace=0.2)
        for row, (t, name) in enumerate(((coding, 'protein-coding'), (lnc, 'lncRNA'))):
            c = np.log1p(frame_counts(seqs[idx[t]], 5).reshape(3, 32, 32).astype(float)); c = c / c.max()
            for f in range(3):
                a = fig.add_subplot(sub[row, f]); a.imshow(1 - c[f], cmap='gray', vmin=0, vmax=1, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color(fr[f]); sp.set_linewidth(1.1)
                if row == 0:
                    a.set_title(f'f = {f}', fontsize=6.6, color=MUTED, pad=2)
                if f == 0:
                    a.set_ylabel(f'{name}\n{len(seqs[idx[t]]):,} nt', fontsize=6.4)
        fig.text(0.355, 0.515, 'E   FR-CGR (k = 5) of two test transcripts; log counts, darker = more frequent', fontsize=7.2)
        self.save(fig, 'fig_encoding')
        return coding, lnc

    # ------------------------------------------------------------------------------- frame signal
    def fig_signal(self):
        _, seqs = read_seqs(os.path.join(self.data, 'extraction1', 'test_seqs.txt'))
        sig = frame_signal(seqs); lens = np.array([len(s) for s in seqs]); y = self.R.y
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={'wspace': 0.3})
        bins = np.linspace(np.percentile(sig, 0.5), np.percentile(sig, 99.5), 50)
        for lab, col, nm in ((0, G2, 'lncRNA'), (1, BLUE, 'Protein-coding')):
            axes[0].hist(sig[y == lab], bins=bins, histtype='step', color=col, lw=1.1, density=True, label=nm)
            axes[1].scatter(lens[y == lab], sig[y == lab], s=1.4, color=col, alpha=0.3, linewidths=0, label=nm, rasterized=True)
        axes[0].set_xlabel('Frame signal (nats)'); axes[0].set_ylabel('Density'); axes[0].legend(loc='upper right', frameon=False)
        axes[1].set_xscale('log'); axes[1].set_xlabel('Transcript length (nucleotides)'); axes[1].set_ylabel('Frame signal (nats)')
        axes[1].axhline(0, color=G4, lw=0.6); axes[1].set_ylim(np.percentile(sig, 0.2), np.percentile(sig, 99.8))
        leg = axes[1].legend(loc='upper right', frameon=False, markerscale=4)
        for h in leg.legend_handles:
            h.set_alpha(1)
        tidy(axes[0]); tidy(axes[1]); panel(axes[0], 'A', -0.15); panel(axes[1], 'B', -0.15)
        self.save(fig, 'fig_signal')
        return 100 * roc_auc_score(y, sig)

    # ------------------------------------------------------------------------------- ROC
    def fig_roc(self):
        R = self.R
        models = [('GenoDirNet', GDN, BLUE, '-', 1.8), ('Sequence–ORF network', 'seq_orf_net', ORANGE, '-', 1.1),
                  ('Sequence network', 'seq_net', AQUA, '-', 1.1), ('FR-CGR network', 'frcgr_k5', YELLOW, '-', 1.1),
                  ('RNAsamba, retrained', 'rnasamba_retrained', VIOLET, DASH, 1.0), ('ORF-feature LR', 'orf_lr', G4, (0, (5, 1.5, 1, 1.5)), 1.0),
                  ('Periodic Markov chain', 'markov5', G3, DOT, 1.2), ('RNAsamba', 'rnasamba', G1, (0, (1.2, 1.6)), 1.2),
                  ('CPAT', 'cpat', G2, (0, (6, 1.5, 1, 1.5)), 1.0), ('CPC2', 'cpc2', G3, (0, (3.5, 1.5)), 1.0)]
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45))
        for label, spec, col, ls, lw in models:
            p = get(R, spec)[0]
            fpr, tpr, _ = roc_curve(R.y, p); pr, rc, _ = precision_recall_curve(R.y, p)
            lab = f'{label} ({self.auroc(spec):.1f})'
            axs[0].plot(fpr, tpr, color=col, ls=ls, lw=lw, label=lab); axs[1].plot(fpr, tpr, color=col, ls=ls, lw=lw)
            axs[2].plot(rc, pr, color=col, ls=ls, lw=lw)
        axs[0].plot([0, 1], [0, 1], color=GRID, lw=0.7, zorder=0)
        axs[0].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
        axs[1].set(xlim=(0, 0.2), ylim=(0.5, 1.0), xlabel='False positive rate', ylabel='True positive rate')
        axs[2].set(xlim=(0, 1), ylim=(0.4, 1.01), xlabel='Recall', ylabel='Precision')
        for ax, s, t in zip(axs, 'ABC', ['ROC curves', 'False positive rate below 0.2', 'Precision and recall']):
            tidy(ax, 'both'); panel(ax, s, -0.22); ax.set_title(t, loc='left', fontsize=7.2, pad=3)
        fig.legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.6, columnspacing=1.4)
        fig.subplots_adjust(wspace=0.4)
        self.save(fig, 'fig_roc')

    # ------------------------------------------------------------------------------- fragments
    def fig_fragments(self, gains):
        R = self.R
        keys = [f'test_w{L}' for L in WINDOW_LENGTHS] + ['test']
        xl = [f'{L:,}' for L in WINDOW_LENGTHS] + ['Complete']
        models = [('GenoDirNet', GDN_P, BLUE, '-', 'o', True, 1.6), ('GenoDirNet, no FAT', GDN_P0, BLUE, DASH, 'o', False, 1.1),
                  ('RNAsamba, retrained', 'rnasamba_retrained', ORANGE, '-', 's', True, 1.2),
                  ('RNAsamba, retrained, no FAT', 'rnasamba_retrained_nofat', ORANGE, DASH, 's', False, 1.0),
                  ('CPAT, retrained', 'cpat_retrained', AQUA, '-', 'P', True, 1.1), ('CPAT, retrained, no FAT', 'cpat_retrained_nofat', AQUA, DASH, 'P', False, 1.0),
                  ('Nucleotide Transformer', 'nt', VIOLET, '-', 'D', True, 1.2), ('Nucleotide Transformer, no FAT', 'nt_nofat', VIOLET, DASH, 'D', False, 1.0),
                  ('RNAsamba, released', 'rnasamba', G1, DOT, '^', False, 1.1), ('CPAT, released', 'cpat', G2, DOT, 'v', False, 1.0),
                  ('CPC2, released', 'cpc2', G3, DOT, 'x', False, 1.0)]
        fig = plt.figure(figsize=(7.0, 2.9))
        axA = fig.add_axes([0.065, 0.36, 0.25, 0.56]); axB = fig.add_axes([0.39, 0.36, 0.25, 0.56]); axC = fig.add_axes([0.82, 0.36, 0.17, 0.56])
        x = np.arange(len(keys))
        for label, spec, col, ls, mk, filled, lw in models:
            au = [self.auroc(spec, k) for k in keys]
            se = [100 * np.mean([sensitivity_at_fpr(R.y, p) for p in get(R, spec, k)]) for k in keys]
            for ax, v in ((axA, au), (axB, se)):
                ax.plot(x, v, color=col, ls=ls, lw=lw, marker=mk, ms=3.0, mfc=col if filled else 'white', mec=col, mew=0.8,
                        label=label if ax is axA else None)
        for ax, yl, lo in ((axA, 'AUROC (%)', 48), (axB, 'Sensitivity at 5% FPR (%)', 0)):
            ax.set_xticks(x); ax.set_xticklabels(xl, rotation=35, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Window length (nucleotides)'); ax.set_ylabel(yl); ax.set_ylim(lo, 100); tidy(ax)
        panel(axA, 'A', -0.24); panel(axB, 'B', -0.24)
        fig.legend(loc='upper left', ncol=4, frameon=False, bbox_to_anchor=(0.03, 0.075), handlelength=2.6, columnspacing=1.2)
        yy = np.arange(len(gains))[::-1]; h = 0.38
        axC.barh(yy + h / 2, [g[1] for g in gains], height=h, color=BLUE, label='relative windows')
        axC.barh([y - h / 2 for y, g in zip(yy, gains) if g[2] is not None], [g[2] for g in gains if g[2] is not None], height=h, color=ORANGE,
                 label='absolute windows')
        for yv, g in zip(yy, gains):                              # one label per row, right of both bars
            label = f'{g[1]:.1f}' if g[2] is None else f'{g[1]:.1f} / {g[2]:.1f}'
            axC.text(max(g[1], g[2] or 0, 0) + 0.4, yv, label, va='center', fontsize=5.4, color=MUTED)
        axC.axvline(0, color=INK, lw=0.6)
        axC.set_yticks(yy); axC.set_yticklabels([g[0] for g in gains], fontsize=6.2)
        top = max(max(g[1], g[2] or 0) for g in gains)
        axC.set_xlabel('AUROC gain on 300-nt windows (points)'); axC.set_xlim(-1.5, top + 9); tidy(axC, 'x')
        axC.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False, handlelength=1.0, fontsize=6.0)
        panel(axC, 'C', -0.95)
        self.save(fig, 'fig_fragments')

    # ------------------------------------------------------------------------------- errors
    def fig_errors(self):
        R = self.R
        meta = read_meta(os.path.join(self.data, 'extraction1', 'test_meta.csv'))
        lens = np.array([int(m['length']) for m in meta]); cls = np.array([lncrna_class(m['name']) for m in meta])
        fig = plt.figure(figsize=(7.0, 2.5))
        axA = fig.add_axes([0.07, 0.25, 0.27, 0.64]); axB = fig.add_axes([0.44, 0.25, 0.55, 0.64])
        for label, spec, col, ls, mk in (('GenoDirNet', GDN, BLUE, '-', 'o'), ('Sequence network', 'seq_net', AQUA, '-', 's'),
                                         ('RNAsamba', 'rnasamba', G1, DOT, '^'), ('RNAsamba, partial-length', 'rnasamba_partial', G2, DOT, 'v'),
                                         ('CPC2', 'cpc2', G3, DOT, 'x')):
            v = [100 * np.mean([roc_auc_score(R.y[(lens >= lo) & (lens < hi)], p[(lens >= lo) & (lens < hi)]) for p in get(R, spec)]) for lo, hi, _ in LENGTH_CLASSES]
            axA.plot(range(len(v)), v, color=col, ls=ls, lw=1.3 if col in (BLUE, AQUA) else 1.0, marker=mk, ms=3.0, label=label)
        axA.set_xticks(range(len(LENGTH_CLASSES))); axA.set_xticklabels([n.replace('-', '–') for _, _, n in LENGTH_CLASSES], fontsize=6.2)
        axA.set_xlabel('Transcript length (nucleotides)'); axA.set_ylabel('AUROC (%)'); axA.set_ylim(80, 100)
        tidy(axA); panel(axA, 'A', -0.25); axA.legend(loc='lower right', frameon=False, fontsize=5.9, handlelength=2.2)
        names = {'unnamed': 'Unnamed', 'intergenic': 'Intergenic\n(LINC, LOC)', 'antisense': 'Antisense\n(AS)', 'divergent': 'Divergent\n(DT)',
                 'other named': 'Other\nnamed'}
        bars = [('GenoDirNet', GDN, BLUE), ('Sequence network', 'seq_net', AQUA), ('FR-CGR network', 'frcgr_k5', YELLOW),
                ('ORF-feature LR', 'orf_lr', G4), ('CPC2', 'cpc2', G3), ('RNAsamba', 'rnasamba', G1)]
        n = len(bars); w = 0.8 / n; xx = np.arange(len(LNC_CLASSES))
        for i, (label, spec, col) in enumerate(bars):
            v = [100 * np.mean([(p[(R.y == 0) & (cls == c)] > np.quantile(p[R.y == 1], 0.05)).mean() for p in get(R, spec)]) for c in LNC_CLASSES]
            axB.bar(xx + (i - (n - 1) / 2) * w, v, width=w * 0.92, color=col, label=label)
        axB.set_xticks(xx); axB.set_xticklabels([names[c] for c in LNC_CLASSES], fontsize=6.4)
        axB.set_ylabel('lncRNAs called coding at\n95% sensitivity (%)'); tidy(axB); panel(axB, 'B', -0.09)
        axB.legend(loc='upper left', frameon=False, ncol=2, fontsize=6.0, handlelength=1.0)
        self.save(fig, 'fig_errors')

    # ------------------------------------------------------------------------------- calibration
    def fig_reliability(self):
        from genodirnet.metrics import ece, reliability
        R = self.R
        fig, ax = plt.subplots(figsize=(4.5, 2.6))
        ax.plot([0, 1], [0, 1], color=G4, lw=0.9, zorder=0)
        for label, spec, col, ls, mk, lw in (('GenoDirNet', GDN, BLUE, '-', 'o', 1.6), ('Sequence–ORF network', 'seq_orf_net', ORANGE, '-', 's', 1.1),
                                             ('Sequence network', 'seq_net', AQUA, '-', '^', 1.1), ('FR-CGR network', 'frcgr_k5', YELLOW, '-', 'D', 1.1),
                                             ('CPC2', 'cpc2', G3, DASH, 'v', 1.0), ('RNAsamba', 'rnasamba', G1, DOT, 'x', 1.0)):
            xy = reliability(R.y, get(R, spec)[0])
            e = 100 * np.mean([ece(R.y, q) for q in get(R, spec)])
            ax.plot(xy[:, 0], xy[:, 1], color=col, ls=ls, lw=lw, marker=mk, ms=3, label=f'{label} (ECE {e:.1f}%)')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted probability of coding', ylabel='Observed fraction coding')
        tidy(ax, 'both'); ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.4)
        self.save(fig, 'fig_reliability')

    # ------------------------------------------------------------------------------- GenoDirNet-2
    def fig_gdn2(self):
        keys = ['w100', 'w200', 'w300', 'w500', 'w1000', 'full', 'w300_indel2']
        groups = ['100', '200', '300', '500', '1,000', 'Complete', '300 +\nindels']
        runs = lambda name: [{k: r['test'][k]['auroc'] for k in keys} for r in g2runs(self.G, name)]
        ref = runs('so'); refm = {k: np.mean([r[k] for r in ref]) for k in keys}
        ser = [('+ likelihood-ratio features', runs('llr'), AQUA), ('+ error-aware augmentation', runs('erroraware'), YELLOW),
               ('GenoDirNet-2 (all additions)', runs('gdn2'), VIOLET)]
        fig, ax = plt.subplots(figsize=(4.7, 2.4))
        x = np.arange(len(keys)); w = 0.26
        for i, (name, rr, col) in enumerate(ser):
            xi = x + (i - 1) * w
            ax.bar(xi, [np.mean([r[k] for r in rr]) - refm[k] for k in keys], width=w - 0.04, color=col, label=name, zorder=2)
            for xx_, k in zip(xi, keys):
                ax.plot([xx_] * len(rr), [r[k] - refm[k] for r in rr], 'o', ms=1.9, mfc='white', mec=INK, mew=0.5, zorder=3)
        ax.axhline(0, color=INK, lw=0.6, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=6.4)
        ax.set_xlabel('Test input (window length in nucleotides)'); ax.set_ylabel('AUROC change from\nGenoDirNet-SO (points)')
        tidy(ax); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.1, handlelength=1.0)
        self.save(fig, 'fig_gdn2')

    # ------------------------------------------------------------------------------- real assemblies
    def fig_real(self):
        cols = {'genodirnet': BLUE, 'genodirnet_absolute': NAVY, 'gdn2': VIOLET,
                'rnasamba_retrained': AQUA}
        short = {'GenoDirNet, absolute windows': 'GenoDirNet (absolute)', 'RNAsamba, retrained': 'RNAsamba (retrained)',
                 'RNAsamba, partial-length model': 'RNAsamba (partial)'}
        models = [(short.get(lab, lab), m) for lab, m in REAL if m not in ('cpat_retrained', 'cpc2_retrained')]
        sc = {ds: subsets(pd.read_csv(os.path.join(self.real, f'scores_{ds}.tsv'), sep='\t')) for ds in ('wtc11', 'chess')}

        def au(ds, sub, m, ci=False):
            d = sc[ds][sub][[m, 'label']].dropna()
            return auroc_ci(d.label.values.astype(int), [d[m].values.astype(float)]) if ci else 100 * roc_auc_score(d.label, d[m])
        fig = plt.figure(figsize=(7.0, 3.2))
        ax1 = fig.add_axes([0.14, 0.15, 0.165, 0.70]); ax2 = fig.add_axes([0.325, 0.15, 0.165, 0.70])
        ax3 = fig.add_axes([0.665, 0.15, 0.155, 0.70]); ax4 = fig.add_axes([0.835, 0.15, 0.155, 0.70])
        n = len(models)
        for ax, ds, title in ((ax1, 'wtc11', 'WTC11, long reads'), (ax2, 'chess', 'CHESS 3, short reads')):
            for r, (lab, m) in enumerate(models):
                yv = n - 1 - r; col = cols.get(m, G2)
                for sub, filled in (('trunc', True), ('complete', False)):
                    a, lo, hi = au(ds, sub, m, ci=True)
                    ax.plot([100 * lo, 100 * hi], [yv, yv], color=col, lw=0.9)
                    ax.plot(100 * a, yv, 'o', ms=3.3, color=col, mfc=col if filled else 'white', mew=0.9, zorder=3)
            ax.set_yticks(range(n)); ax.set_yticklabels([lab for lab, _ in models][::-1] if ax is ax1 else [], fontsize=6.3)
            ax.set_xlim(72, 100); ax.set_xticks([75, 85, 95]); ax.set_ylim(-0.7, n - 0.3)
            ax.set_title(title, fontsize=6.9, pad=2); ax.set_xlabel('AUROC (%)'); tidy(ax, 'x')
        ax1.plot([], [], 'o', color=MUTED, ms=3.3, label='truncated models'); ax1.plot([], [], 'o', color=MUTED, mfc='white', ms=3.3, label='complete models')
        ax1.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.07), ncol=2, frameon=False, fontsize=6.1, handlelength=0.8)
        panel(ax1, 'A', -1.05)
        marks = [('whole', 'all of it', 'o', True), ('part', 'part of it', 's', True), ('none', 'none of it', 'D', False)]
        for ax, ds, title in ((ax3, 'wtc11', 'WTC11'), (ax4, 'chess', 'CHESS 3')):
            for r, (lab, m) in enumerate(models):
                yv = n - 1 - r; col = cols.get(m, G2)
                for sub, slab, mk, filled in marks:
                    ax.plot(au(ds, sub, m), yv, mk, ms=3.3, color=col, mfc=col if filled else 'white', mew=0.9,
                            label=slab if (r == 0 and ax is ax3) else None)
            ax.axvline(50, color=G4, lw=0.8, zorder=0)
            ax.set_yticks(range(n)); ax.set_yticklabels([lab for lab, _ in models][::-1] if ax is ax3 else [], fontsize=6.3)
            ax.set_xlim(35, 100); ax.set_xticks([50, 75, 100]); ax.set_ylim(-0.7, n - 0.3)
            ax.set_title(title, fontsize=6.9, pad=2); ax.set_xlabel('AUROC (%)'); tidy(ax, 'x')
        ax3.legend(title='Annotated coding sequence in the model', title_fontsize=6.1, loc='lower left', bbox_to_anchor=(-0.1, 1.07),
                   ncol=3, frameon=False, fontsize=6.1, handlelength=0.8, columnspacing=0.8)
        panel(ax3, 'B', -1.2)
        self.save(fig, 'fig_real')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results'); ap.add_argument('--data', default='data'); ap.add_argument('--out', default='paper/figures')
    ap.add_argument('--tables', default='paper/tables', help='folder of make_tables.py (gains of fragment-aware training)')
    a = ap.parse_args()
    F = Figures(a.results, a.data, a.out)
    gains = pd.read_csv(os.path.join(a.tables, 'fig_fat_gains.tsv'), sep='\t')
    short = {'Word-frequency LR (k = 1 to 6)': 'Word-frequency LR', 'Frame-resolved MLP (k = 5)': 'Frame-resolved MLP', 'FR-CGR network (k = 5)': 'FR-CGR network',
             'Sequence-ORF network': 'Sequence–ORF network', 'RNAsamba, retrained': 'RNAsamba (retrained)', 'CPAT, retrained': 'CPAT (retrained)',
             'CPC2, retrained': 'CPC2 (retrained)'}
    F.fig_data()
    print('encoding transcripts:', F.fig_encoding())
    print('frame signal AUROC:', F.fig_signal())
    F.fig_roc()
    F.fig_fragments([(short.get(r.Model, r.Model), r._2, None if pd.isna(r._3) else r._3) for r in gains.itertuples()])
    F.fig_errors()
    F.fig_reliability()
    F.fig_gdn2()
    F.fig_real()
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
