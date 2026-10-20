# This file reimplements the inference of the rnasamba package (https://github.com/apcamargo/RNAsamba, GPL-3.0)
# and is distributed under the GNU General Public License version 3.
"""Inference of RNAsamba 0.1.6 (Camargo et al., NAR Genomics and Bioinformatics 2020) in NumPy.

Reproduces the classifier of the rnasamba package (rnasamba/core/model.py, miniigloo.py, inputs.py and
sequences.py of version 0.1.6) from a weight file written by the package, so that the released and the
retrained RNAsamba models can be applied to many sets of sequences without TensorFlow. The weight files are read
with pyfive (a pure-Python HDF5 reader). On the complete test transcripts and on the 300-nt windows, the coding
scores agree with those of rnasamba 0.1.6 (TensorFlow 1.14) to the five decimals the package writes, for the
released full-length and partial-length models and for both retrained models.
"""
import re
from collections import Counter
from itertools import product

import numpy as np

MAXLEN, PROT_MAXLEN = 3000, 1000
NUC_TOKENS = {'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
AA_INDEX = {'A': 4, 'C': 18, 'D': 12, 'E': 3, 'F': 14, 'G': 5, 'H': 16, 'I': 13, 'K': 9, 'L': 1, 'M': 19, 'N': 15,
            'P': 6, 'Q': 11, 'R': 8, 'S': 2, 'T': 10, 'V': 7, 'W': 20, 'X': 21, 'Y': 17}
_BASES = 'TCAG'
_AA = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
CODON = {a + b + c: _AA[16 * i + 4 * j + k] for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}


def _translate_codon(codon):
    """Standard code; an ambiguous codon translates to the amino acid shared by all its readings (Biopython rule)."""
    if codon in CODON:
        return CODON[codon]
    opts = [''.join(p) for p in product(*[('ACGT' if c == 'N' else c) for c in codon])]
    aas = {CODON.get(o, 'X') for o in opts}
    if len(aas) == 1:
        return aas.pop()
    return 'X'


def longest_orf(seq):
    """(protein length, start, protein) of the longest ORF from any ATG, translated codon by codon up to the first stop
    codon or the last complete codon."""
    longest = (0, 0, '')
    for m in re.finditer('ATG', seq):
        s = seq[m.start():]
        prot = []
        for i in range(0, len(s) - 2, 3):
            aa = _translate_codon(s[i:i + 3])
            if aa == '*':
                break
            prot.append(aa)
        if len(prot) > longest[0]:
            longest = (len(prot), m.start(), ''.join(prot))
    return longest


def _pad(rows, maxlen):
    """keras pad_sequences(padding='post', truncating='pre'): keep the last maxlen values, pad with zeros at the end."""
    out = np.zeros((len(rows), maxlen), dtype=np.int64)
    for i, r in enumerate(rows):
        r = r[-maxlen:] if len(r) > maxlen else r
        out[i, :len(r)] = r
    return out


_KMERS = [[''.join(p) for p in product('ATCG', repeat=k)] for k in (2, 3, 4)]


def inputs(seqs):
    """The five network inputs of RNAsamba for a list of nucleotide sequences."""
    seqs = [s.upper().replace('U', 'T') for s in seqs]
    tokens = [[NUC_TOKENS[c] for c in s if c in NUC_TOKENS] or [0] for s in seqs]
    orfs = [longest_orf(s) for s in seqs]
    kmer = np.zeros((len(seqs), 336), dtype=np.float32)
    for i, s in enumerate(seqs):
        col = 0
        for k, words in zip((2, 3, 4), _KMERS):
            tot = len(s) - k + 1
            cnt = Counter(s[j:j + k] for j in range(max(tot, 0)))
            for w in words:
                kmer[i, col] = cnt.get(w, 0) / tot if tot > 0 else 0.0
                col += 1
    orf_frac = np.zeros(len(seqs), dtype=np.float32)
    for i, (n_aa, start, _) in enumerate(orfs):
        if n_aa > 1:
            v = np.zeros(MAXLEN); v[start:start + n_aa * 3] = 1; orf_frac[i] = v.sum() / MAXLEN
    prot = _pad([[AA_INDEX[a] for a in p] for _, _, p in orfs], PROT_MAXLEN)
    aa = np.zeros((len(seqs), 21), dtype=np.float32)
    for i, (_, _, p) in enumerate(orfs):
        c = Counter(AA_INDEX[a] for a in p); n = max(len(p), 1)
        aa[i] = [c.get(j, 0) / n for j in range(1, 22)]
    return _pad(tokens, MAXLEN), orf_frac, kmer, prot, aa


def _weights(path):
    import pyfive
    f = pyfive.File(path)
    w = {}
    for layer in f.attrs['layer_names']:
        layer = layer.decode() if isinstance(layer, bytes) else layer
        g = f[layer]
        names = g.attrs.get('weight_names', [])
        for n in names:
            n = n.decode() if isinstance(n, bytes) else n
            w[(layer, n.split('/')[-1].split(':')[0].lower())] = np.asarray(g[n][()])
    return w


class RNAsamba:
    """RNAsamba classifier with the weights of one weight file (released or retrained)."""

    def __init__(self, path):
        w = _weights(path)
        layers = sorted({k[0] for k in w})
        num = lambda name: int(re.search(r'_(\d+)$', name).group(1)) if re.search(r'_(\d+)$', name) else 0
        kind = lambda prefix: sorted([n for n in layers if n.startswith(prefix)], key=num)
        conv, bn, dense = kind('conv1d'), kind('batch_normalization'), kind('dense')
        igloo = sorted([n for n in layers if n.startswith(('patchy_layer', 'core_igloo'))], key=num)
        att = [n for n in layers if 'attention' in n][0]
        emb = kind('embedding')
        get = lambda layer, name: w[(layer, name)]
        self.conv = [(get(c, 'kernel'), get(c, 'bias')) for c in conv]
        self.bn = [tuple(get(b, n) for n in ('gamma', 'beta', 'moving_mean', 'moving_variance')) for b in bn]
        self.igloo = [(get(g, 'random_patches')[:, :, 0], get(g, 'w_mult')[0], get(g, 'w_bias')[0]) for g in igloo]
        self.dense = [(get(d, 'kernel'), get(d, 'bias')) for d in dense]
        self.att = (get(att, 'w_1'), get(att, 'w_2'))
        mats = [get(e, 'embeddings') for e in emb]
        self.emb_nuc = next(m for m in mats if m.shape == (5, 4)); self.emb_prot = next(m for m in mats if m.shape == (22, 21))

    @staticmethod
    def _conv(x, kernel, bias, dilation):
        k = kernel.shape[0]; pad = dilation * (k - 1); left = pad // 2
        xp = np.pad(x, ((0, 0), (left, pad - left), (0, 0)))
        L = x.shape[1]
        out = sum(xp[:, j * dilation:j * dilation + L, :] @ kernel[j] for j in range(k))
        return out + bias

    @staticmethod
    def _bn(x, p):
        g, b, m, v = p
        return (x - m) / np.sqrt(v + 1e-3) * g + b

    @staticmethod
    def _lrelu(x):
        return np.where(x > 0, x, 0.1 * x)

    def _igloo(self, x, convs, bns, igs, pool):
        """IGLOO1D: six stacks of convolution, batch normalisation and leaky ReLU, each read by a layer of 4-position patches."""
        outs = []
        for i, ((kern, bias), bnp, (patches, wm, wb)) in enumerate(zip(convs, bns, igs)):
            x = self._lrelu(self._bn(self._conv(x, kern, bias, 1 if i == 0 else 3), bnp))
            if i == 0:
                L = x.shape[1] // pool
                x = x[:, :L * pool].reshape(x.shape[0], L, pool, -1).max(axis=2)
            outs.append(self._lrelu(np.einsum('bpsc,psc->bp', x[:, patches, :], wm) + wb))
        return np.concatenate(outs, axis=1)

    def predict(self, seqs, batch=256):
        out = []
        for i in range(0, len(seqs), batch):
            out.append(self._predict(*inputs(seqs[i:i + batch])))
        return np.concatenate(out)

    def _predict(self, nuc, orf_frac, kmer, prot, aa):
        relu = lambda z: np.maximum(z, 0)
        dense = lambda z, j: z @ self.dense[j][0] + self.dense[j][1]
        emb_n = self.emb_nuc[nuc]
        emb_p = self.emb_prot[prot]
        b1 = self._igloo(emb_n, self.conv[0:6], self.bn[0:6], self.igloo[0:6], 8)
        b2 = self._igloo(emb_n, self.conv[6:12], self.bn[6:12], self.igloo[6:12], 8)
        first = relu(self._bn(dense(np.concatenate([b1, b2], 1), 0), self.bn[12]))
        km = relu(self._bn(dense(kmer, 1), self.bn[13]))
        km = relu(self._bn(dense(km, 2), self.bn[14]))
        orf = np.tile(orf_frac[:, None], (1, 64))
        pr = self._igloo(emb_p, self.conv[12:18], self.bn[15:21], self.igloo[12:18], 2)
        pr = relu(self._bn(dense(pr, 3), self.bn[21]))
        a = relu(self._bn(dense(aa, 4), self.bn[22]))
        a = relu(self._bn(dense(a, 5), self.bn[23]))
        second = relu(self._bn(dense(np.concatenate([km, orf, pr, a], 1), 6), self.bn[24]))
        z = orf @ self.att[0] + self.att[1]
        alpha = np.exp(z - z.max(1, keepdims=True)); alpha /= alpha.sum(1, keepdims=True)
        h = alpha[:, :1] * first + alpha[:, 1:] * second
        logits = dense(h, 7)
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return p[:, 1]
