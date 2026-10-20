"""Frame-resolved chaos game representation (FR-CGR) and its controls.

A word w = w_1 ... w_k occupies the cell (row(w), col(w)) of a 2^k x 2^k grid with row bits b(A) = b(C) = 0,
b(T) = b(G) = 1 and column bits a(A) = a(T) = 0, a(C) = a(G) = 1, the j-th letter giving bit j - 1, so that the
last letter selects the quadrant. Channel f in {0, 1, 2} counts the words that start at a position t with
(t - 1) mod 3 = f. The network input is log(1 + N_f(w)), standardised per channel.
"""
import numpy as np

_BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}
_LUT = np.full(256, -1, dtype=np.int64)
for _b, (_r, _c) in _BITS.items():
    _LUT[ord(_b)] = _r * 2 + _c
    _LUT[ord(_b.lower())] = _r * 2 + _c


def encode(seq):
    """Integer code of every nucleotide (row bit * 2 + column bit), -1 for ambiguous bases."""
    return _LUT[np.frombuffer(seq.encode('ascii', 'replace'), dtype=np.uint8)]


def _cells(seq, k):
    codes = encode(seq); n = codes.shape[0]; m = n - k + 1
    rows, cols, valid = (codes >> 1) & 1, codes & 1, codes >= 0
    r = np.zeros(m, dtype=np.int64); c = np.zeros(m, dtype=np.int64); ok = np.ones(m, dtype=bool)
    for j in range(k):
        sl = slice(j, j + m)
        r |= rows[sl] << j; c |= cols[sl] << j; ok &= valid[sl]
    return r, c, ok


def cgr_counts(seq, k):
    """Frequency chaos game representation: count of every word of length k (2^k x 2^k)."""
    size = 1 << k
    grid = np.zeros((size, size), dtype=np.float32)
    if len(seq) < k:
        return grid
    r, c, ok = _cells(seq, k)
    np.add.at(grid, (r[ok], c[ok]), 1.0)
    return grid


def frame_counts(seq, k=7):
    """Frame-resolved counts N_f(w): channel f counts the words that start at positions p with p mod 3 = f."""
    size = 1 << k
    grid = np.zeros((3, size, size), dtype=np.int32)
    if len(seq) < k:
        return grid
    r, c, ok = _cells(seq, k)
    frame = np.arange(len(r)) % 3
    np.add.at(grid, (frame[ok], r[ok], c[ok]), 1)
    return grid


_ASSIGN = np.random.default_rng(777).integers(0, 3, size=40001)


def partition_counts(seq, k=7):
    """Control: the words are assigned to the three channels by a fixed random partition of the start positions."""
    size = 1 << k
    grid = np.zeros((3, size, size), dtype=np.int32)
    if len(seq) < k:
        return grid
    r, c, ok = _cells(seq, k)
    part = _ASSIGN[np.arange(len(r)) % len(_ASSIGN)]
    np.add.at(grid, (part[ok], r[ok], c[ok]), 1)
    return grid


def pool_counts(counts, k_out):
    """Counts of shorter words from those of length k: summing the 2 x 2 blocks of the grid removes the first letter
    of every word, whose start moves by one position, so the channels are rotated by one. Pooling the counts of
    7-letter words gives the counts of the k_out-letter words that start at positions 7 - k_out, ..., L - k_out."""
    c = counts; k = int(np.log2(c.shape[-1]))
    while k > k_out:
        s = c.shape[-1] // 2
        c = np.roll(c.reshape(3, s, 2, s, 2).sum(axis=(2, 4)), 1, axis=0)
        k -= 1
    return c


def standardise(g):
    """log(1 + count), then zero mean and unit variance over the cells of one channel."""
    g = np.log1p(g.astype(np.float32))
    return (g - g.mean()) / (g.std() + 1e-6)


_PERMS = {}


def _permutation(n):
    if n not in _PERMS:
        _PERMS[n] = np.random.default_rng(12345).permutation(n)
    return _PERMS[n]


CONTROLS = ('summed', 'partition', 'permuted')


def network_input(seq, k=5, control=None):
    """Input of the FR-CGR network (3 x 2^k x 2^k, float32), computed from the counts of 7-letter words.

    control: None (FR-CGR), 'summed' (the three channels summed: the standard FCGR, repeated in three channels),
    'partition' (channels from a random partition of the start positions) or 'permuted' (FR-CGR with the cells
    in a fixed random order).
    """
    counts = partition_counts(seq, 7) if control == 'partition' else frame_counts(seq, 7)
    c = pool_counts(counts, k)
    if control == 'summed':
        g = standardise(c.sum(axis=0))
        return np.stack([g, g, g], axis=0)
    x = np.stack([standardise(c[f]) for f in range(3)], axis=0)
    if control == 'permuted':
        n = x.shape[-1] * x.shape[-2]
        x = x.reshape(3, n)[:, _permutation(n)].reshape(x.shape)
    return x
