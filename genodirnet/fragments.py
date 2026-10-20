"""Simulated fragments (Methods: Simulated fragments and sequencing errors) and fragment-aware training windows.

fixed_windows draws one random window per transcript and length with a fixed seed, so that every model scores
the same windows; a transcript shorter than the window is kept whole. add_indels places single-nucleotide
insertions and deletions at a given rate per position, half of each. fat_window is the window of fragment-aware
training; retraining_set is the fixed training set of the retrained released tools.
"""
import numpy as np

WINDOW_LENGTHS = (100, 200, 300, 500, 1000)
INDEL_RATES = {'i1': 0.01, 'i2': 0.02}
P_WINDOW, MIN_FRAC, MIN_WINDOW = 0.8, 0.5, 200


def window_starts(seqs, length, seed=0):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, len(s) - min(len(s), length) + 1)) for s in seqs]


def fixed_windows(seqs, length, seed=0):
    """One random window of at most `length` nucleotides per transcript (the same windows for every model)."""
    return [s[st:st + min(len(s), length)] for s, st in zip(seqs, window_starts(seqs, length, seed))]


def add_indels(seq, rate, rng):
    """Single-nucleotide deletions (rate / 2) and insertions of a uniform random nucleotide (rate / 2) per position."""
    out = []
    for ch in seq:
        r = rng.random()
        if r < rate / 2:
            continue
        out.append(ch)
        if r > 1 - rate / 2:
            out.append('ACGT'[int(rng.integers(0, 4))])
    return ''.join(out) if len(out) >= 20 else seq


def indel_windows(seqs, length, tag):
    """The fixed windows of one length with 1 (tag 'i1') or 2 (tag 'i2') insertions or deletions per 100 nt."""
    rate = INDEL_RATES[tag]
    rng = np.random.default_rng(1000 + int(rate * 1000))
    return [add_indels(s, rate, rng) for s in fixed_windows(seqs, length)]


def dinucleotide_shuffle(seq, rng):
    """A random permutation of `seq` with the same dinucleotide counts (Altschul and Erickson, 1985): a random
    last-edge tree of the dinucleotide graph is drawn, the remaining edges of each vertex are shuffled, and the
    Euler path is read off. The sequence is returned unchanged if it has fewer than three nucleotides."""
    if len(seq) < 3:
        return seq
    letters = sorted(set(seq)); last = seq[-1]
    edges = {a: [] for a in letters}
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    while True:                                           # last edges: a random tree with its root at the last letter
        tree = {}
        for a in letters:
            if a == last:
                continue
            tree[a] = edges[a][int(rng.integers(0, len(edges[a])))] if edges[a] else None
        ok = True
        for a in tree:
            seen, v = set(), a
            while v != last and v is not None and v not in seen:
                seen.add(v); v = tree.get(v)
            if v != last:
                ok = False; break
        if ok:
            break
    rest = {}
    for a in letters:
        e = list(edges[a])
        if a in tree and tree[a] is not None:
            e.remove(tree[a])
        rng.shuffle(e)
        rest[a] = e + ([tree[a]] if a in tree and tree[a] is not None else [])
    out, v = [seq[0]], seq[0]
    while len(out) < len(seq):
        v = rest[v].pop(0); out.append(v)
    return ''.join(out)


def shuffled_windows(windows, seed=0):
    """Dinucleotide-preserving shuffles of a set of windows (control for compositional signal)."""
    rng = np.random.default_rng(2000 + seed)
    return [dinucleotide_shuffle(w, rng) for w in windows]


ABSOLUTE_RANGE = (100, 1000)


def fat_window(seq, rng, mode='relative'):
    """Fragment-aware training: with probability 0.8 a random contiguous window of the transcript, covering a uniform
    fraction of 50-100 % of its length and at least 200 nt (mode 'relative'), or of a length drawn log-uniformly
    between 100 and 1,000 nt (mode 'absolute'; a shorter transcript is kept whole)."""
    if rng.random() < P_WINDOW:
        n = len(seq)
        if mode == 'relative':
            w = max(MIN_WINDOW, int(n * rng.uniform(MIN_FRAC, 1.0)))
        else:
            w = min(n, int(round(np.exp(rng.uniform(np.log(ABSOLUTE_RANGE[0]), np.log(ABSOLUTE_RANGE[1]))))))
        st = int(rng.integers(0, n - w + 1))
        return seq[st:st + w]
    return seq


def retraining_set(labels, seqs, copies=2, seed=7):
    """Training set of the retrained released tools (RNAsamba, CPAT, CPC2), which train on a fixed set of sequences:
    every transcript once complete and `copies` times as a random window covering 50-100 % of its length (at least
    200 nt), drawn once before training; coding transcripts first. Returns a list of (label, name, sequence)."""
    rng = np.random.default_rng(seed); out = []
    for lab in (1, 0):
        for i, (y, s) in enumerate(zip(labels, seqs)):
            if y != lab:
                continue
            out.append((lab, f't{i}', s))
            for c in range(copies):
                n = len(s); w = max(MIN_WINDOW, int(n * rng.uniform(MIN_FRAC, 1.0))); st = int(rng.integers(0, n - w + 1))
                out.append((lab, f't{i}w{c}', s[st:st + w]))
    return out
