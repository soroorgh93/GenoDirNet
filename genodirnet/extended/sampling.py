"""Fragment-aware windows, error-aware noise and the window-length bins of the length-weighted loss.

All randomness comes from the NumPy generator passed in, so every draw is reproducible.
"""

# ------------------------------------------------------------------ fragment-aware training
class FragmentSampler:
    """With probability p_window, replace a transcript of L nucleotides by a random contiguous window of
    max(min_len, round(f L)) nucleotides, f uniform in [min_frac, max_frac]."""

    def __init__(self, p_window=0.8, min_frac=0.5, max_frac=1.0, min_len=100):
        self.p_window, self.min_frac, self.max_frac, self.min_len = p_window, min_frac, max_frac, min_len

    def __call__(self, seq, rng):
        L = len(seq)
        if L <= self.min_len or rng.random() >= self.p_window:
            return seq
        frac = rng.uniform(self.min_frac, self.max_frac)
        w = max(self.min_len, int(round(frac * L)))
        if w >= L:
            return seq
        start = rng.integers(0, L - w + 1)
        return seq[start:start + w]


# ------------------------------------------------------------------ error-aware augmentation
class ErrorAwareNoise:
    """With probability p_apply, delete and insert single nucleotides, each at half of a rate drawn uniformly in
    [0, max_indel_rate] per nucleotide, and replace nucleotides at sub_rate_factor times that rate by a uniformly drawn
    nucleotide (which can be the original one)."""

    def __init__(self, p_apply=0.5, max_indel_rate=0.02, sub_rate_factor=0.5, alphabet='ACGT'):
        self.p_apply, self.max_indel_rate, self.sub_rate_factor, self.alphabet = p_apply, max_indel_rate, sub_rate_factor, alphabet

    def __call__(self, seq, rng):
        if rng.random() >= self.p_apply:
            return seq
        rate = rng.uniform(0.0, self.max_indel_rate)
        if rate <= 0:
            return seq
        sub_rate = rate * self.sub_rate_factor
        out = []
        A = self.alphabet
        for c in seq:
            u = rng.random()
            if u < rate / 2:                      # deletion
                continue
            if u < rate:                          # insertion before c
                out.append(A[rng.integers(0, 4)])
            if rng.random() < sub_rate:           # substitution
                c = A[rng.integers(0, 4)]
            out.append(c)
        return ''.join(out) if out else seq


# ------------------------------------------------------------------ window-length bins
LENGTH_BINS = [(0, 200), (200, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 10 ** 9)]


def length_bin(length):
    for g, (lo, hi) in enumerate(LENGTH_BINS):
        if lo <= length < hi:
            return g
    return len(LENGTH_BINS) - 1
