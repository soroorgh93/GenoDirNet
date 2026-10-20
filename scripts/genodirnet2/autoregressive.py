"""Class-conditional autoregressive models, their likelihood-ratio classifier and the synthetic windows.

    python scripts/genodirnet2/autoregressive.py --mode train --class coding    --data data/extraction2 --out results/genodirnet2/lm
    python scripts/genodirnet2/autoregressive.py --mode train --class noncoding --data data/extraction2 --out results/genodirnet2/lm
    python scripts/genodirnet2/autoregressive.py --mode score  --data data/extraction2 --out results/genodirnet2/lm
    python scripts/genodirnet2/autoregressive.py --mode sample --data data/extraction2 --out results/genodirnet2/lm \
        --filter-model results/genodirnet2/so_s0.pt

train   one model p_c (coding) or p_n (noncoding): 128 channels, 20 blocks, next-nucleotide cross-entropy on
        fragment-aware windows of at most 4,096 nucleotides of the training transcripts of the class, AdamW
        (learning rate 1e-3, weight decay 0.01, one-cycle schedule), batch 32, 15 epochs; the epoch with the lowest
        validation code length (bits per nucleotide) is kept                          -> <out>/lm_<class>.pt
score   code lengths on the test transcripts and the AUROC of the per-nucleotide log-likelihood ratio Lambda(x)
        on the complete test transcripts and the fixed test windows                 -> <out>/genlr.json
sample  synthetic windows of 100-600 nucleotides from each model (nucleus sampling, top-p 0.95, context 512),
        accepted when (1) their per-nucleotide log-likelihood under their own model lies within the 5th-95th
        percentiles of real validation windows of the class, (2) Lambda(x) has the sign of the class and exceeds
        the 10th percentile of real windows of the class, (3) the filter model gives the sampled class a probability
        of at most 0.8 and (4) the window shares no 32-nucleotide word with a training transcript and no
        35-nucleotide stretch with an accepted window; up to 10,000 windows per class from at most 400,000
        proposals per class                                                         -> <out>/synthetic.tsv
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.extended.data import TEST_WINDOW_SETS, TranscriptDataset, read_split, worker_init, write_test_windows
from genodirnet.extended.features import one_hot
from genodirnet.extended.models import CausalConvLM, LLRFeatures, build_two_branch
from genodirnet.extended.sampling import FragmentSampler
from genodirnet.metrics import auroc, bootstrap_auroc

CODE = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(b'ACGT'):
    CODE[_c] = _i


class WindowDataset(Dataset):
    def __init__(self, seqs, fragment_aware=True, max_len=4096, seed=0):
        self.seqs, self.max_len = seqs, max_len
        self.sampler = FragmentSampler() if fragment_aware else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        if self.sampler is not None:
            s = self.sampler(s, self.rng)
            if len(s) > self.max_len:                  # training: a random window of max_len nucleotides
                a = int(self.rng.integers(0, len(s) - self.max_len + 1)); s = s[a:a + self.max_len]
        else:
            s = s[:self.max_len]                       # scoring: the first max_len nucleotides
        return s


def collate(seqs):
    L = int(math.ceil(max(len(s) for s in seqs) / 64) * 64)
    xs, ms = zip(*[one_hot(s, L) for s in seqs])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))


def rows_to_seqs(rows, label):
    return [r['sequence'].upper() for r in rows if r['label'] == label]


def load_lm(path, dev):
    ck = torch.load(path, map_location=dev)
    lm = CausalConvLM(**ck['config']); lm.load_state_dict(ck['state_dict'])
    return lm.to(dev).eval()


def macs_per_nucleotide(lm):
    """Multiply-accumulate operations of one prediction step of a CausalConvLM."""
    c, k = lm.channels, lm.kernel
    skip = lm.skips[0].out_channels
    return 5 * c + lm.n_blocks * (2 * c * c * k + c * c + c * skip) + skip * skip + skip * 4


def bits_per_base(lm, seqs, dev, batch=32, max_len=8192):
    dl = DataLoader(WindowDataset(seqs, fragment_aware=False, max_len=max_len), batch_size=batch, shuffle=False,
                    num_workers=4, collate_fn=collate)
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for x, m in dl:
            x, m = x.to(dev), m.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                ll = lm.log_prob(x, m)
            tot += -ll.sum().item() / math.log(2.0); n += m.sum().item()
    return tot / max(n, 1.0)


def train_lm(args, dev):
    label = 1 if args.cls == 'coding' else 0
    train = rows_to_seqs(read_split(os.path.join(args.data, 'train.tsv')), label)
    val = rows_to_seqs(read_split(os.path.join(args.data, 'val.tsv')), label)
    print(f'{args.cls}: {len(train)} training, {len(val)} validation transcripts')
    cfg = dict(channels=128, n_blocks=20, kernel=3, skip_channels=128, cycle=10)
    lm = CausalConvLM(**cfg).to(dev)
    ds = WindowDataset(train, fragment_aware=True, max_len=4096, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=collate,
                    pin_memory=(dev == 'cuda'), worker_init_fn=worker_init if args.workers > 0 else None)
    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    best, hist = 1e9, []
    os.makedirs(args.out, exist_ok=True); path = os.path.join(args.out, f'lm_{args.cls}.pt')
    for ep in range(args.epochs):
        lm.train(); t0 = time.time(); tot, n = 0.0, 0
        for x, m in dl:
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                logits = lm(x)
            ce = F.cross_entropy(logits.float(), x.argmax(1), reduction='none') * m
            loss = ce.sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        lm.eval(); vb = bits_per_base(lm, val, dev)
        hist.append({'epoch': ep + 1, 'train_bits': tot / n / math.log(2.0), 'val_bits': vb})
        print(f'epoch {ep + 1:2d}  training {tot / n / math.log(2.0):.4f} bits/nt  validation {vb:.4f} bits/nt  ({time.time() - t0:.0f} s)', flush=True)
        if vb < best:
            best = vb
            torch.save({'config': cfg, 'state_dict': lm.state_dict(), 'class': args.cls, 'val_bits': vb, 'history': hist}, path)
    print('saved', path, f'(validation {best:.4f} bits/nt)')


@torch.no_grad()
def llr_features(feat, seqs, dev, batch=64):
    dl = DataLoader(WindowDataset(seqs, fragment_aware=False, max_len=8192), batch_size=batch, shuffle=False,
                    num_workers=4, collate_fn=collate)
    out = []
    for x, m in dl:
        x, m = x.to(dev), m.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            out.append(feat.raw(x, m).float().cpu().numpy())
    return np.concatenate(out)


def score_mode(args, dev):
    lm_c, lm_n = load_lm(os.path.join(args.out, 'lm_coding.pt'), dev), load_lm(os.path.join(args.out, 'lm_noncoding.pt'), dev)
    n_params = sum(p.numel() for p in lm_c.parameters())
    feat = LLRFeatures(lm_c, lm_n).to(dev)
    test = read_split(os.path.join(args.data, 'test.tsv'))
    wdir = os.path.join(args.data, 'test_windows')
    if not os.path.exists(os.path.join(wdir, 'w300_indel2.tsv')):
        write_test_windows(test, wdir)
    sets = {'full': test, **{k: read_split(os.path.join(wdir, f'{k}.tsv')) for k in TEST_WINDOW_SETS}}
    rec = {'n_params_each': n_params, 'macs_per_nucleotide_each': macs_per_nucleotide(lm_c), 'bits_per_base': {}, 'test': {}}
    for name, lm, label in (('coding_model_on_coding', lm_c, 1), ('noncoding_model_on_noncoding', lm_n, 0),
                            ('coding_model_on_noncoding', lm_c, 0), ('noncoding_model_on_coding', lm_n, 1)):
        rec['bits_per_base'][name] = bits_per_base(lm, rows_to_seqs(test, label), dev)
    for k, rows in sets.items():
        s = llr_features(feat, [r['sequence'].upper() for r in rows], dev)[:, 0]
        y = np.array([r['label'] for r in rows])
        lo, hi = np.percentile(bootstrap_auroc(y, s), [2.5, 97.5])
        rec['test'][k] = {'genlr_auroc': 100 * auroc(y, s), 'genlr_ci': [100 * lo, 100 * hi], 'n': len(y)}
        print(f'{k:12s} likelihood ratio AUROC {100 * auroc(y, s):.2f} [{100 * lo:.2f}, {100 * hi:.2f}]', flush=True)
    json.dump(rec, open(os.path.join(args.out, 'genlr.json'), 'w'), indent=2)
    print(json.dumps(rec['bits_per_base'], indent=1))


def kmer_codes(seq, k=32):
    """2-bit codes of all k-mers (k <= 32) of a sequence; k-mers with an ambiguous base are skipped."""
    a = CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(a) < k:
        return np.zeros(0, dtype=np.uint64)
    bad = a == 255; a = a.astype(np.uint64) & np.uint64(3)
    win = np.lib.stride_tricks.sliding_window_view(a, k)
    codes = (win * (np.uint64(4) ** np.arange(k - 1, -1, -1, dtype=np.uint64))).sum(1, dtype=np.uint64)
    if bad.any():
        codes = codes[~np.lib.stride_tricks.sliding_window_view(bad, k).any(1)]
    return codes


class KmerIndex:
    """The k-mers of the training transcripts, and every fourth k-mer of the accepted windows."""

    def __init__(self, seqs, k=32):
        self.k = k
        parts = [kmer_codes(s, k) for s in seqs]
        self.codes = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)
        self.extra = set()

    def shares(self, seq):
        c = kmer_codes(seq, self.k)
        if len(c) == 0:
            return False
        if len(self.codes):
            pos = np.searchsorted(self.codes, c); pos[pos >= len(self.codes)] = 0
            if bool((self.codes[pos] == c).any()):
                return True
        return any(int(v) in self.extra for v in c)

    def add(self, seq):
        self.extra.update(int(v) for v in kmer_codes(seq, self.k)[::4])


def onehot_to_str(x):
    return [''.join('ACGT'[i] for i in row) for row in x.argmax(1).cpu().numpy()]


def load_filter(path, dev):
    """Sequence-ORF network trained by scripts/genodirnet2/train.py, with its hexamer table and standardisation."""
    from genodirnet.extended.models import load_llr
    ck = torch.load(path, map_location=dev)
    llr = load_llr(ck['llr_dir'], dev) if ck.get('llr_dir') else None
    model = build_two_branch(ck['spec'], llr=llr).to(dev)
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    if [k for k in missing if not k.startswith('llr.lm_')] or unexpected:
        sys.exit(f'{path} does not match the network it describes')
    model.eval()
    return model, ck['hexamer_table'], np.array(ck['feat_mean'], dtype=np.float32), np.array(ck['feat_std'], dtype=np.float32)


def sample_mode(args, dev):
    lm = {1: load_lm(os.path.join(args.out, 'lm_coding.pt'), dev), 0: load_lm(os.path.join(args.out, 'lm_noncoding.pt'), dev)}
    feat = LLRFeatures(lm[1], lm[0]).to(dev)
    train, val = read_split(os.path.join(args.data, 'train.tsv')), read_split(os.path.join(args.data, 'val.tsv'))
    rng = np.random.default_rng(args.seed)
    ref = {}
    for label in (1, 0):                              # reference bands from 2,000 real validation windows per class
        seqs = rows_to_seqs(val, label); rng.shuffle(seqs); seqs = seqs[:2000]
        wins = []
        for s in seqs:
            L = int(rng.integers(args.min_len, args.max_len + 1))
            if len(s) <= L:
                wins.append(s)
            else:
                a = int(rng.integers(0, len(s) - L + 1)); wins.append(s[a:a + L])
        Fw = llr_features(feat, wins, dev)
        own = []
        with torch.no_grad():
            for x, m in DataLoader(WindowDataset(wins, fragment_aware=False), batch_size=64, collate_fn=collate):
                x, m = x.to(dev), m.to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                    ll = lm[label].log_prob(x, m).float()
                own.append((ll.sum(1) / m.sum(1)).cpu().numpy() / math.log(2.0))
        own = np.concatenate(own); sgn = 1 if label == 1 else -1
        ref[label] = {'ll_lo': float(np.percentile(own, 5)), 'll_hi': float(np.percentile(own, 95)),
                      'llr_min': float(np.percentile(sgn * Fw[:, 0], 10))}
        print(f'class {label}: own log-likelihood band [{ref[label]["ll_lo"]:.3f}, {ref[label]["ll_hi"]:.3f}] bits/nt, '
              f'class-consistent likelihood ratio >= {ref[label]["llr_min"]:.3f}')
    index = KmerIndex(rows_to_seqs(train, 1) + rows_to_seqs(train, 0), k=32)
    model, table, mean, std = load_filter(args.filter_model, dev)
    accepted = []
    stats = {'proposed': 0, 'rejected_plausibility': 0, 'rejected_class': 0, 'rejected_difficulty': 0, 'rejected_novelty': 0, 'accepted': 0}
    for label in (1, 0):
        n_acc, sgn, proposed = 0, (1 if label == 1 else -1), 0
        while n_acc < args.n_synth // 2 and proposed < args.max_proposals:
            proposed += args.gen_batch
            L = int(rng.integers(args.min_len, args.max_len + 1)); b = args.gen_batch
            x = lm[label].sample(b, L, dev, temperature=1.0, top_p=0.95, context=512)
            m = torch.ones(b, L, device=dev)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                f = feat.raw(x, m).float().cpu().numpy()
                own = (lm[label].log_prob(x, m).float().sum(1) / L).cpu().numpy() / math.log(2.0)
            seqs = onehot_to_str(x); stats['proposed'] += b
            keep = (own >= ref[label]['ll_lo']) & (own <= ref[label]['ll_hi']); stats['rejected_plausibility'] += int((~keep).sum())
            c = sgn * f[:, 0] >= max(ref[label]['llr_min'], 0.0); stats['rejected_class'] += int((keep & ~c).sum()); keep &= c
            if keep.any():
                rows = [{'sequence': s, 'label': label} for s, k in zip(seqs, keep) if k]
                p = []
                with torch.no_grad():
                    for xx, mm, ff, _, _ in DataLoader(TranscriptDataset(rows, table, mean, std, fragment_aware=False), batch_size=128):
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                            lg = model(xx.to(dev), ff.to(dev), mm.to(dev)).float()
                        p.append(torch.softmax(lg, 1)[:, label].cpu().numpy())
                hard = np.concatenate(p) <= 0.8
                idx = np.where(keep)[0]; stats['rejected_difficulty'] += int((~hard).sum()); keep[idx[~hard]] = False
            for i in np.where(keep)[0]:
                if index.shares(seqs[i]):
                    stats['rejected_novelty'] += 1; continue
                accepted.append({'id': f'synth_{label}_{len(accepted)}', 'label': label, 'chrom': 'synthetic', 'sequence': seqs[i]})
                index.add(seqs[i]); n_acc += 1; stats['accepted'] += 1
                if n_acc >= args.n_synth // 2:
                    break
            print(f'class {label}: accepted {n_acc}/{args.n_synth // 2}  {json.dumps(stats)}', flush=True)
    out = os.path.join(args.out, 'synthetic.tsv')
    with open(out, 'w') as fh:
        fh.write('id\tlabel\tchrom\tsequence\n')
        for r in accepted:
            fh.write(f"{r['id']}\t{r['label']}\t{r['chrom']}\t{r['sequence']}\n")
    json.dump({'stats': stats, 'reference_bands': {str(k): v for k, v in ref.items()}},
              open(os.path.join(args.out, 'synthetic_report.json'), 'w'), indent=2)
    print('wrote', out, len(accepted), 'windows')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['train', 'score', 'sample'], required=True)
    ap.add_argument('--class', dest='cls', choices=['coding', 'noncoding'])
    ap.add_argument('--data', required=True, help='folder with train.tsv, val.tsv and test.tsv (extraction 2)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--filter-model', help='sample: network that applies the difficulty check')
    ap.add_argument('--n-synth', type=int, default=20000)
    ap.add_argument('--min-len', type=int, default=100); ap.add_argument('--max-len', type=int, default=600)
    ap.add_argument('--gen-batch', type=int, default=256); ap.add_argument('--max-proposals', type=int, default=400000)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'train':
        if not args.cls:
            ap.error('--mode train needs --class')
        train_lm(args, dev)
    elif args.mode == 'score':
        score_mode(args, dev)
    else:
        if not args.filter_model:
            ap.error('--mode sample needs --filter-model')
        sample_mode(args, dev)


if __name__ == '__main__':
    main()
