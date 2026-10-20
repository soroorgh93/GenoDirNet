"""Pretraining of the sequence network: replaced-nucleotide detection or masked denoising.

    python scripts/genodirnet2/pretrain.py --method rnd    --data data/extraction2 --out results/genodirnet2/encoder_rnd.pt
    python scripts/genodirnet2/pretrain.py --method rnd    --data data/extraction2 --arch results/genodirnet2/search1/best_arch.json \
        --out results/genodirnet2/encoder_rnd_deeper.pt
    python scripts/genodirnet2/pretrain.py --method masked --data data/extraction2 --out results/genodirnet2/encoder_masked.pt

Both methods read fragment-aware windows of the training transcripts (at most 8,192 nucleotides), train for 20
epochs with AdamW (learning rate 1e-3, weight decay 0.01, one-cycle schedule, batch 64, gradient clipping at norm 1)
and save the sequence network after every epoch.

rnd     replaced-nucleotide detection: 25 % of each window is hidden in whole codons on one random frame (spans of
        3, 6, 9, ... nucleotides, mean two codons); a generator of three convolutions with 64 filters predicts the
        hidden nucleotides (cross-entropy L_G); the hidden positions are replaced by samples from the generator, and
        the sequence network with a light transposed-convolution decoder labels every position as original or
        replaced (binary cross-entropy L_RND). Loss L_G + 50 L_RND.
masked  masked denoising: 15 % of each window is chosen in spans of geometric length (mean 5 nucleotides); of the
        chosen positions 80 % are zeroed, 10 % replaced by a nucleotide drawn from the composition of the window and
        10 % left unchanged; the decoder predicts the nucleotide at the chosen positions (cross-entropy).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.extended.data import read_split, worker_init
from genodirnet.extended.features import one_hot
from genodirnet.extended.models import MaskedDenoisingModel, SequenceEncoder, SmallGenerator, load_spec, spec_name
from genodirnet.extended.sampling import FragmentSampler


class MaskedDataset(Dataset):
    def __init__(self, seqs, seq_len=8192, mask_frac=0.15, span_mean=5.0, seed=0):
        self.seqs, self.seq_len, self.mask_frac, self.span_mean = seqs, seq_len, mask_frac, span_mean
        self.sampler = FragmentSampler(p_window=0.8, min_frac=0.5, max_frac=1.0)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.sampler(self.seqs[i], self.rng)[:self.seq_len]
        x, _ = one_hot(s, self.seq_len)
        L = len(s)
        target = x.argmax(0).astype(np.int64)
        target[x.sum(0) == 0] = -100                         # padding and ambiguous bases are ignored
        chosen = np.zeros(self.seq_len, dtype=bool)
        n_target, n_chosen = int(self.mask_frac * L), 0
        self.rng.integers(0, 3)                              # one draw before the span sampling
        while n_chosen < n_target:
            span = max(1, int(self.rng.geometric(1.0 / self.span_mean)))
            a = int(self.rng.integers(0, max(1, L - span + 1)))
            chosen[a:a + span] = True
            n_chosen = int(chosen.sum())
        chosen &= (target != -100)
        xm = x.copy()
        u = self.rng.random(self.seq_len)
        zero = chosen & (u < 0.8)
        rand = chosen & (u >= 0.8) & (u < 0.9)
        xm[:, zero] = 0.0
        if rand.any():                                       # replacement drawn from the composition of the window
            comp = x[:, :L].sum(1) + 1.0
            rnd = self.rng.choice(4, size=int(rand.sum()), p=comp / comp.sum())
            xm[:, rand] = 0.0
            xm[rnd, np.where(rand)[0]] = 1.0
        y = np.full(self.seq_len, -100, dtype=np.int64)
        y[chosen] = target[chosen]
        return torch.from_numpy(xm), torch.from_numpy(y)


class CorruptDataset(Dataset):
    """Clean one-hot window, generator input (hidden positions zeroed), hidden positions and real positions."""

    def __init__(self, seqs, seq_len=8192, mask_frac=0.25, seed=0):
        self.seqs, self.seq_len, self.mask_frac = seqs, seq_len, mask_frac
        self.sampler = FragmentSampler(p_window=0.8, min_frac=0.5, max_frac=1.0)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.sampler(self.seqs[i], self.rng)[:self.seq_len]
        x, _ = one_hot(s, self.seq_len)
        L = len(s)
        real = x.sum(0) > 0
        chosen = np.zeros(self.seq_len, dtype=bool)
        n_target, n_chosen, frame = int(self.mask_frac * L), 0, int(self.rng.integers(0, 3))
        while n_chosen < n_target:
            span = 3 * max(1, int(self.rng.geometric(0.5)))
            a = frame + 3 * int(self.rng.integers(0, max(1, (L - span) // 3 + 1)))
            chosen[a:a + span] = True
            n_chosen = int(chosen.sum())
        chosen &= real
        xg = x.copy(); xg[:, chosen] = 0.0
        return torch.from_numpy(x), torch.from_numpy(xg), torch.from_numpy(chosen), torch.from_numpy(real)


def train_rnd(args, seqs, spec, dev):
    dl = DataLoader(CorruptDataset(seqs, args.seq_len, 0.25, args.seed), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=(dev == 'cuda'), worker_init_fn=worker_init if args.workers > 0 else None)
    gen = SmallGenerator().to(dev)
    disc = MaskedDenoisingModel(SequenceEncoder(spec), out_dim=1).to(dev)
    params = list(gen.parameters()) + list(disc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    for ep in range(args.epochs):
        gen.train(); disc.train(); t0 = time.time(); lg = ld = 0.0; n = 0
        for x, xg, chosen, real in dl:
            x, xg, chosen, real = (t.to(dev, non_blocking=True) for t in (x, xg, chosen, real))
            target = x.argmax(1)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                g_logits = gen(xg)
            loss_gen = F.cross_entropy(g_logits.float().transpose(1, 2)[chosen], target[chosen])
            with torch.no_grad():
                probs = torch.softmax(g_logits.float().transpose(1, 2)[chosen], dim=1)
                xt = x.transpose(1, 2).clone()
                xt[chosen] = F.one_hot(torch.multinomial(probs, 1).squeeze(1), 4).to(xt.dtype)
                xc = xt.transpose(1, 2).contiguous()
                replaced = (xc.argmax(1) != target) & real          # a sample equal to the original counts as original
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                d_logits = disc(xc).squeeze(1)
            loss_disc = F.binary_cross_entropy_with_logits(d_logits.float()[real], replaced[real].float())
            loss = loss_gen + 50.0 * loss_disc
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            lg += loss_gen.item(); ld += loss_disc.item(); n += 1
        print(f'epoch {ep + 1:3d}  generator {lg / n:.4f}  detection {ld / n:.4f}  ({time.time() - t0:.0f} s)', flush=True)
        torch.save({'encoder': disc.encoder.state_dict(), 'spec': spec, 'method': 'rnd', 'epoch': ep + 1}, args.out)


def train_masked(args, seqs, spec, dev):
    dl = DataLoader(MaskedDataset(seqs, args.seq_len, 0.15, 5.0, args.seed), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=(dev == 'cuda'), worker_init_fn=worker_init if args.workers > 0 else None)
    model = MaskedDenoisingModel(SequenceEncoder(spec)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    for ep in range(args.epochs):
        model.train(); tot, n, t0 = 0.0, 0, time.time()
        for xm, y in dl:
            xm, y = xm.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                logits = model(xm)
            loss = F.cross_entropy(logits.float(), y, ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        print(f'epoch {ep + 1:3d}  masked cross-entropy {tot / n:.4f}  ({time.time() - t0:.0f} s)', flush=True)
        torch.save({'encoder': model.encoder.state_dict(), 'spec': spec, 'method': 'masked', 'epoch': ep + 1}, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=['rnd', 'masked'], required=True)
    ap.add_argument('--data', required=True, help='folder with train.tsv (extraction 2)')
    ap.add_argument('--arch', default='default', help='design specification of the sequence network (JSON)')
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cudnn.benchmark = True
    seqs = [r['sequence'].upper() for r in read_split(os.path.join(args.data, 'train.tsv'))]
    seqs = [s for s in seqs if len(s) >= 200]
    spec = load_spec(args.arch)
    print(f'{len(seqs)} training transcripts; sequence network {spec_name(spec)}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    (train_rnd if args.method == 'rnd' else train_masked)(args, seqs, spec, dev)
    print('saved the sequence network to', args.out)


if __name__ == '__main__':
    main()
