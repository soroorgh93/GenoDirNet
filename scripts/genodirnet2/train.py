"""Train a network of the extended schedule: GenoDirNet-SO, one addition at a time, or GenoDirNet-2.

    python scripts/genodirnet2/train.py --data data/extraction2 --recipe so --seed 0 --out results/genodirnet2/so_s0.json

Recipes (each sets the switches below; the files they need are given explicitly):
  so               GenoDirNet-SO: the sequence-ORF network of GenoDirNet, constant learning rate, early stopping
  llr              + likelihood-ratio features                      (--llr-dir)
  erroraware       + error-aware augmentation
  pretrain         + replaced-nucleotide pretraining of the default sequence network     (--init-encoder)
  masked           + masked denoising pretraining of the default sequence network        (--init-encoder)
  synthetic        + synthetic windows                              (--synthetic)
  deeper           + the sequence network of the first architecture search, no pretraining   (--arch)
  lengthweighted   + length-weighted loss                           (--tau)
  gdn2             GenoDirNet-2: likelihood-ratio features, error-aware augmentation, the deeper sequence network
                   pretrained by replaced-nucleotide detection, synthetic windows and the length-weighted loss,
                   with a cosine learning-rate schedule   (--arch, --init-encoder,
                   --llr-dir, --synthetic, --tau)
A sequence network found by a search (for example the compact network of the second search) is trained as
GenoDirNet-SO with --recipe so --arch <specification>.

Settings of the extended schedule: AdamW (learning rate 1e-3, weight decay 0.01), batch 64, at most 30 epochs,
bfloat16 autocast on the GPU, gradient clipping at norm 5, exponential moving average of the weights (decay
0.999), fragment-aware windows (probability 0.8, 50-100 % of the transcript, at least 100 nucleotides). Model
selection uses the mean of the validation AUROC on complete transcripts and on one 300-nucleotide window per
validation transcript (seed 999); with the constant schedule, training stops after 6 epochs without improvement.
Synthetic windows form at most 20 % of each epoch, with loss weight 0.5.

Writes the record <out> (settings, history, test AUROC on the complete test transcripts and the fixed test windows)
and the checkpoint <out without .json>.pt.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.extended.data import (TEST_WINDOW_SETS, TranscriptDataset, feature_prep, make_test_windows,
                                      read_split, worker_init, write_test_windows)
from genodirnet.extended.models import (SEQ_LEN, LengthWeightedLoss, build_two_branch, count_macs_reader,
                                        count_parameters, load_llr, load_spec, spec_name)
from genodirnet.extended.sampling import LENGTH_BINS
from genodirnet.metrics import auroc, auprc

RECIPES = {
    'so': dict(), 'llr': dict(needs=['llr_dir']), 'erroraware': dict(error_aware=True),
    'pretrain': dict(needs=['init_encoder']), 'masked': dict(needs=['init_encoder']),
    'synthetic': dict(needs=['synthetic']), 'deeper': dict(needs=['arch']), 'lengthweighted': dict(needs=['tau']),
    'gdn2': dict(error_aware=True, sched='cosine', needs=['arch', 'init_encoder', 'llr_dir', 'synthetic', 'tau']),
}


class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            if ps.requires_grad or pm.requires_grad:
                ps.mul_(self.decay).add_(pm.detach(), alpha=1 - self.decay)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)


class WeightedDataset(TranscriptDataset):
    """TranscriptDataset that also returns the loss weight of each example (1 for transcripts, w for synthetic)."""

    def __init__(self, rows, weights, *a, **k):
        super().__init__(rows, *a, **k); self.w = weights

    def __getitem__(self, i):
        return (*super().__getitem__(i), torch.tensor(self.w[i], dtype=torch.float32))


def autocast(dev, on):
    return torch.autocast('cuda', dtype=torch.bfloat16) if (on and dev == 'cuda') else torch.autocast('cpu', enabled=False)


@torch.no_grad()
def score(model, rows, table, mean, std, dev, batch=128, amp=True, workers=6):
    """Logit difference (coding minus noncoding) for every row."""
    ds = TranscriptDataset(rows, table, mean, std, fragment_aware=False, error_aware=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=(dev == 'cuda'))
    out, ys = [], []
    model.eval()
    for x, m, f, y, _ in dl:
        with autocast(dev, amp):
            logit = model(x.to(dev, non_blocking=True), f.to(dev), m.to(dev)).float()
        out.append((logit[:, 1] - logit[:, 0]).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with train.tsv, val.tsv and test.tsv (extraction 2)')
    ap.add_argument('--windows-dir', default=None, help='fixed test windows (default <data>/test_windows; written if absent)')
    ap.add_argument('--recipe', required=True, choices=sorted(RECIPES))
    ap.add_argument('--init-encoder', default=None, help='pretrained sequence network (scripts/genodirnet2/pretrain.py)')
    ap.add_argument('--arch', default='default', help='design specification of the sequence network (JSON)')
    ap.add_argument('--llr-dir', default=None, help='folder with lm_coding.pt and lm_noncoding.pt')
    ap.add_argument('--synthetic', default=None, help='synthetic windows (scripts/genodirnet2/autoregressive.py --mode sample)')
    ap.add_argument('--tau', type=float, default=None, help='temperature of the length-weighted loss')
    ap.add_argument('--synthetic-frac', type=float, default=0.2); ap.add_argument('--synthetic-weight', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=6); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--max-train', type=int, default=None, help='random subset of the training transcripts (short runs)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    for need in recipe.get('needs', []):
        if getattr(args, need) in (None, 'default'):
            sys.exit(f'recipe {args.recipe} needs --{need.replace("_", "-")}')
    error_aware, sched_name = recipe.get('error_aware', False), recipe.get('sched', 'const')
    tau = args.tau if args.recipe in ('lengthweighted', 'gdn2') else None
    llr_dir = args.llr_dir if args.recipe in ('llr', 'gdn2') else None
    synthetic_path = args.synthetic if args.recipe in ('synthetic', 'gdn2') else None
    init_encoder = args.init_encoder if args.recipe in ('pretrain', 'masked', 'gdn2') else None

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    train, val, test = (read_split(os.path.join(args.data, f'{s}.tsv')) for s in ('train', 'val', 'test'))
    if args.max_train and len(train) > args.max_train:
        idx = np.random.default_rng(args.seed).permutation(len(train))[:args.max_train]; train = [train[i] for i in idx]
    print(f'train {len(train)}  val {len(val)}  test {len(test)}')
    table, mean, std = feature_prep(train, seed=args.seed)

    wdir = args.windows_dir or os.path.join(args.data, 'test_windows')
    if not os.path.exists(os.path.join(wdir, 'w300_indel2.tsv')):
        write_test_windows(test, wdir)
    window_sets = {k: read_split(os.path.join(wdir, f'{k}.tsv')) for k in TEST_WINDOW_SETS}
    val_w300 = make_test_windows(val, lengths=(300,), half=False, seed=999)['w300']

    synthetic = read_split(synthetic_path) if synthetic_path else None
    k_syn = min(len(synthetic), int(args.synthetic_frac / (1 - args.synthetic_frac) * len(train))) if synthetic else 0

    def make_loader(epoch):
        rows, w = list(train), [1.0] * len(train)
        if synthetic:
            idx = np.random.default_rng(args.seed * 1000 + epoch).permutation(len(synthetic))[:k_syn]
            rows += [synthetic[i] for i in idx]; w += [args.synthetic_weight] * k_syn
        ds = WeightedDataset(rows, w, table, mean, std, fragment_aware=True, error_aware=error_aware, seed=args.seed * 1000 + epoch)
        return DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          pin_memory=(dev == 'cuda'), worker_init_fn=worker_init if args.workers > 0 else None)

    spec = load_spec(args.arch)
    llr = load_llr(llr_dir, dev) if llr_dir else None
    model = build_two_branch(spec, llr=llr).to(dev)
    if init_encoder:
        ck = torch.load(init_encoder, map_location='cpu')
        if 'spec' in ck and dict(ck['spec']) != dict(spec):
            sys.exit(f'{init_encoder} was pretrained with another sequence network ({spec_name(ck["spec"])}, not {spec_name(spec)})')
        model.seq.encoder.load_state_dict(ck['encoder']); print('sequence network initialised from', init_encoder)
    n_par = count_parameters(model)
    print(f'{spec_name(spec)}: {n_par / 1e6:.3f} M trainable parameters, sequence network {count_macs_reader(spec) / 1e6:.0f} M MACs per {SEQ_LEN} nt')
    ema = EMA(model)
    weighted = LengthWeightedLoss(len(LENGTH_BINS), tau=tau).to(dev) if tau else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max((len(train) + k_syn) // args.batch, 1)
    sched = None
    if sched_name == 'cosine':
        warm = max(1, int(0.05 * total_steps))
        lam = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(args.epochs):
        dl = make_loader(ep)
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for x, m, f, y, g, w in dl:
            x, m, f, y, g, w = (t.to(dev, non_blocking=True) for t in (x, m, f, y, g, w))
            with autocast(dev, args.amp):
                logits = model(x, f, m)
            per = F.cross_entropy(logits.float(), y, reduction='none') * w
            loss = weighted(per, g) if weighted is not None else per.sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); ema.update(model)
            if sched is not None:
                sched.step()
            tot += loss.item(); n += 1
        s, yv = score(ema.shadow, val, table, mean, std, dev, amp=args.amp, workers=args.workers)
        va_full = auroc(yv, s)
        s2, yv2 = score(ema.shadow, val_w300, table, mean, std, dev, amp=args.amp, workers=args.workers)
        va_w300 = auroc(yv2, s2)
        va = 0.5 * (va_full + va_w300)
        print(f'epoch {ep + 1:2d}  loss {tot / max(n, 1):.4f}  validation AUROC complete {100 * va_full:.2f}  300 nt {100 * va_w300:.2f}  ({time.time() - t0:.0f} s)', flush=True)
        history.append({'epoch': ep + 1, 'loss': tot / max(n, 1), 'val_auroc': 100 * va, 'val_auroc_full': 100 * va_full,
                        'val_auroc_w300': 100 * va_w300, 'seconds': time.time() - t0})
        if va > best:
            best, best_state, bad = va, copy.deepcopy(ema.shadow.state_dict_without_llr()), 0
        else:
            bad += 1
            if bad >= args.patience and sched_name == 'const':
                break
    ema.shadow.load_state_dict(best_state, strict=False)

    settings = dict(vars(args), error_aware=error_aware, sched=sched_name, tau=tau, llr_dir=llr_dir, synthetic=synthetic_path, init_encoder=init_encoder)
    rec = {'recipe': args.recipe, 'seed': args.seed, 'settings': settings, 'spec': spec, 'arch_name': spec_name(spec), 'params': n_par,
           'reader_mmac': count_macs_reader(spec) / 1e6, 'val_auroc': 100 * best, 'history': history, 'test': {}}
    s, y = score(ema.shadow, test, table, mean, std, dev, amp=args.amp, workers=args.workers)
    rec['test']['full'] = {'auroc': 100 * auroc(y, s), 'auprc': 100 * auprc(y, s)}
    for k, rows in window_sets.items():
        s, y = score(ema.shadow, rows, table, mean, std, dev, amp=args.amp, workers=args.workers)
        rec['test'][k] = {'auroc': 100 * auroc(y, s), 'auprc': 100 * auprc(y, s)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rec, open(args.out, 'w'), indent=2)
    torch.save({'state_dict': best_state, 'spec': spec, 'llr_dir': llr_dir, 'settings': settings, 'hexamer_table': table,
                'feat_mean': [float(v) for v in mean], 'feat_std': [float(v) for v in std]}, args.out.replace('.json', '.pt'))
    print(json.dumps(rec['test'], indent=1))


if __name__ == '__main__':
    main()
