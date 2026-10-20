"""Train a network of the standard schedule with PyTorch on a GPU.

    python scripts/genodirnet/train_torch.py --model frcgr_k5 --data data/extraction1 --out results/genodirnet \
        --models models/genodirnet --sets data/extraction1/sets/test_w300_seqs.txt [--seed 1] [--no-fat]

Networks: frcgr_k5, frcgr_k6, frcgr_k7 (FR-CGR network), resnet18_k7 (ResNet18 on the FR-CGR), seq_net and
seq_orf_net. Settings of the standard schedule: AdamW (betas 0.9, 0.999; weight decay 0.03), learning rate
1e-3 with two warm-up epochs and cosine decay to 1e-5, batch 64, at most 12 epochs with early stopping on the
validation AUROC (patience 5), cross-entropy with label smoothing 0.1, gradient clipping at norm 5, mixed precision,
exponential moving average of the weights for validation, selection and testing. The FR-CGR network adds Gaussian
noise (s.d. 0.05) to the standardised input and auxiliary heads on stages 3 and 4 (loss weights 0.2 and 0.1) that
are also distilled from the main head (temperature 2, weight 0.08); the sequence-ORF network adds the losses of
linear heads on its two branches (weight 0.3 each).

Fragment-aware training (default): with probability 0.8 a transcript is replaced by a random window covering 50-100 %
of its length (at least 200 nt); --no-fat trains on complete transcripts only.

The run <model>[_nofat][_seed<s>] writes <out>/<run>.json, <out>/<run>_{val,test}_probs.npy, the labels and
<out>/<run>_<key>_probs.npy for every set given with --sets, and the weights <models>/<run>_best.pt.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet import orf
from genodirnet.data import one_hot, read_set, set_key, split
from genodirnet.frcgr import network_input
from genodirnet.metrics import metrics
from genodirnet.torch_nets import FRCGRNet, build, count_macs, count_parameters
from genodirnet.training import ModelEMA, WarmupCosineSchedule, seed_everything


class Transcripts(Dataset):
    """Network input of every transcript, with the fragment-aware window drawn when an item is requested."""

    def __init__(self, seqs, labels, model, orf_fit=None, train=False, fat=True):
        self.seqs, self.y, self.model, self.orf_fit, self.train, self.fat = seqs, np.asarray(labels), model, orf_fit, train, fat and train
        self.k = int(model.split('_k')[-1]) if '_k' in model else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s = self.seqs[i]
        if self.fat and np.random.rand() < 0.8:
            n = len(s); w = max(200, int(n * np.random.uniform(0.5, 1.0))); st = np.random.randint(0, n - w + 1); s = s[st:st + w]
        y = int(self.y[i])
        if self.k is not None:
            x = torch.from_numpy(network_input(s, self.k))
            if self.train:
                x = x + 0.05 * torch.randn_like(x)
            return x, y
        x = torch.from_numpy(one_hot(s))
        if self.model == 'seq_orf_net':
            table, mean, std = self.orf_fit
            return x, torch.from_numpy(((orf.features([s], table)[0] - mean) / std).astype(np.float32)), y
        return x, y


def forward(net, batch, device):
    *xs, y = batch
    out = net(*[x.to(device, non_blocking=True) for x in xs])
    return out, y.to(device, non_blocking=True)


def main_logits(out):
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def predict(net, loader, device):
    net.eval(); probs = []
    for batch in loader:
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            out, _ = forward(net, batch, device)
        probs.append(torch.softmax(main_logits(out).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def loss_of(out, y, model, smoothing):
    if model == 'seq_orf_net':
        fused, heads = out
        loss = F.cross_entropy(fused.float(), y, label_smoothing=smoothing)
        for v in heads.values():
            loss = loss + 0.3 * F.cross_entropy(v.float(), y, label_smoothing=smoothing)
        return loss, fused
    if isinstance(out, tuple):
        main, a3, a4 = out
        loss = F.cross_entropy(main, y, label_smoothing=smoothing) + 0.2 * F.cross_entropy(a3, y, label_smoothing=smoothing) \
            + 0.1 * F.cross_entropy(a4, y, label_smoothing=smoothing)
        t = 2.0; teacher = torch.softmax(main.detach().float() / t, 1)
        for aux in (a3, a4):
            loss = loss + 0.08 * t * t * F.kl_div(torch.log_softmax(aux.float() / t, 1), teacher, reduction='batchmean')
        return loss, main
    return F.cross_entropy(out, y, label_smoothing=smoothing), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--sets', nargs='*', default=[]); ap.add_argument('--no-fat', action='store_true')
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.03)
    ap.add_argument('--warmup', type=int, default=2); ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9998); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=25520); ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 2))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True); seed_everything(a.seed)
    run = a.model + ('_nofat' if a.no_fat else '') + ('' if a.seed == 25520 else f'_seed{a.seed}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    orf_fit = orf.fit(str_, ytr) if a.model == 'seq_orf_net' else None
    mk = lambda s, y, train=False: Transcripts(s, y, a.model, orf_fit, train, not a.no_fat)
    ltr = DataLoader(mk(str_, ytr, True), a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    ev = lambda s, y: DataLoader(mk(s, y), 128, num_workers=a.workers)
    lva = ev(sva, yva)
    net = build(a.model).to(device)
    n_params = count_parameters(net)
    steps = a.epochs * len(ltr)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.wd)
    sched = WarmupCosineSchedule(opt, steps, min(a.warmup * len(ltr), steps // 4), a.lr, 1e-5)
    ema = ModelEMA(net, a.ema)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    hist, best, best_ep, best_state = [], -1.0, 0, None
    for ep in range(1, a.epochs + 1):
        net.train(); t0 = time.time(); tl = tn = 0
        for batch in ltr:
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                out, y = forward(net, batch, device)
                loss, _ = loss_of(out, y, a.model, a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(net)
            tl += float(loss.detach()) * len(y); tn += len(y)
        va = metrics(yva, predict(ema.module, lva, device))['auroc']
        hist.append(dict(epoch=ep, train_loss=tl / tn, val_auroc=va, seconds=time.time() - t0))
        print(f'[{run}] epoch {ep}/{a.epochs} loss {tl / tn:.3f} val AUROC {va:.4f} ({time.time() - t0:.0f} s)', flush=True)
        if va > best:
            best, best_ep, best_state = va, ep, copy.deepcopy(ema.module.state_dict())
        elif ep - best_ep >= a.patience:
            print(f'early stop at epoch {ep} (best epoch {best_ep})', flush=True); break
    ema.module.load_state_dict(best_state); best_net = ema.module.eval()
    torch.save({'state': best_state, 'model': a.model, 'epoch': best_ep}, os.path.join(a.models, f'{run}_best.pt'))
    pt, pv = predict(best_net, ev(ste, yte), device), predict(best_net, lva, device)
    for name, arr in ((f'{run}_test_probs.npy', pt), (f'{run}_val_probs.npy', pv), ('test_labels.npy', yte), ('val_labels.npy', yva)):
        np.save(os.path.join(a.out, name), arr)
    for path in a.sets:
        yk, sk, _ = read_set(path)
        np.save(os.path.join(a.out, f'{run}_{set_key(path)}_probs.npy'), predict(best_net, ev(sk, np.zeros(len(sk)) if yk is None else yk), device))
    sample = next(iter(ev(ste[:4], yte[:4])))
    inputs = [x.to(device) for x in sample[:-1]]
    infer = best_net.deployment_copy() if isinstance(best_net, FRCGRNet) else best_net
    res = dict(model=a.model, run=run, seed=a.seed, fat=not a.no_fat, params=n_params, macs=count_macs(infer, *inputs), best_epoch=best_ep,
               epochs=len(hist), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', val=metrics(yva, pv), test=metrics(yte, pt), history=hist)
    json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1))


if __name__ == '__main__':
    main()
