"""Train a network of the standard schedule with the NumPy implementation (processors only, no PyTorch).

    python scripts/genodirnet/train_numpy.py --model frcgr_k5 --data data/extraction1 --out results/genodirnet \
        --models models/genodirnet [--no-fat] [--windows relative|absolute] [--seed 1]

Networks: frcgr_k5, frcgr_k6, frcgr_k7 (FR-CGR network), frcgr_k5_summed, frcgr_k5_partition, frcgr_k5_permuted
(controls of the input), frcgr_k5_onekernel, frcgr_k5_noneighbour, plaincnn_k5 (variants of the network), seq_net
and seq_orf_net. Settings of the standard schedule: AdamW (weight decay 0.03), learning rate 1e-3 with
two warm-up epochs and cosine decay to 1e-5, batch 64, at most 12 epochs with early stopping on the validation AUROC
(patience 5), label smoothing 0.1, gradient clipping at norm 5, exponential moving average of the weights,
Gaussian noise (s.d. 0.05) on the standardised FR-CGR; auxiliary heads of the FR-CGR network (loss weights 0.2 and
0.1, with distillation from the main head) and of the two branches of the sequence-ORF network (weight 0.3 each).

Fragment-aware training (default): every transcript is replaced with probability 0.8 by a random window, either
covering 50-100 % of its length (--windows relative, the setting of the article) or of 100-1,000 nucleotides
drawn log-uniformly (--windows absolute). --no-fat trains on complete transcripts only.

The run <model>_np[_nofat|_absolute][_seed<s>] writes <out>/<run>.json (settings, history, test metrics),
<out>/<run>_val_probs.npy, <out>/<run>_test_probs.npy, <out>/val_labels.npy, <out>/test_labels.npy and the
checkpoint <models>/<run>_best.npz. Score further sets with scripts/genodirnet/score_numpy.py. The training state is
saved after every epoch (<models>/<run>_state.pkl), so an interrupted run continues from its last complete epoch
when the same command is started again, with the same result.
"""
import argparse
import copy
import json
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet import orf
from genodirnet.data import one_hot, split
from genodirnet.fragments import fat_window
from genodirnet.frcgr import network_input
from genodirnet.metrics import metrics
from genodirnet.numpy_nets import set_threads
from genodirnet.numpy_nets.layers import Linear
from genodirnet.numpy_nets.models import FRCGR_INPUTS, build, predict, softmax

EVAL_BS = 64


class Inputs:
    """Network inputs of a list of transcripts, with the fragment-aware window drawn when an item is requested."""

    def __init__(self, seqs, labels, model, orf_fit=None, train=False, fat='relative', rng=None):
        self.seqs, self.y, self.model, self.orf_fit = seqs, np.asarray(labels), model, orf_fit
        self.train, self.fat, self.rng = train, fat if train else None, rng

    def __len__(self):
        return len(self.y)

    def encode(self, s):
        if self.model in FRCGR_INPUTS:
            k, control = FRCGR_INPUTS[self.model]
            x = network_input(s, k, control).transpose(1, 2, 0)
            if self.train:
                x = x + 0.05 * self.rng.standard_normal(x.shape, dtype=np.float32)
            return x
        x = one_hot(s).T[:, None, :]
        if self.model == 'seq_net':
            return x
        table, mean, std = self.orf_fit
        return x, ((orf.features([s], table)[0] - mean) / std).astype(np.float32)

    def item(self, i):
        s = self.seqs[i]
        if self.fat is not None:
            s = fat_window(s, self.rng, self.fat)
        return self.encode(s)

    def batch(self, idx):
        items = [self.item(i) for i in idx]
        if isinstance(items[0], tuple):
            return (np.ascontiguousarray(np.stack([a for a, _ in items])), np.ascontiguousarray(np.stack([b for _, b in items]))), self.y[idx]
        return np.ascontiguousarray(np.stack(items)), self.y[idx]


def ce_smooth(logits, y, eps, k=2):
    """Label-smoothed cross-entropy (mean over the batch) and its gradient with respect to the logits."""
    p = softmax(logits.astype(np.float64)); q = np.full_like(p, eps / k); q[np.arange(len(y)), y] += 1 - eps
    return float(-(q * np.log(p + 1e-12)).sum(1).mean()), ((p - q) / len(y)).astype(np.float32)


class AdamW:
    def __init__(self, model, lr, wd, betas=(0.9, 0.999), eps=1e-8):
        self.model, self.lr, self.wd, self.b1, self.b2, self.eps = model, lr, wd, betas[0], betas[1], eps
        self.state = {}; self.t = 0

    def params(self):
        for name, m, k in self.model.named_params():
            if k not in m.frozen:
                yield name, m, k

    def step(self):
        self.t += 1
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for name, m, k in self.params():
            g = m.grads.get(k)
            if g is None:
                continue
            p = m.params[k]
            st = self.state.setdefault(name, dict(m=np.zeros_like(p), v=np.zeros_like(p)))
            p *= (1 - self.lr * self.wd)
            st['m'] = self.b1 * st['m'] + (1 - self.b1) * g
            st['v'] = self.b2 * st['v'] + (1 - self.b2) * g * g
            p -= (self.lr * (st['m'] / c1) / (np.sqrt(st['v'] / c2) + self.eps)).astype(np.float32)

    def clip(self, max_norm):
        total = math.sqrt(sum(float((m.grads[k].astype(np.float64) ** 2).sum()) for _, m, k in self.params() if k in m.grads))
        if total > max_norm:
            s = max_norm / (total + 1e-6)
            for _, m, k in self.params():
                if k in m.grads:
                    m.grads[k] = m.grads[k] * s
        return total


def lr_at(step, total, warmup, base, floor):
    """Linear warm-up, then cosine decay from base to floor."""
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    pr = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * pr))


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval(); self.decay = decay; self.updates = 0

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000.0))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            v *= d; v += (1 - d) * src[k]


def generators(model):
    """Random generators held by the layers (dropout and drop path), in a fixed order."""
    out = []
    for m in model.modules():
        g = getattr(m, 'rng', None)
        if isinstance(g, np.random.Generator) and all(g is not h for h in out):
            out.append(g)
    return out


def evaluate(model, data):
    p = predict(model, (data.batch(np.arange(i, min(i + EVAL_BS, len(data))))[0] for i in range(0, len(data), EVAL_BS)))
    return metrics(data.y, p), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--no-fat', action='store_true', help='train on complete transcripts only')
    ap.add_argument('--windows', choices=['relative', 'absolute'], default='relative')
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.03)
    ap.add_argument('--warmup', type=int, default=2); ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9998); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=25520); ap.add_argument('--threads', type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    set_threads(a.threads); os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True)
    run = a.model + '_np' + ('_nofat' if a.no_fat else '_absolute' if a.windows == 'absolute' else '') + ('' if a.seed == 25520 else f'_seed{a.seed}')
    rng = np.random.default_rng(a.seed)
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    orf_fit = orf.fit(str_, ytr) if a.model == 'seq_orf_net' else None
    fat = None if a.no_fat else a.windows
    tr = Inputs(str_, ytr, a.model, orf_fit, train=True, fat=fat, rng=rng)
    va, te = Inputs(sva, yva, a.model, orf_fit), Inputs(ste, yte, a.model, orf_fit)
    steps = len(tr) // a.batch
    model = build(a.model, rng); n_params = model.n_params()
    opt = AdamW(model, a.lr, a.wd); ema = EMA(model, a.ema); step = 0
    hist, best, best_ep, best_state, start = [], -1.0, 0, None, 1
    state_path = os.path.join(a.models, f'{run}_state.pkl')
    if os.path.exists(state_path):
        with open(state_path, 'rb') as fh:
            st = pickle.load(fh)
        model.load_state_dict(st['model']); ema.module.load_state_dict(st['ema']); ema.updates = st['ema_updates']
        for g, state in zip(generators(model), st['generators']):
            g.bit_generator.state = state
        opt.state, opt.t, step = st['opt_state'], st['opt_t'], st['step']
        hist, best, best_ep, best_state, start = st['hist'], st['best'], st['best_ep'], st['best_state'], st['epoch'] + 1
        rng = np.random.default_rng([a.seed, start]); tr.rng = rng
        print(f'[{run}] continuing after epoch {st["epoch"]}', flush=True)
    for ep in range(start, a.epochs + 1):
        model.train(); t0 = time.time(); tl = tc = tn = 0
        order = rng.permutation(len(tr))
        for bi in range(steps):
            x, y = tr.batch(order[bi * a.batch:(bi + 1) * a.batch])
            out = model.forward(x)
            if a.model == 'seq_orf_net':
                main_l, hs, ho = out
                loss, dmain = ce_smooth(main_l, y, a.label_smoothing)
                ls, ds = ce_smooth(hs, y, a.label_smoothing); lo, do = ce_smooth(ho, y, a.label_smoothing)
                loss += 0.3 * ls + 0.3 * lo
                model.zero_grad(); model.backward(dmain, 0.3 * ds, 0.3 * do)
            elif isinstance(out, tuple):
                main_l, a3, a4 = out
                loss, dmain = ce_smooth(main_l, y, a.label_smoothing)
                l3, d3 = ce_smooth(a3, y, a.label_smoothing); l4, d4 = ce_smooth(a4, y, a.label_smoothing)
                loss += 0.2 * l3 + 0.1 * l4; d3 *= 0.2; d4 *= 0.1
                t = 2.0; teacher = softmax(main_l.astype(np.float64) / t)
                for aux, dd in ((a3, d3), (a4, d4)):
                    pa = softmax(aux.astype(np.float64) / t)
                    loss += 0.08 * t * t * float((teacher * (np.log(teacher + 1e-12) - np.log(pa + 1e-12))).sum(1).mean())
                    dd += (0.08 * t * (pa - teacher) / len(y)).astype(np.float32)
                model.zero_grad(); model.backward(dmain, d3, d4)
            else:
                main_l = out
                loss, dmain = ce_smooth(main_l, y, a.label_smoothing)
                model.zero_grad(); model.backward(dmain)
            opt.clip(5.0); opt.lr = lr_at(step, a.epochs * steps, a.warmup * steps, a.lr, 1e-5); opt.step(); step += 1
            ema.update(model); model.free()
            tl += loss * len(y); tc += int((main_l.argmax(1) == y).sum()); tn += len(y)
        vm, _ = evaluate(ema.module, va)
        hist.append(dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn, val_auroc=vm['auroc'], seconds=time.time() - t0))
        print(f"[{run}] epoch {ep}/{a.epochs} loss {tl / tn:.3f} val AUROC {vm['auroc']:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if vm['auroc'] > best:
            best, best_ep = vm['auroc'], ep; best_state = {k: v.copy() for k, v in ema.module.state_dict().items()}
        elif ep - best_ep >= a.patience:
            print(f'early stop at epoch {ep} (best epoch {best_ep})', flush=True); break
        rng = np.random.default_rng([a.seed, ep + 1]); tr.rng = rng
        with open(state_path + '.tmp', 'wb') as fh:
            pickle.dump(dict(epoch=ep, model=model.state_dict(), ema=ema.module.state_dict(), ema_updates=ema.updates,
                             opt_state=opt.state, opt_t=opt.t, step=step, hist=hist, best=best, best_ep=best_ep,
                             best_state=best_state, generators=[g.bit_generator.state for g in generators(model)]), fh)
        os.replace(state_path + '.tmp', state_path)
    ema.module.load_state_dict(best_state)
    np.savez(os.path.join(a.models, f'{run}_best.npz'), state=best_state, model=a.model, epoch=best_ep)
    test, pt = evaluate(ema.module, te); val, pv = evaluate(ema.module, va)
    for name, arr in ((f'{run}_test_probs.npy', pt), (f'{run}_val_probs.npy', pv), ('test_labels.npy', yte), ('val_labels.npy', yva)):
        np.save(os.path.join(a.out, name), arr)
    x = te.batch(np.arange(1))[0]; ema.module.eval(); ema.module.forward(x, aux=False)
    macs = int(sum(getattr(m, 'macs', 0) for m in ema.module.modules()) + sum(m.params['w'].size for m in ema.module.modules() if isinstance(m, Linear)))
    res = dict(model=a.model, run=run, seed=a.seed, fat=fat, params=n_params, macs=macs, best_epoch=best_ep, epochs=len(hist),
               framework='numpy', val=val, test=test, history=hist)
    if a.model in FRCGR_INPUTS and a.model != 'plaincnn_k5':
        res['deploy_params'] = ema.module.deploy_params()
    json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1))


if __name__ == '__main__':
    main()
