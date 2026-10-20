"""Fine-tune the Nucleotide Transformer v2 (multispecies, 96 million parameters) on the training transcripts.

    python scripts/comparisons/nucleotide_transformer.py --data data/extraction1 --out results/genodirnet \
        --sets data/extraction1/sets/*_seqs.txt [--no-fat] [--seed 1]

All weights are fine-tuned for two epochs with AdamW (weight decay 0.01), a one-cycle schedule with peak learning
rate 2e-5, effective batch size 16 (2 x 8 accumulation steps), cross-entropy with label smoothing 0.1, gradient
clipping at norm 1 and bfloat16 autocast; inputs are cut to 8,192 nt (at most 2,048 tokens). The epoch with the
highest validation AUROC is kept. Fragment-aware training as for GenoDirNet (window of 50-100 %, at least 200 nt,
probability 0.8). The run nt[_nofat][_seed<s>] writes <out>/<run>.json and <out>/<run>_<key>_probs.npy.
Requires transformers==4.44.2 and a GPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import read_set, set_key, split
from genodirnet.metrics import metrics

MODEL = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
MAX_NT = 8192


class Seqs(Dataset):
    def __init__(self, seqs, y, fat):
        self.seqs, self.y, self.fat = seqs, y, fat

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s = self.seqs[i]
        if self.fat and np.random.rand() < 0.8:
            n = len(s); w = max(200, int(n * np.random.uniform(0.5, 1.0))); st = np.random.randint(0, n - w + 1); s = s[st:st + w]
        return s[:MAX_NT], int(self.y[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--sets', nargs='*', default=[])
    ap.add_argument('--no-fat', action='store_true'); ap.add_argument('--seed', type=int, default=25520)
    ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--batch', type=int, default=2); ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--max-tokens', type=int, default=2048)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = 'nt' + ('_nofat' if a.no_fat else '') + ('' if a.seed == 25520 else f'_seed{a.seed}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, trust_remote_code=True).to(device)
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))

    def collate(items):
        enc = tok([s for s, _ in items], padding=True, truncation=True, max_length=a.max_tokens, return_tensors='pt')
        return enc, torch.tensor([y for _, y in items])

    @torch.no_grad()
    def predict(seqs, bs=8):
        model.eval(); out = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for i in range(0, len(seqs), bs):
                enc = tok([s[:MAX_NT] for s in seqs[i:i + bs]], padding=True, truncation=True, max_length=a.max_tokens, return_tensors='pt').to(device)
                out.append(torch.softmax(model(**enc).logits.float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    ltr = DataLoader(Seqs(str_, ytr, not a.no_fat), a.batch, shuffle=True, collate_fn=collate, drop_last=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(ltr) // a.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, steps), pct_start=0.1, anneal_strategy='cos')
    best, best_state, hist = -1.0, None, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); opt.zero_grad()
        for i, (enc, y) in enumerate(ltr):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = torch.nn.functional.cross_entropy(model(**enc.to(device)).logits.float(), y.to(device), label_smoothing=0.1) / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
                if sched.last_epoch + 1 < steps:
                    sched.step()
        va = roc_auc_score(yva, predict(sva)); hist.append(dict(epoch=ep, val_auroc=float(va), seconds=time.time() - t0))
        print(f'[{run}] epoch {ep}/{a.epochs} val AUROC {va:.4f} ({time.time() - t0:.0f} s)', flush=True)
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = dict(model='nt', run=run, base=MODEL, params=int(sum(p.numel() for p in model.parameters())), fat=not a.no_fat, history=hist)
    for key, seqs, y in [('test', ste, yte), ('val', sva, yva)] + [(set_key(p), *read_set(p)[1::-1]) for p in a.sets]:
        p = predict(seqs); np.save(os.path.join(a.out, f'{run}_{key}_probs.npy'), p)
        if key in ('test', 'val'):
            res[key] = metrics(y, p)
    json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1))


if __name__ == '__main__':
    main()
