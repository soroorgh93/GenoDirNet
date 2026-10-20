"""First architecture search over the design space of the sequence network (extended schedule).

    python scripts/genodirnet2/search_first.py --data data/extraction2 --out results/genodirnet2/search1

Two arms with the same validation protocol. The learned sampler (independent categorical distributions over the
ten design decisions) proposes 64 networks per round; of those within the budget (2.5 million parameters,
800 million multiply-accumulate operations of the sequence network), the 6 with the largest Fisher information at
initialisation among those not validated before are validated, and the sampler is updated by REINFORCE with a moving
baseline (10 rounds). The random arm validates 60 networks drawn uniformly. Validation: two epochs of training on 15,000 training
transcripts (the same subset and batches for every network), objective = mean of the validation AUROC on complete
transcripts and on one 300-nucleotide window per transcript. The network with the highest objective, among both
arms and the default sequence network, is written to <out>/best_arch.json; every candidate is kept in
<out>/history.json, and an interrupted search continues from it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.extended.data import TranscriptDataset, feature_prep, read_split
from genodirnet.extended.models import DEFAULT_SPEC, SPACE, spec_name, vector_to_spec
from genodirnet.extended.search import (PROXIES, ArchitecturePolicy, make_val_sets, short_train, within_budget,
                                        zero_cost_proxies)

KEYS = ('name', 'spec', 'params', 'mmac', 'objective', 'se', 'val_auroc_full', 'val_auroc_w300')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with train.tsv and val.tsv (extraction 2)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=10); ap.add_argument('--n-sample', type=int, default=64); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--random-total', type=int, default=60)
    ap.add_argument('--short-train', type=int, default=15000); ap.add_argument('--short-epochs', type=int, default=2)
    ap.add_argument('--max-params', type=float, default=2.5e6); ap.add_argument('--max-mmac', type=float, default=800)
    ap.add_argument('--policy-lr', type=float, default=0.15); ap.add_argument('--entropy', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train, val = read_split(os.path.join(args.data, 'train.tsv')), read_split(os.path.join(args.data, 'val.tsv'))
    table, mean, std = feature_prep(train, seed=args.seed)
    idx = rng.permutation(len(train))[:args.short_train]; train_sub = [train[i] for i in idx]
    val_full, val_w300 = make_val_sets(val, len(val), args.seed)
    print(f'short training on {len(train_sub)} transcripts; validation on {len(val_full)} complete transcripts and {len(val_w300)} windows')

    batches = []                                     # two fixed mini-batches of training windows for the proxies
    for x, m, f, y, _ in DataLoader(TranscriptDataset(train_sub[:256], table, mean, std, fragment_aware=True, seed=1), batch_size=32):
        batches.append((x, m, f, y))
        if len(batches) == 2:
            break

    policy = ArchitecturePolicy(); popt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    hist_path, pol_path = os.path.join(args.out, 'history.json'), os.path.join(args.out, 'policy.pt')
    history, baseline = (json.load(open(hist_path)) if os.path.exists(hist_path) else []), None
    seen = {h['name']: h for h in history}
    if os.path.exists(pol_path):
        ck = torch.load(pol_path); policy.load_state_dict(ck['state_dict']); baseline = ck['baseline']
    done_rounds = max([h['round'] for h in history if h['arm'] == 'policy'], default=0)

    def evaluate_candidate(spec, vec, arm, rnd):
        ok, p, mm = within_budget(spec, args.max_params, args.max_mmac)
        rec = {'spec': spec, 'vec': vec, 'name': spec_name(spec), 'params': p, 'mmac': mm, 'arm': arm, 'round': rnd, 'within_budget': ok}
        if ok:
            prev = seen.get(rec['name'], {})
            rec.update({k: prev[k] for k in PROXIES} if prev.get('fisher') is not None else zero_cost_proxies(spec, batches, dev))
        return rec

    def validate(rec):
        prev = seen.get(rec['name'], {})
        if 'objective' in prev:
            rec.update({k: prev[k] for k in ('val_auroc_full', 'val_auroc_w300', 'objective', 'se', 'train_seconds')}); return
        r, _ = short_train(rec['spec'], train_sub, val_full, val_w300, table, mean, std, dev, args.short_epochs, args.workers, seed=args.seed)
        rec.update(r)
        print(f"    validated {rec['name']}: objective {r['objective']:.2f} +- {r['se']:.2f}", flush=True)

    def save():
        json.dump(history, open(hist_path, 'w'), indent=1)

    t_start = time.time()
    for rnd in range(done_rounds + 1, args.rounds + 1):
        print(f'round {rnd}/{args.rounds}  ({(time.time() - t_start) / 60:.0f} min)', flush=True)
        specs, vecs = policy.sample(args.n_sample, rng)
        cands, names = [], set()
        for s, v in zip(specs, vecs):
            if spec_name(s) not in names:
                names.add(spec_name(s)); cands.append(evaluate_candidate(s, v, 'policy', rnd))
        inb = [c for c in cands if c['within_budget']]
        if not inb:
            continue
        order = sorted(range(len(inb)), key=lambda i: -inb[i]['fisher'])
        fresh = [inb[i] for i in order if 'objective' not in seen.get(inb[i]['name'], {})]
        old = [inb[i] for i in order if 'objective' in seen.get(inb[i]['name'], {})]
        chosen = (fresh + old)[:args.k]
        for c in chosen:
            validate(c)
        objs = torch.tensor([c['objective'] for c in chosen])
        baseline = objs.mean().item() if baseline is None else 0.7 * baseline + 0.3 * objs.mean().item()
        adv = (objs - baseline) / (objs.std().item() + 1e-3 if len(objs) > 1 else 1.0)
        loss = -sum(a * policy.log_prob(c['vec']) for a, c in zip(adv, chosen)) / len(chosen) - args.entropy * policy.entropy()
        popt.zero_grad(); loss.backward(); popt.step()
        for c in cands:
            seen[c['name']] = c
        history = [h for h in history if not (h['round'] == rnd and h['arm'] == 'policy')] + cands
        save(); torch.save({'state_dict': policy.state_dict(), 'baseline': baseline}, pol_path)

    n_random = sum(1 for h in history if h['arm'] == 'random' and 'objective' in h)
    rr = np.random.default_rng(args.seed + 999)
    while n_random < args.random_total:
        vec = [int(rr.integers(0, len(v))) for v in SPACE.values()]; spec = vector_to_spec(vec)
        if 'objective' in seen.get(spec_name(spec), {}):
            continue
        rec = evaluate_candidate(spec, vec, 'random', 0)
        if not rec['within_budget']:
            continue
        validate(rec); seen[rec['name']] = rec; history.append(rec); n_random += 1
        save()

    ref = evaluate_candidate(dict(DEFAULT_SPEC), [SPACE[k].index(DEFAULT_SPEC[k]) for k in SPACE], 'reference', 0)
    validate(ref); seen[ref['name']] = ref
    if not any(h['name'] == ref['name'] and h['arm'] == 'reference' for h in history):
        history.append(ref)
    save()
    validated, names = [], set()
    for h in history:
        if 'objective' in h and h['arm'] in ('policy', 'random') and h['name'] not in names:
            names.add(h['name']); validated.append(h)
    sel = sorted(validated + [ref], key=lambda h: -h['objective'])[0]
    summary = {'n_validated': len(validated), 'n_validated_policy': sum(h['arm'] == 'policy' for h in validated),
               'n_validated_random': sum(h['arm'] == 'random' for h in validated), 'short_train': len(train_sub),
               'short_epochs': args.short_epochs, 'reference': {k: ref[k] for k in KEYS}, 'selected': dict({k: sel[k] for k in KEYS}, arm=sel['arm'])}
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    json.dump({'spec': sel['spec'], 'name': sel['name'], 'arm': sel['arm'], 'objective': sel['objective'], 'params': sel['params'], 'mmac': sel['mmac']},
              open(os.path.join(args.out, 'best_arch.json'), 'w'), indent=2)
    print(f"selected {sel['name']} ({sel['arm']} arm), objective {sel['objective']:.2f}; default network {ref['objective']:.2f}")


if __name__ == '__main__':
    main()
