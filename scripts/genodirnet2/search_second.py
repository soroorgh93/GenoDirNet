"""Second architecture search: Gaussian-process surrogate over the design graph and the Fisher geometry.

    python scripts/genodirnet2/search_second.py --data data/extraction2 --first results/genodirnet2/search1 \
        --out results/genodirnet2/search2

1. Every network validated by the first search receives the quantities of genodirnet.extended.search
   (proxies at initialisation, Fisher geometry on 160 probe windows at initialisation and after 50 training steps).
2. Two arms propose networks by expected improvement of a Gaussian process (GraphGP) on these quantities and the
   design decisions, from candidate pools of 32 (uniform draws and one- or two-decision changes of the five best):
   bo_s0 starts from 12 random networks and runs 8 rounds of 6; bo_warm runs 4 rounds of 6 with a surrogate fitted
   on every network validated so far, including those of the first search. Validation as in the first search.
3. The five best networks are validated again with a second training seed.
4. The four with the highest two-seed mean are trained under the full extended schedule with two seeds each
   (scripts/genodirnet2/train.py --recipe so --arch ...), and the one with the highest mean validation objective is
   written to <out>/best_arch.json; with --selected-out PREFIX its two runs are also copied to PREFIX_s0.json and
   PREFIX_s1.json (with their checkpoints).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.extended.data import TranscriptDataset, feature_prep, read_split
from genodirnet.extended.models import DEFAULT_SPEC, SPACE, spec_name, spec_to_vector, vector_to_spec
from genodirnet.extended.search import (DEC, FEATS, PROXIES, GraphGP, build_context, expected_improvement, gaussianize,
                                        info_proxies, log, make_val_sets, short_train, within_budget, zero_cost_proxies)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with train.tsv and val.tsv (extraction 2)')
    ap.add_argument('--first', required=True, help='output folder of search_first.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', type=int, default=12); ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--pool', type=int, default=32); ap.add_argument('--n-probe', type=int, default=160); ap.add_argument('--warm-steps', type=int, default=50)
    ap.add_argument('--warm-rounds', type=int, default=4); ap.add_argument('--recheck', type=int, default=5)
    ap.add_argument('--finalists', type=int, default=4)
    ap.add_argument('--short-train', type=int, default=15000); ap.add_argument('--short-epochs', type=int, default=2)
    ap.add_argument('--max-params', type=float, default=2.5e6); ap.add_argument('--max-mmac', type=float, default=800)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--selected-out', default=None)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    train, val = read_split(os.path.join(args.data, 'train.tsv')), read_split(os.path.join(args.data, 'val.tsv'))
    table, mean, std = feature_prep(train, seed=0)
    sub_idx = np.random.default_rng(0).permutation(len(train))[:args.short_train]
    train_sub = [train[i] for i in sub_idx]
    val_full, val_w300 = make_val_sets(val, len(val), 0)
    batches = []
    for x, m, f, y, _ in DataLoader(TranscriptDataset(train_sub[:256], table, mean, std, fragment_aware=True, seed=1), batch_size=32):
        batches.append((x, m, f, y))
        if len(batches) == 2:
            break
    ctx = build_context(train, sub_idx, table, mean, std, args.n_probe, args.warm_steps, dev)
    log(f'{len(ctx.items)} probe windows, {len(ctx.warm)} warm-up batches')

    p_rec, p_arm = os.path.join(args.out, 'records.json'), os.path.join(args.out, 'arms.json')
    records = json.load(open(p_rec)) if os.path.exists(p_rec) else {}
    arms = json.load(open(p_arm)) if os.path.exists(p_arm) else {}

    def save():
        json.dump(records, open(p_rec + '.tmp', 'w')); os.replace(p_rec + '.tmp', p_rec)
        json.dump(arms, open(p_arm, 'w'), indent=1)

    if 'first_random' not in arms:                   # networks validated by the first search
        for h in json.load(open(os.path.join(args.first, 'history.json'))):
            if 'objective' not in h:
                continue
            r = records.setdefault(h['name'], {})
            for k in ['spec', 'vec', 'name', 'params', 'mmac', 'within_budget', 'val_auroc_full', 'val_auroc_w300', 'objective', 'se'] + PROXIES:
                if k in h:
                    r[k] = h[k]
            arm = {'policy': 'first_policy', 'random': 'first_random', 'reference': 'reference'}[h['arm']]
            if h['name'] not in arms.setdefault(arm, []):
                arms[arm].append(h['name'])
        save()

    def ensure_proxies(spec):
        name = spec_name(spec)
        r = records.setdefault(name, {'spec': spec, 'vec': spec_to_vector(spec), 'name': name})
        if 'within_budget' not in r or 'mmac' not in r:
            ok, p, mm = within_budget(spec, args.max_params, args.max_mmac)
            r.update(within_budget=ok, params=p, mmac=mm)
        if not r['within_budget']:
            return r
        if r.get('fisher') is None:
            r.update(zero_cost_proxies(spec, batches, dev))
        if 'fk_ngd' not in r:
            r.update(info_proxies(spec, ctx))
        return r

    def validate(name, seed=0):
        r = records[name]
        key = 'objective' if seed == 0 else f'objective_s{seed}'
        if key not in r:
            rec, _ = short_train(r['spec'], train_sub, val_full, val_w300, table, mean, std, dev, args.short_epochs, args.workers, seed=seed)
            r.update(rec if seed == 0 else {f'{k}_s{seed}': v for k, v in rec.items()})
            log(f'    validated {name} (training seed {seed}): objective {rec["objective"]:.2f} +- {rec["se"]:.2f}')
        return r

    for n in [n for n in records if 'objective' in records[n] and 'fk_ngd' not in records[n]]:
        ensure_proxies(records[n]['spec']); save()

    def features(rs):
        C = np.array([r['vec'] for r in rs], float)
        Q = gaussianize(np.array([[r.get(k, np.nan) for k in FEATS] for r in rs], float))
        return C, Q

    def sample_uniform(rng):
        return vector_to_spec([int(rng.integers(0, len(v))) for v in SPACE.values()])

    def mutate(spec, rng):
        vec = spec_to_vector(spec)
        for d in rng.choice(len(DEC), size=int(rng.integers(1, 3)), replace=False):
            vec[d] = int(rng.choice([i for i in range(len(SPACE[DEC[d]])) if i != vec[d]]))
        return vector_to_spec(vec)

    t0 = time.time()
    for arm, seed, n_init, n_rounds, use_all in (('bo_s0', 0, args.init, args.rounds, False), ('bo_warm', 99, 0, args.warm_rounds, True)):
        seq = arms.setdefault(arm, [])
        rng = np.random.default_rng(1000 + seed)
        tries = 0
        while len(seq) < n_init and tries < 10000:
            tries += 1
            s = sample_uniform(rng); n = spec_name(s)
            if n in seq:
                continue
            if not ensure_proxies(s)['within_budget']:
                continue
            validate(n); seq.append(n); save()
        for rnd in range(max(1, (len(seq) - n_init) // args.k + 1), n_rounds + 1):
            rr = np.random.default_rng(10000 * seed + rnd)
            if use_all:
                obs = [records[n] for n in sorted({n for v in arms.values() for n in v if 'objective' in records[n]})]
            else:
                obs = [records[n] for n in seq]
            ranked = sorted(obs, key=lambda r: -r['objective'])
            cand, names, tries = [], ({r['name'] for r in obs} | set(seq)), 0
            while len(cand) < args.pool and tries < 20000:
                tries += 1
                s = mutate(ranked[int(rr.integers(0, min(5, len(ranked))))]['spec'], rr) if len(cand) % 2 else sample_uniform(rr)
                n = spec_name(s)
                if n in names:
                    continue
                names.add(n)
                r = ensure_proxies(s)
                if r['within_budget']:
                    cand.append(r)
            save()
            C, Q = features(obs + cand); no = len(obs)
            y = np.array([r['objective'] for r in obs]); nv = np.array([r['se'] ** 2 for r in obs])
            gp = GraphGP(len(DEC), len(FEATS)).fit(C[:no], Q[:no], y, nv, restarts=3, seed=rnd)
            best = float(np.max(gp.predict(C[:no], Q[:no])[0]))
            chosen, avail, g = [], list(range(len(cand))), gp
            for _ in range(min(args.k, len(cand))):          # batch by kriging believer
                mu, var = g.predict(C[no:][avail], Q[no:][avail])
                j = avail[int(np.argmax(expected_improvement(mu, var, best)))]
                chosen.append(j)
                g = g.condition(C[no + j], Q[no + j], float(mu[avail.index(j)]), float(nv.mean()))
                avail.remove(j)
            log(f'{arm} round {rnd}/{n_rounds}: ' + ', '.join(cand[j]['name'] for j in chosen))
            for j in chosen:
                validate(cand[j]['name']); seq.append(cand[j]['name']); save()
            log(f'   best in {arm}: {max(records[n]["objective"] for n in seq):.2f}  ({(time.time() - t0) / 60:.0f} min)')

    ref = spec_name(DEFAULT_SPEC)
    if not arms.get('reference'):
        ensure_proxies(dict(DEFAULT_SPEC)); validate(ref); arms['reference'] = [ref]; save()
    ref = arms['reference'][0]
    validated = sorted({n for v in arms.values() for n in v if 'objective' in records[n]})
    top = [r['name'] for r in sorted([records[n] for n in validated if n != ref], key=lambda r: -r['objective'])[:args.recheck]]
    for n in top + [ref]:
        validate(n, seed=1); save()

    def mean2(n):
        return 0.5 * (records[n]['objective'] + records[n]['objective_s1'])
    finalists = sorted(top, key=lambda n: -mean2(n))[:args.finalists]
    full = {}
    for i, n in enumerate(finalists):
        spec_path = os.path.join(args.out, f'finalist{i}.json')
        json.dump({'spec': records[n]['spec'], 'name': n}, open(spec_path, 'w'))
        vals = []
        for s in (0, 1):
            out = os.path.join(args.out, f'finalist{i}_s{s}.json')
            if not os.path.exists(out):
                log('full schedule:', n, 'seed', s)
                subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'), '--data', args.data,
                                '--recipe', 'so', '--arch', spec_path, '--seed', str(s), '--workers', str(args.workers), '--out', out], check=True)
            if os.path.exists(out):
                vals.append(json.load(open(out))['val_auroc'])
        if len(vals) == 2:
            full[n] = float(np.mean(vals))
    sel = max(full, key=full.get)
    summary = {'n_validated': len(validated), 'arms': {a: len(v) for a, v in arms.items()},
               'recheck': {n: {'objective_seed0': records[n]['objective'], 'objective_seed1': records[n]['objective_s1'], 'mean': mean2(n)} for n in top + [ref]},
               'full_schedule_validation': full, 'selected': sel}
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    r = records[sel]
    json.dump({'spec': r['spec'], 'name': sel, 'params': r['params'], 'mmac': r['mmac'], 'val_auroc_full_schedule': full[sel]},
              open(os.path.join(args.out, 'best_arch.json'), 'w'), indent=2)
    if args.selected_out:
        i = finalists.index(sel)
        for s in (0, 1):
            for ext in ('.json', '.pt'):
                shutil.copy(os.path.join(args.out, f'finalist{i}_s{s}{ext}'), f'{args.selected_out}_s{s}{ext}')
    log(f'selected {sel} (mean validation objective {full[sel]:.2f} under the full schedule)')


if __name__ == '__main__':
    main()
