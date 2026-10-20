"""CPC2 retrained with its own features on the training transcripts of the article.

    python scripts/comparisons/retrain_cpc2.py --data data/extraction1 --out results/genodirnet \
        --cpc2 CPC2_standalone/bin/CPC2.py --models models/cpc2 --sets data/extraction1/sets/*_seqs.txt data/real/wtc11.fa [--no-fat]

The four features of CPC2 (peptide length, Fickett score, isoelectric point and integrity of the longest ORF) are
computed by CPC2 standalone 1.0.1 itself (scripts/comparisons/released_tools.py applies its Python 3 changes). As in
CPC2, the features are scaled to [-1, 1] with the range of the training set and classified by a support vector
machine with a radial basis kernel (gamma 0.125, the value of the released model); the cost C is chosen among 0.25,
1, 4 and 16 by the validation AUROC on complete transcripts. Coding probabilities are the decision values after a
logistic (Platt) calibration on the validation transcripts. Training set: every transcript once complete and twice as
a random window (fragments.retraining_set), or once complete with --no-fat.
Writes <out>/cpc2_retrained[_nofat]_{val,test,<set>}_probs.npy, <out>/cpc2_retrained[_nofat].json and the fitted model
<models>/cpc2_retrained[_nofat].pkl; a stored model is reused, and sets already scored are skipped.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..')); sys.path.insert(0, HERE)
from genodirnet.data import read_set, set_key, split
from genodirnet.fragments import retraining_set
from genodirnet.metrics import metrics
from released_tools import patch_cpc2

COSTS = (0.25, 1.0, 4.0, 16.0)


def cpc2_features(cpc2, seqs, tmp, tag):
    """Peptide length, Fickett score, pI and ORF integrity of every sequence, as written by CPC2."""
    fasta, prefix = os.path.join(tmp, f'{tag}.fa'), os.path.join(tmp, f'{tag}_cpc2')
    with open(fasta, 'w') as fh:
        for i, s in enumerate(seqs):
            fh.write(f'>S{i}\n{s}\n')
    subprocess.run([sys.executable, cpc2, '-i', fasta, '-o', prefix], check=True, capture_output=True)
    feats = {}
    for line in open(prefix + '.txt'):
        if not line.startswith('#'):
            f = line.rstrip('\n').split('\t'); feats[f[0]] = [float(f[2]), float(f[3]), float(f[4]), float(f[5])]
    return np.array([feats[f'S{i}'] for i in range(len(seqs))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--cpc2', required=True)
    ap.add_argument('--models', required=True, help='folder for the fitted model')
    ap.add_argument('--sets', nargs='*', default=[]); ap.add_argument('--no-fat', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True); patch_cpc2(a.cpc2)
    run = 'cpc2_retrained' + ('_nofat' if a.no_fat else '')
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    model_path = os.path.join(a.models, f'{run}.pkl')
    with tempfile.TemporaryDirectory() as tmp:
        if os.path.exists(model_path):
            fit = pickle.load(open(model_path, 'rb'))
        else:
            train = retraining_set(ytr, str_, copies=0 if a.no_fat else 2)
            Xtr = cpc2_features(a.cpc2, [s for _, _, s in train], tmp, 'train'); y = np.array([lab for lab, _, _ in train])
            lo, hi = Xtr.min(0), Xtr.max(0); span = np.where(hi > lo, hi - lo, 1.0)
            Xva = -1.0 + 2.0 * (cpc2_features(a.cpc2, sva, tmp, 'val') - lo) / span
            fits = {}
            for c in COSTS:
                svm = SVC(kernel='rbf', gamma=0.125, C=c, cache_size=1000).fit(-1.0 + 2.0 * (Xtr - lo) / span, y)
                fits[c] = (roc_auc_score(yva, svm.decision_function(Xva)), svm)
                print(f'C = {c}: validation AUROC {100 * fits[c][0]:.2f}', flush=True)
            c = max(fits, key=lambda k: fits[k][0]); svm = fits[c][1]
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(svm.decision_function(Xva)[:, None], yva)
            fit = dict(lo=lo, span=span, svm=svm, platt=platt, cost=c, n_train=len(y), validation_auroc={str(k): v[0] for k, v in fits.items()})
            pickle.dump(fit, open(model_path, 'wb'))
        prob = lambda X: fit['platt'].predict_proba(fit['svm'].decision_function(-1.0 + 2.0 * (X - fit['lo']) / fit['span'])[:, None])[:, 1]
        res = dict(model='cpc2_retrained', run=run, fat=not a.no_fat, n_train=fit['n_train'], cost=fit['cost'],
                   validation_auroc=fit['validation_auroc'], support_vectors=int(fit['svm'].n_support_.sum()))
        for key, seqs, yy in (('val', sva, yva), ('test', ste, yte)):
            p = prob(cpc2_features(a.cpc2, seqs, tmp, key)); res[key] = metrics(yy, p)
            np.save(os.path.join(a.out, f'{run}_{key}_probs.npy'), p)
        for path in a.sets:
            key = set_key(path); f = os.path.join(a.out, f'{run}_{key}_probs.npy')
            if not os.path.exists(f):
                np.save(f, prob(cpc2_features(a.cpc2, read_set(path)[1], tmp, key))); print(run, key, flush=True)
    json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
    print(run, f"C = {res['cost']}, test AUROC {100 * res['test']['auroc']:.2f}")


if __name__ == '__main__':
    main()
