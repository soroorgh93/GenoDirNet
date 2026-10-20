"""Fit the logistic fusion of GenoDirNet on the validation chromosomes and store it with the networks.

    python scripts/genodirnet/fuse.py --results results/genodirnet --models models/genodirnet --data data/extraction1 \
        --runs seq_orf_net_np frcgr_k5_np [--name genodirnet]

A logistic regression on the logits of the two networks, fitted on their validation probabilities; it is written to
<models>/<name>_fusion.json together with the coding fraction of the validation chromosomes (the prior of the
calibrated output, used by predict.py --prior). The hexamer table and the standardisation of the ORF features of the
training transcripts are written to <models>/orf_features.json, so that predict.py needs no training data.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet import orf
from genodirnet.data import split


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True); ap.add_argument('--models', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--runs', nargs=2, required=True)
    ap.add_argument('--name', default='genodirnet', help='genodirnet, or genodirnet_absolute for the networks trained with absolute windows')
    a = ap.parse_args()
    yv = np.load(os.path.join(a.results, 'val_labels.npy'))
    x = np.column_stack([logit(np.load(os.path.join(a.results, f'{r}_val_probs.npy'))) for r in a.runs])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(x, yv)
    fusion = dict(networks=a.runs, coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), prior=float(yv.mean()))
    json.dump(fusion, open(os.path.join(a.models, f'{a.name}_fusion.json'), 'w'), indent=1)
    ytr, str_ = split(a.data, 'train'); orf.save_fit(os.path.join(a.models, 'orf_features.json'), orf.fit(str_, ytr))
    print(json.dumps(fusion, indent=1))


if __name__ == '__main__':
    main()
