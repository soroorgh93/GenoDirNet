"""CPAT retrained with its own features on the training transcripts of the article.

    python scripts/comparisons/retrain_cpat.py --data data/extraction1 --out results/genodirnet \
        --models models/cpat --sets data/extraction1/sets/*_seqs.txt data/real/wtc11.fa data/real/chess.fa [--no-fat]

The hexamer table is built as by make_hexamer_tab of CPAT 3.0.5 (in-frame hexamers of the annotated coding sequences
of the coding training transcripts, <data>/train_cds.fa, against hexamers at every position of the training
lncRNAs). Each sequence is described by the four features of CPAT, computed with its cpmodule package: the sequence
length, the length of the longest ORF (ATG to stop on the forward strand, longer than 75 nucleotides), and the
Fickett and hexamer scores of that ORF. The logistic regression of CPAT (no penalty) is fitted on the training set
of the retrained released tools: every transcript once complete and twice as a random window (fragments.retraining_set),
or once complete with --no-fat. A sequence without an ORF gets probability 0, as in CPAT.
Writes <out>/cpat_retrained[_nofat]_{val,test,<set>}_probs.npy, <out>/cpat_retrained[_nofat].json and the fitted model
with its hexamer table, <models>/cpat_retrained[_nofat].pkl; a stored model is reused, and sets already scored are skipped.
"""
import argparse
import json
import os
import pickle
import sys
import tempfile

import numpy as np
from cpmodule import FrameKmer, fickett, find_orfs
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import read_fasta, read_set, set_key, split
from genodirnet.fragments import retraining_set
from genodirnet.metrics import metrics

START, STOP, MIN_ORF = ['ATG'], ['TAG', 'TAA', 'TGA'], 75


def hexamer_table(cds, lncrna):
    """Hexamer frequencies of make_hexamer_tab: (coding, noncoding) dictionaries."""
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, seqs in (('cds.fa', cds), ('lnc.fa', lncrna)):
            files.append(os.path.join(tmp, name))
            with open(files[-1], 'w') as fh:
                for i, s in enumerate(seqs):
                    fh.write(f'>s{i}\n{s}\n')
        cod = FrameKmer.kmer_freq_file(fastafile=files[0], word_size=6, step_size=3, frame=0)
        non = FrameKmer.kmer_freq_file(fastafile=files[1], word_size=6, step_size=1, frame=0)
    cs, ns = float(sum(cod.values())), float(sum(non.values()))
    coding = {k: cod[k] / cs for k in cod if 'N' not in k}
    noncoding = {k: non[k] / ns for k in cod if 'N' not in k}
    return coding, noncoding


def features(seq, coding, noncoding):
    """mRNA, ORF, Fickett and Hexamer of CPAT for one sequence, or None without an ORF."""
    seq = seq.upper()
    orfs = find_orfs.ORFFinder(seq=seq, min_orf=MIN_ORF).orf_candidates(antisense=False, n_candidate=1, start_coden=START, stop_coden=STOP)
    if not orfs:
        return None
    orf = orfs[0][-1]
    return [len(seq), orfs[0][4], fickett.fickett_value(orf), FrameKmer.kmer_ratio(orf, 6, 3, coding, noncoding)]


def predict(model, seqs, coding, noncoding):
    f = [features(s, coding, noncoding) for s in seqs]
    ok = np.array([x is not None for x in f]); p = np.zeros(len(seqs))
    if ok.any():
        p[ok] = model.predict_proba(np.array([x for x in f if x is not None], dtype=float))[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--sets', nargs='*', default=[])
    ap.add_argument('--models', required=True, help='folder for the fitted model'); ap.add_argument('--no-fat', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True)
    run = 'cpat_retrained' + ('_nofat' if a.no_fat else '')
    (ytr, str_), (yva, sva), (yte, ste) = (split(a.data, s) for s in ('train', 'val', 'test'))
    model_path = os.path.join(a.models, f'{run}.pkl')
    if os.path.exists(model_path):
        model, coding, noncoding, n_train = pickle.load(open(model_path, 'rb'))
    else:
        coding, noncoding = hexamer_table(read_fasta(os.path.join(a.data, 'train_cds.fa'))[1], [s for s, y in zip(str_, ytr) if y == 0])
        X, y = [], []
        for lab, _, s in retraining_set(ytr, str_, copies=0 if a.no_fat else 2):
            f = features(s, coding, noncoding)
            if f is not None:
                X.append(f); y.append(lab)
        model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=10000)).fit(np.array(X, dtype=float), np.array(y))
        n_train = len(y); pickle.dump((model, coding, noncoding, n_train), open(model_path, 'wb'))
    res = dict(model='cpat_retrained', run=run, fat=not a.no_fat, n_train=n_train)
    for key, seqs, yy in (('val', sva, yva), ('test', ste, yte)):
        p = predict(model, seqs, coding, noncoding); res[key] = metrics(yy, p)
        np.save(os.path.join(a.out, f'{run}_{key}_probs.npy'), p)
    for path in a.sets:
        f = os.path.join(a.out, f'{run}_{set_key(path)}_probs.npy')
        if not os.path.exists(f):
            np.save(f, predict(model, read_set(path)[1], coding, noncoding)); print(run, set_key(path), flush=True)
    json.dump(res, open(os.path.join(a.out, f'{run}.json'), 'w'), indent=1)
    print(run, f"test AUROC {100 * res['test']['auroc']:.2f}")


if __name__ == '__main__':
    main()
