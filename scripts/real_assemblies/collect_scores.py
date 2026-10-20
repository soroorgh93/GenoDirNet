"""One table of scores per set of real assembled transcripts: results/real/scores_<set>.tsv.

    python scripts/real_assemblies/collect_scores.py --data data --results results

Columns: id, label, truncated, frac_covered, orf_status, class_code and one score per model (higher = more likely
coding):
  genodirnet                  logistic fusion of seq_orf_net_np and frcgr_k5_np, fitted on their validation logits
  genodirnet_absolute         the same for the networks trained with absolute windows (*_np_absolute)
  gdn2                        GenoDirNet-2, scripts/genodirnet2/predict.py -> results/real/gdn2_<set>.tsv
  rnasamba*, cpat*, cpc2*     results/genodirnet/<model>_<set>_probs.npy (released and retrained models)
  lncfinder, lncdc            outputs of run_tools.sh
  transdecoder                log-likelihood score of the best ORF; sequences without an ORF get the lowest score - 1
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FUSED = ('seq_orf_net_np', 'frcgr_k5_np')
FUSED_ABSOLUTE = ('seq_orf_net_np_absolute', 'frcgr_k5_np_absolute')
NPY = ('rnasamba', 'rnasamba_partial', 'rnasamba_retrained', 'cpat', 'cpc2', 'cpat_retrained', 'cpc2_retrained')


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fused(folder, key, runs=FUSED):
    yv = np.load(os.path.join(folder, 'val_labels.npy'))
    xv = np.column_stack([logit(np.load(os.path.join(folder, f'{r}_val_probs.npy'))) for r in runs])
    x = np.column_stack([logit(np.load(os.path.join(folder, f'{r}_{key}_probs.npy'))) for r in runs])
    return LogisticRegression(C=1.0, max_iter=2000).fit(xv, yv).predict_proba(x)[:, 1]


def read_lncdc(path):
    t = pd.read_csv(path)
    idcol = [c for c in t.columns if c.lower() in ('description', 'transcript', 'id', 'name', 'transcript_id')][0]
    low = {c.lower(): c for c in t.columns}
    return dict(zip(t[idcol], 1.0 - t[low['noncoding_prob']])) if 'noncoding_prob' in low else dict(zip(t[idcol], t[low['coding_prob']]))


def read_transdecoder(folder):
    t = pd.read_csv(glob.glob(os.path.join(folder, '**', 'longest_orfs.cds.scores'), recursive=True)[0], sep='\t')
    acc = '#acc' if '#acc' in t.columns else t.columns[0]
    score = [c for c in t.columns if c.lower().startswith('score')][0]
    t['seq'] = t[acc].astype(str).str.rsplit('.p', n=1).str[0]
    return t.groupby('seq')[score].max().to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--results', required=True)
    a = ap.parse_args()
    G, R = os.path.join(a.results, 'genodirnet'), os.path.join(a.results, 'real')
    for key in ('wtc11', 'chess'):
        base = pd.read_csv(os.path.join(a.data, 'real', f'{key}.tsv'), sep='\t')
        out = base[['id', 'label', 'truncated', 'frac_covered', 'orf_status', 'class_code']].copy()
        out['genodirnet'] = fused(G, key)
        out['genodirnet_absolute'] = fused(G, key, FUSED_ABSOLUTE)
        t = pd.read_csv(os.path.join(R, f'gdn2_{key}.tsv'), sep='\t')
        out['gdn2'] = out.id.map(dict(zip(t.id, t.score)))
        for m in NPY:
            f = os.path.join(G, f'{m}_{key}_probs.npy')
            if os.path.exists(f):
                out[m] = np.load(f)
        tools = os.path.join(R, 'tools', key)
        t = pd.read_csv(os.path.join(tools, 'lncfinder.tsv'), sep='\t'); out['lncfinder'] = out.id.map(dict(zip(t.id, t.score)))
        out['lncdc'] = out.id.map(read_lncdc(os.path.join(tools, 'lncdc.csv')))
        td = read_transdecoder(os.path.join(tools, 'td')); floor = min(td.values()) - 1.0
        out['transdecoder'] = out.id.map(lambda i: td.get(i, floor))
        out.to_csv(os.path.join(R, f'scores_{key}.tsv'), sep='\t', index=False)
        print(key, len(out), 'models; columns:', ', '.join(out.columns[6:]))


if __name__ == '__main__':
    main()
