"""Coding probabilities of the sequences of a FASTA file from a network trained by scripts/genodirnet2/train.py.

    python scripts/genodirnet2/predict.py --model results/genodirnet2/gdn2_s0.pt --fasta data/real/wtc11.fa \
        --out results/real/gdn2_wtc11.tsv

The hexamer table, the feature standardisation, the design specification and, for GenoDirNet-2, the folder of the
autoregressive models are read from the checkpoint. Output: a TSV with the header id, score and one line per sequence
(identifier and coding probability).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import read_fasta
from genodirnet.extended.data import TranscriptDataset
from genodirnet.extended.models import build_two_branch, load_llr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True); ap.add_argument('--fasta', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--llr-dir', default=None, help='folder of the autoregressive models, if not the one stored in the checkpoint')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.model, map_location=dev)
    llr_dir = a.llr_dir or ck.get('llr_dir')
    model = build_two_branch(ck['spec'], llr=load_llr(llr_dir, dev) if llr_dir else None).to(dev)
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    if [k for k in missing if not k.startswith('llr.lm_')] or unexpected:
        sys.exit(f'{a.model} does not match the network it describes')
    model.eval()
    table, mean, std = ck['hexamer_table'], np.array(ck['feat_mean'], dtype=np.float32), np.array(ck['feat_std'], dtype=np.float32)
    ids, seqs = read_fasta(a.fasta)
    rows = [{'id': i, 'label': 0, 'sequence': s.upper()} for i, s in zip(ids, seqs)]
    probs = []
    with torch.no_grad():
        for x, m, f, _, _ in DataLoader(TranscriptDataset(rows, table, mean, std, fragment_aware=False), batch_size=128, num_workers=4):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                logit = model(x.to(dev), f.to(dev), m.to(dev)).float()
            probs.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write('id\tscore\n')
        for i, p in zip(ids, probs):
            fh.write(f'{i}\t{p:.6f}\n')
    print('wrote', a.out, len(ids), 'sequences')


if __name__ == '__main__':
    main()
