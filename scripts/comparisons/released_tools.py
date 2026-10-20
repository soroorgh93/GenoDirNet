"""CPC2 and CPAT (released human models) on the validation and test transcripts and on every evaluation set.

    python scripts/comparisons/released_tools.py --data data/extraction1 --out results/genodirnet \
        --cpc2 CPC2_standalone/bin/CPC2.py --cpat-hexamer Human_Hexamer.tsv --cpat-logit Human_logitModel.RData \
        --sets data/extraction1/sets/*_seqs.txt

CPC2 standalone 1.0.1 (github.com/gao-lab/CPC2_standalone; run python3 after the Python 3 changes made by
--patch-cpc2) and CPAT 3.0.5 (the cpat command, with --top-orf=1) write their coding probability for every sequence;
a sequence without a score (no ORF) gets probability 0. RNAsamba is scored by score_rnasamba.py.
Writes <out>/<tool>_<key>_probs.npy for the keys val, test and every set.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import read_set, set_key, split


def patch_cpc2(cpc2):
    """Python 3 syntax for the Python 2 scripts of CPC2 standalone 1.0.1."""
    for name in ('CPC2.py', 'seqio.py', 'compress.py'):
        path = os.path.join(os.path.dirname(cpc2), name)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        new = src.replace('import commands', 'import subprocess').replace('commands.getstatusoutput', 'subprocess.getstatusoutput')
        new = re.sub(r'(\b[\w\.]+)\.next\(\)', r'next(\1)', new)
        new = re.sub(r'^(\s*)print (?!\()(.*)$', r'\1print(\2)', new, flags=re.M)
        new = new.replace('.has_key(', '.__contains__(')
        new = re.sub(r'(?<![\w\.])file\(', 'open(', new)
        if new != src:
            open(path, 'w').write(new)


def cpc2(a, fasta, prefix):
    subprocess.run([sys.executable, a.cpc2, '-i', fasta, '-o', prefix], check=True, capture_output=True)
    prob = {}
    for line in open(prefix + '.txt'):
        if not line.startswith('#'):
            f = line.rstrip('\n').split('\t'); prob[f[0]] = float(f[6])
    return prob


def cpat(a, fasta, prefix):
    subprocess.run(['cpat', '-x', a.cpat_hexamer, '-d', a.cpat_logit, '-g', fasta, '-o', prefix, '--top-orf=1'], check=True, capture_output=True)
    prob = {}
    with open(prefix + '.ORF_prob.best.tsv') as fh:
        c = fh.readline().rstrip('\n').split('\t').index('Coding_prob')
        for line in fh:
            f = line.rstrip('\n').split('\t'); prob[f[0].split('_ORF')[0].upper()] = float(f[c])
    return prob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--sets', nargs='*', default=[])
    ap.add_argument('--cpc2', default=''); ap.add_argument('--cpat-hexamer', default=''); ap.add_argument('--cpat-logit', default='')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.cpc2:
        patch_cpc2(a.cpc2)
    inputs = {s: split(a.data, s)[1] for s in ('val', 'test')}
    inputs.update({set_key(p): read_set(p)[1] for p in a.sets})
    tools = [(n, f) for n, f, ok in (('cpc2', cpc2, a.cpc2), ('cpat', cpat, a.cpat_hexamer)) if ok]
    with tempfile.TemporaryDirectory() as tmp:
        for key, seqs in inputs.items():
            names = [f'S{i}' for i in range(len(seqs))]
            fasta = os.path.join(tmp, f'{key}.fa')
            with open(fasta, 'w') as fh:
                for n, s in zip(names, seqs):
                    fh.write(f'>{n}\n{s}\n')
            for tool, fn in tools:
                out = os.path.join(a.out, f'{tool}_{key}_probs.npy')
                if not os.path.exists(out):
                    prob = fn(a, fasta, os.path.join(tmp, f'{tool}_{key}'))
                    np.save(out, np.array([prob.get(n.upper() if tool == 'cpat' else n, 0.0) for n in names]))
                    print(tool, key, flush=True)


if __name__ == '__main__':
    main()
