"""Wall-clock time and peak memory of scoring on a CPU (Table: runtime).

    python scripts/comparisons/runtime.py --data data/extraction1 --models models/genodirnet --out results/genodirnet \
        [--cpc2 tools/CPC2_standalone/bin/CPC2.py] [--threads 1 2]

Scores the complete test transcripts and their 300-nt windows (data/sets/test_w300_seqs.txt) with GenoDirNet
(both networks and the fusion, as predict.py), with each network alone and, when given, with CPC2 standalone.
Every measurement runs in a fresh process; the wall-clock time excludes the Python start-up and the reading of
the sequences, and the memory is the peak resident set size of the process. Results are written to
<out>/runtime.json with the processor name.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
from genodirnet.data import read_fasta, read_seqs, split

NETWORKS = {'genodirnet': None, 'seq_orf_net_np': 'seq_orf_net_np', 'frcgr_k5_np': 'frcgr_k5_np'}


def cpu_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return os.uname().machine


def worker(a):
    """Score one FASTA file with one method and print seconds and peak memory (MB) as JSON."""
    ids, seqs = read_fasta(a.fasta)
    if a.method == 'cpc2':
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            subprocess.run([sys.executable, a.cpc2, '-i', a.fasta, '-o', os.path.join(tmp, 'cpc2')], check=True, capture_output=True)
            sec = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    else:
        sys.path.insert(0, os.path.join(ROOT, 'scripts', 'genodirnet'))
        from genodirnet import orf
        from genodirnet.numpy_nets.models import load, predict
        from score_numpy import BATCH, batches, encoder
        orf_fit = orf.load_fit(os.path.join(a.models, 'orf_features.json'))
        fusion = json.load(open(os.path.join(a.models, 'genodirnet_fusion.json')))
        runs = fusion['networks'] if a.method == 'genodirnet' else [a.method]
        t0 = time.perf_counter()
        z = []
        for run in runs:
            model, name = load(os.path.join(a.models, f'{run}_best.npz')); enc = encoder(name, orf_fit)
            p = predict(model, (batches([enc(s) for s in seqs[i:i + BATCH]]) for i in range(0, len(seqs), BATCH)))
            z.append(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)))
        if a.method == 'genodirnet':
            1 / (1 + np.exp(-(fusion['intercept'] + np.column_stack(z) @ np.array(fusion['coef']))))
        sec = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({'n': len(seqs), 'seconds': sec, 'peak_mb': rss, 'nt': int(sum(len(s) for s in seqs))}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--models', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cpc2', default=''); ap.add_argument('--threads', type=int, nargs='+', default=[1])
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--method', default=''); ap.add_argument('--fasta', default='')
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    sets = {'complete': split(a.data, 'test'), 'w300': read_seqs(os.path.join(a.data, 'sets', 'test_w300_seqs.txt'))}
    results = {'cpu': cpu_name(), 'runs': []}
    methods = list(NETWORKS) + (['cpc2'] if a.cpc2 else [])
    with tempfile.TemporaryDirectory() as tmp:
        for key, (_, seqs) in sets.items():
            fasta = os.path.join(tmp, f'{key}.fa')
            with open(fasta, 'w') as fh:
                for i, s in enumerate(seqs):
                    fh.write(f'>t{i}\n{s}\n')
            for method in methods:
                for th in (a.threads if method != 'cpc2' else [1]):
                    env = dict(os.environ, OMP_NUM_THREADS=str(th), OPENBLAS_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th))
                    cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--data', a.data, '--models', a.models, '--out', a.out,
                           '--method', method, '--fasta', fasta, '--cpc2', a.cpc2]
                    r = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
                    rec = json.loads(r.stdout.strip().splitlines()[-1]); rec.update(set=key, method=method, threads=th)
                    results['runs'].append(rec)
                    print(f"{key:9s} {method:16s} {th} thread(s): {rec['seconds']:8.1f} s, {rec['peak_mb']:7.0f} MB", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(os.path.join(a.out, 'runtime.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
