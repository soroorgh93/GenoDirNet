"""RefSeq transcripts of human (GRCh38.p14) and mouse (GRCm39) for the transfer to another annotation and species.

    python scripts/data/prepare_refseq.py --raw data/raw --data data/extraction1

Downloads rna.fna.gz and feature_table.txt.gz of each assembly from NCBI (unless present in --raw/<species>) and
keeps the coding (NM_) and lncRNA (NR_) transcripts on chromosomes, one per gene (the longest), of 200 to 20,000 nt
without ambiguous bases (human: without N), in a random order (seed 0). Writes <data>/sets/refseq_<species>_seqs.txt and
refseq_<species>_meta.csv, and for mouse the windows of 300 and 1,000 nt (refseq_mouse_w300, refseq_mouse_w1000).
"""
import argparse
import gzip
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genodirnet.data import write_seqs
from genodirnet.fragments import fixed_windows

NCBI = 'https://ftp.ncbi.nlm.nih.gov/genomes/all/'
ASSEMBLIES = {'human': NCBI + 'GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_',
              'mouse': NCBI + 'GCF/000/001/635/GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_'}
HUMAN_CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y'}


def transcripts(species, raw):
    d = os.path.join(raw, species); os.makedirs(d, exist_ok=True)
    for name in ('rna.fna.gz', 'feature_table.txt.gz'):
        if not os.path.exists(os.path.join(d, name)):
            print('downloading', species, name, flush=True); urllib.request.urlretrieve(ASSEMBLIES[species] + name, os.path.join(d, name))
    info = {}
    with gzip.open(os.path.join(d, 'feature_table.txt.gz'), 'rt') as fh:
        header = fh.readline().lstrip('# ').rstrip('\n').split('\t'); ci = {c: i for i, c in enumerate(header)}
        for line in fh:
            r = line.rstrip('\n').split('\t')
            if len(r) <= max(ci.values()):
                continue
            feat, cls, chrom, acc, sym, gid = (r[ci[k]] for k in ('feature', 'class', 'chromosome', 'product_accession', 'symbol', 'GeneID'))
            if r[ci['seq_type']] != 'chromosome' or not acc:
                continue
            if species == 'human' and (r[ci['assembly_unit']] != 'Primary Assembly' or chrom not in HUMAN_CHROMS):
                continue
            if species == 'mouse' and (not chrom or chrom == 'MT'):
                continue
            if feat == 'mRNA' and acc.startswith('NM_'):
                info[acc] = (1, chrom, sym, gid)
            elif feat == 'ncRNA' and cls == 'lncRNA' and acc.startswith('NR_'):
                info[acc] = (0, chrom, sym, gid)
    best = {}

    def keep(acc, buf):
        if acc in info:
            s = ''.join(buf).upper(); lab, chrom, sym, gid = info[acc]
            clean = ('N' not in s) if species == 'human' else set(s) <= set('ACGT')
            if 200 <= len(s) <= 20000 and clean and (gid not in best or len(s) > len(best[gid][1])):
                best[gid] = (acc, s, lab, chrom, sym, gid)
    with gzip.open(os.path.join(d, 'rna.fna.gz'), 'rt') as fh:
        acc, buf = None, []
        for line in fh:
            if line.startswith('>'):
                keep(acc, buf); acc, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
        keep(acc, buf)
    rows = sorted(best.values()); np.random.default_rng(0).shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='data/raw'); ap.add_argument('--data', default='data/extraction1')
    a = ap.parse_args()
    out = os.path.join(a.data, 'sets'); os.makedirs(out, exist_ok=True)
    for species in ('human', 'mouse'):
        rows = transcripts(species, a.raw); key = f'refseq_{species}'
        y = [r[2] for r in rows]; s = [r[1] for r in rows]
        write_seqs(os.path.join(out, f'{key}_seqs.txt'), y, s)
        with open(os.path.join(out, f'{key}_meta.csv'), 'w') as fh:
            fh.write('accession,gene_id,symbol,chrom,length,label\n')
            for acc, seq, lab, chrom, sym, gid in rows:
                fh.write(f"{acc},{gid},{sym.replace(',', ' ')},chr{chrom},{len(seq)},{lab}\n")
        if species == 'mouse':
            for L in (300, 1000):
                write_seqs(os.path.join(out, f'{key}_w{L}_seqs.txt'), y, fixed_windows(s, L))
        print(f'{key}: {len(rows):,} transcripts, {sum(y):,} coding', flush=True)


if __name__ == '__main__':
    main()
