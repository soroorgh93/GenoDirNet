"""Label assembled transcript models by the biotype of the GENCODE transcript they match (gffcompare).

    python scripts/real_assemblies/label_models.py --set chess --tmap work/chess.assembled.gtf.tmap \
        --fasta work/chess_tx.fa --workdir work --out data/real/chess.tsv

Keeps models of class code '=' (complete match of the intron chain) or 'c' (contained in a longer reference
transcript; truncated) on the test chromosomes whose matched reference is protein coding with an annotated coding
sequence (label 1) or a lncRNA (label 0), at least 200 nucleotides long; identical sequences are kept once. For each
model, frac_covered is its length over that of the reference, and orf_status records whether the model contains
all, part or none of the annotated coding sequence of its reference (complete, partial, absent; noncoding for
lncRNAs). Reads ref_biotype.tsv, ref_tx_len.tsv, ref_cds_len.tsv and <set>_cds_overlap.tsv of --workdir and writes
the TSV file and the FASTA file (same name, .fa) of the set.
"""
import argparse
import glob
import os

import pandas as pd

TEST_CHR = ['chr4', 'chr9', 'chr14', 'chr19', 'chr21']


def read_fasta_dict(pattern):
    d = {}
    for path in glob.glob(pattern):
        name, buf = None, []
        with open(path) as fh:
            for line in fh:
                if line.startswith('>'):
                    if name:
                        d[name] = ''.join(buf)
                    name, buf = line[1:].split()[0], []
                else:
                    buf.append(line.strip().upper())
        if name:
            d[name] = ''.join(buf)
    return d


def write_set(df, out):
    df.to_csv(out, sep='\t', index=False)
    with open(out[:-4] + '.fa', 'w') as fh:
        for r in df.itertuples():
            fh.write(f'>{r.id}\n{r.sequence}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', choices=['wtc11', 'chess'], required=True); ap.add_argument('--tmap', required=True)
    ap.add_argument('--fasta', required=True); ap.add_argument('--workdir', default='.'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    w = a.workdir
    bio = pd.read_csv(os.path.join(w, 'ref_biotype.tsv'), sep='\t', header=None, names=['chrom', 'tid', 'gid', 'ttype', 'gtype']).drop_duplicates('tid')
    tlen = pd.read_csv(os.path.join(w, 'ref_tx_len.tsv'), sep='\t', header=None, names=['tid', 'ref_len'])
    clen = pd.read_csv(os.path.join(w, 'ref_cds_len.tsv'), sep='\t', header=None, names=['tid', 'cds_len'])
    ref = bio.merge(tlen, on='tid', how='left').merge(clen, on='tid', how='left')
    ref['tid_base'] = ref.tid.str.split('.').str[0]

    d = pd.concat([pd.read_csv(p, sep='\t') for p in sorted(glob.glob(a.tmap))])
    d = d[d.class_code.isin(['=', 'c'])].rename(columns={'len': 'qlen'})
    d['truncated'] = (d.class_code == 'c').astype(int)
    df = d[['qry_id', 'ref_id', 'class_code', 'truncated', 'qlen']].copy()
    df['ref_id'] = df.ref_id.astype(str).str.split('.').str[0].str.split('|').str[0]
    df = df.merge(ref, left_on='ref_id', right_on='tid_base', how='left')
    df = df[df.chrom.isin(TEST_CHR)]
    df['label'] = df.ttype.map(lambda t: 1 if t == 'protein_coding' else (0 if t == 'lncRNA' else None))
    df = df.dropna(subset=['label']); df['label'] = df.label.astype(int)
    df = df[(df.label == 0) | (df.cds_len.fillna(0) > 0)]

    ov = pd.read_csv(os.path.join(w, f'{a.set}_cds_overlap.tsv'), sep='\t', header=None, names=['qry_id', 'tid', 'cds_ov'])
    df = df.merge(ov, on=['qry_id', 'tid'], how='left')
    df['cds_ov'] = df.cds_ov.fillna(0)
    df['frac_covered'] = (df.qlen / df.ref_len).clip(upper=1.0)

    def status(r):
        if r.label == 0:
            return 'noncoding'
        if r.cds_ov >= r.cds_len:
            return 'complete'
        return 'absent' if r.cds_ov == 0 else 'partial'
    df['orf_status'] = df.apply(status, axis=1)

    df['sequence'] = df.qry_id.map(read_fasta_dict(a.fasta))
    df = df.dropna(subset=['sequence'])
    df = df[df.sequence.str.len() >= 200].drop_duplicates('sequence')
    out = df.rename(columns={'qry_id': 'id'})[['id', 'label', 'chrom', 'sequence', 'ref_id', 'class_code', 'frac_covered', 'orf_status', 'truncated']]
    write_set(out, a.out)
    print(f'{a.set}: {len(out)} models ({out.truncated.sum()} truncated; {out.label.sum()} coding, {(out.label == 0).sum()} lncRNA)')


if __name__ == '__main__':
    main()
