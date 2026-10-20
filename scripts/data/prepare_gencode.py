"""The data sets of the article from GENCODE release 47 (Methods: Data sets).

    python scripts/data/prepare_gencode.py --raw data/raw --out data

Downloads gencode.v47.pc_transcripts.fa.gz, gencode.v47.lncRNA_transcripts.fa.gz and gencode.v47.annotation.gtf.gz
into --raw (about 1.3 GB) unless they are present. The transcripts of the splits are listed, in order, in
splits/extraction1.tsv.gz and splits/extraction2.tsv.gz: one transcript per gene (the longest), 200 to 20,000 nt, no
ambiguous bases, no mitochondrial transcripts; test chromosomes 4, 9, 14, 19 and 21, validation chromosomes 6, 12
and 18, all others for training, with the training lncRNAs downsampled to the number of coding transcripts.

Writes
  <out>/extraction1/{train,val,test}_seqs.txt   label, tab, sequence (standard schedule: GenoDirNet and comparisons)
  <out>/extraction1/{train,val,test}_meta.csv   gene, transcript, chrom, name, length, label
  <out>/extraction1/test_cds.csv                transcript, length, cds_length of the coding test transcripts
  <out>/extraction1/train_cds.fa                annotated coding sequences of the coding training transcripts (CPAT)
  <out>/extraction2/{train,val,test}.tsv        id, label, chrom, sequence, cds_start, cds_end (extended schedule)
  <out>/extraction2/train_transcripts.fa        training transcripts (homology screen of the real assemblies)
"""
import argparse
import gzip
import os
import re
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
URL = 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/'
FILES = ('gencode.v47.pc_transcripts.fa.gz', 'gencode.v47.lncRNA_transcripts.fa.gz', 'gencode.v47.annotation.gtf.gz')


def read_fasta_gz(path):
    name, buf = None, []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, ''.join(buf)


def chromosomes(gtf):
    chrom, pat = {}, re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(gtf, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split('\t', 9)
            if f[2] == 'transcript':
                m = pat.search(f[8])
                if m:
                    chrom[m.group(1)] = f[0]
    return chrom


def read_split_table(path):
    rows = {'train': [], 'val': [], 'test': []}
    with gzip.open(path, 'rt') as fh:
        next(fh)
        for line in fh:
            s, t, lab = line.rstrip('\n').split('\t'); rows[s].append((t, int(lab)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='data/raw'); ap.add_argument('--out', default='data')
    a = ap.parse_args(); os.makedirs(a.raw, exist_ok=True)
    for f in FILES:
        if not os.path.exists(os.path.join(a.raw, f)):
            print('downloading', f, flush=True); urllib.request.urlretrieve(URL + f, os.path.join(a.raw, f))
    chrom = chromosomes(os.path.join(a.raw, FILES[2]))
    rec = {}                                       # versioned transcript id -> record
    for path, label in ((FILES[0], 1), (FILES[1], 0)):
        for header, seq in read_fasta_gz(os.path.join(a.raw, path)):
            f = header.split('|')
            cds = next((x[4:] for x in f if x.startswith('CDS:')), '')
            cs, ce = cds.split('-') if cds else ('', '')
            rec[f[0]] = dict(id=f[0], gene=f[1].split('.')[0], name=f[5] if len(f) > 5 else '', chrom=chrom.get(f[0], 'NA'),
                             label=label, seq=seq, cds_start=cs, cds_end=ce)
    unversioned = {k.split('.')[0]: v for k, v in rec.items()}

    out1 = os.path.join(a.out, 'extraction1'); os.makedirs(out1, exist_ok=True)
    for split, rows in read_split_table(os.path.join(ROOT, 'splits', 'extraction1.tsv.gz')).items():
        rs = [unversioned[t] for t, _ in rows]
        assert all(r['label'] == lab for r, (_, lab) in zip(rs, rows)), f'label mismatch in {split}'
        with open(os.path.join(out1, f'{split}_seqs.txt'), 'w') as fs, open(os.path.join(out1, f'{split}_meta.csv'), 'w') as fm:
            fm.write('gene,transcript,chrom,name,length,label\n')
            for r in rs:
                fs.write(f"{r['label']}\t{r['seq']}\n")
                fm.write(f"{r['gene']},{r['id'].split('.')[0]},{r['chrom']},{r['name']},{len(r['seq'])},{r['label']}\n")
        if split == 'train':
            with open(os.path.join(out1, 'train_cds.fa'), 'w') as fc:
                for r in rs:
                    if r['label'] == 1 and r['cds_start']:
                        fc.write(f">{r['id'].split('.')[0]}\n{r['seq'][int(r['cds_start']) - 1:int(r['cds_end'])].upper()}\n")
        if split == 'test':
            with open(os.path.join(out1, 'test_cds.csv'), 'w') as fc:
                fc.write('transcript,length,cds_length\n')
                for r in rs:
                    if r['label'] == 1 and r['cds_start']:
                        fc.write(f"{r['id'].split('.')[0]},{len(r['seq'])},{int(r['cds_end']) - int(r['cds_start']) + 1}\n")
        print(f'extraction 1, {split}: {len(rs):,} transcripts', flush=True)

    out2 = os.path.join(a.out, 'extraction2'); os.makedirs(out2, exist_ok=True)
    for split, rows in read_split_table(os.path.join(ROOT, 'splits', 'extraction2.tsv.gz')).items():
        rs = [rec[t] for t, _ in rows]
        with open(os.path.join(out2, f'{split}.tsv'), 'w') as fh:
            fh.write('id\tlabel\tchrom\tsequence\tcds_start\tcds_end\n')
            for r in rs:
                fh.write(f"{r['id']}\t{r['label']}\t{r['chrom']}\t{r['seq'].upper()}\t{r['cds_start']}\t{r['cds_end']}\n")
        if split == 'train':
            with open(os.path.join(out2, 'train_transcripts.fa'), 'w') as fh:
                for r in rs:
                    fh.write(f">{r['id']}\n{r['seq'].upper()}\n")
        print(f'extraction 2, {split}: {len(rs):,} transcripts', flush=True)


if __name__ == '__main__':
    main()
