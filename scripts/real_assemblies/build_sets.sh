#!/usr/bin/env bash
# Real assembled transcripts labelled by the GENCODE release 47 transcript they match (Methods).
#
#   bash scripts/real_assemblies/build_sets.sh [work_dir] [data_dir]
#
# Set 1: LRGASP WTC11 long-read transcript models of IsoTools and the JHU pipeline.
# Set 2: CHESS 3 StringTie2 models assembled from GTEx.
# Both are compared with GENCODE v47 by gffcompare; models on the test chromosomes with class code '=' or 'c' are
# labelled by label_models.py. Writes <data_dir>/real/{wtc11,chess}.{tsv,fa}; scripts/data/homology_check.py then
# lists the models with a close relative among the training transcripts.
# Tools: gffcompare 0.12.10, gffread 0.12.7, bedtools, samtools, seqkit, python with pandas.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(realpath -m "${1:-work/real}"); DATA=$(realpath -m "${2:-data}")
mkdir -p "$WORK" "$DATA/real"; cd "$WORK"

# ---- reference: GENCODE release 47 ------------------------------------------------------------
GENC=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47
[ -f gencode.v47.annotation.gtf ] || { wget -q $GENC/gencode.v47.annotation.gtf.gz && gunzip gencode.v47.annotation.gtf.gz; }
[ -f GRCh38.primary_assembly.genome.fa ] || { wget -q $GENC/GRCh38.primary_assembly.genome.fa.gz && gunzip GRCh38.primary_assembly.genome.fa.gz; }
[ -f GRCh38.primary_assembly.genome.fa.fai ] || samtools faidx GRCh38.primary_assembly.genome.fa
gffread gencode.v47.annotation.gtf --table @chr,transcript_id,gene_id,transcript_type,gene_type | sort -u > ref_biotype.tsv
awk -F'\t' '$3=="CDS"{match($9,/transcript_id "[^"]+"/); t=substr($9,RSTART+15,RLENGTH-16); print $1"\t"$4-1"\t"$5"\t"t"\t.\t"$7}' \
    gencode.v47.annotation.gtf > ref_cds.bed
awk '{n[$4]+=$3-$2} END{for(t in n) print t"\t"n[t]}' ref_cds.bed > ref_cds_len.tsv
gffread -w ref_tx.fa -g GRCh38.primary_assembly.genome.fa gencode.v47.annotation.gtf
seqkit fx2tab -nl ref_tx.fa | awk '{print $1"\t"$NF}' > ref_tx_len.tsv

keep_genome_contigs() { awk 'NR==FNR{k[$1];next} /^#/||($1 in k)' GRCh38.primary_assembly.genome.fa.fai "$1" > "$1.tmp" && mv "$1.tmp" "$1"; }

# ---- set 1: LRGASP WTC11 models ----------------------------------------------------------------
LRGASP=https://cgl.gi.ucsc.edu/data/LRGASP/submissions
for TEAM in IsoTools JHU_pertea; do
  [ -f $TEAM.tar.gz ] || wget -q $LRGASP/$TEAM.tar.gz
  mkdir -p sub_$TEAM && tar xzf $TEAM.tar.gz -C sub_$TEAM
done
i=0
for GTF in $(find sub_* -name 'models.gtf*' | grep -i wtc11 | sort); do
  i=$((i + 1)); G=wtc11_$i.gtf
  if [[ $GTF == *.gz ]]; then zcat "$GTF" > $G; else cp "$GTF" $G; fi
  keep_genome_contigs $G
  gffcompare -r gencode.v47.annotation.gtf -o wtc11_$i $G
  gffread -w wtc11_${i}_tx.fa -g GRCh38.primary_assembly.genome.fa wtc11_$i.annotated.gtf
done

# ---- set 2: CHESS 3 models ---------------------------------------------------------------------
[ -f assembled.gtf ] || { wget -q https://github.com/chess-genome/chess/raw/master/assembled.gtf.gz && gunzip assembled.gtf.gz; }
keep_genome_contigs assembled.gtf
gffcompare -r gencode.v47.annotation.gtf -o chess assembled.gtf
gffread -w chess_tx.fa -g GRCh38.primary_assembly.genome.fa chess.annotated.gtf

# ---- overlap of every model with the annotated coding sequence of each reference ---------------
for SET in wtc11 chess; do
  if [ $SET = wtc11 ]; then GTFS=$(ls wtc11_*.annotated.gtf); else GTFS=chess.annotated.gtf; fi
  cat $GTFS | awk -F'\t' '$3=="exon"{match($9,/transcript_id "[^"]+"/); t=substr($9,RSTART+15,RLENGTH-16); print $1"\t"$4-1"\t"$5"\t"t"\t.\t"$7}' > ${SET}_exons.bed
  bedtools intersect -a ${SET}_exons.bed -b ref_cds.bed -wo -s | awk '{o[$4"\t"$10]+=$NF} END{for(k in o) print k"\t"o[k]}' > ${SET}_cds_overlap.tsv
done

# ---- labels ------------------------------------------------------------------------------------
python3 "$HERE/label_models.py" --set wtc11 --tmap "wtc11_*.wtc11_*.gtf.tmap" --fasta "wtc11_*_tx.fa" --workdir . --out "$DATA/real/wtc11.tsv"
python3 "$HERE/label_models.py" --set chess --tmap chess.assembled.gtf.tmap --fasta chess_tx.fa --workdir . --out "$DATA/real/chess.tsv"
