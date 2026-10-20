#!/usr/bin/env bash
# LncFinder, LncDC and TransDecoder on the real assembled transcripts (the other models are scored by the scripts
# listed in README.md). Writes results/real/tools/<set>/{lncfinder.tsv,lncdc.csv,td/}.
#
#   bash scripts/real_assemblies/run_tools.sh <data_dir> <results_dir> <LncDC folder> [threads]
#
# LncFinder 1.1.6 (R; human model, without secondary-structure features), LncDC 1.3.6 (github.com/lim74/LncDC),
# TransDecoder 6.0.0 (best ORF of at least 50 codons, partial ORFs allowed).
set -euo pipefail
DATA=$(realpath "$1"); RES=$(realpath -m "$2"); LNCDC=$(realpath "$3"); T=${4:-8}
for SET in wtc11 chess; do
  FA=$DATA/real/$SET.fa; OUT=$RES/real/tools/$SET; mkdir -p "$OUT"
  Rscript -e '
    suppressMessages({library(LncFinder); library(seqinr)})
    a <- commandArgs(TRUE); s <- read.fasta(a[1], seqtype = "DNA", as.string = FALSE)
    r <- lnc_finder(s, SS.features = FALSE, format = "DNA", frequencies.file = "human", svm.model = "human",
                    parallel.cores = as.integer(a[3]))
    write.table(data.frame(id = rownames(r), score = r$Coding.Potential), a[2], sep = "\t", quote = FALSE, row.names = FALSE)' \
    "$FA" "$OUT/lncfinder.tsv" "$T"
  python "$LNCDC/bin/lncDC.py" -i "$FA" -o "$OUT/lncdc.csv" -t "$T"
  (cd "$OUT" && TransDecoder -t "$FA" -m 50 --single_best_only -O td)
done
