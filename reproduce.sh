#!/usr/bin/env bash
# Every experiment of the article, in order. Each step can be run on its own (see README.md); a GPU is needed for the
# steps marked [GPU], CPU threads for the others. Variables: THREADS (CPU threads), TOOLS (folder of the released tools).
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${THREADS:-8}; TOOLS=${TOOLS:-tools}
D=data; E1=$D/extraction1; E2=$D/extraction2; G=results/genodirnet; G2=results/genodirnet2; M=models/genodirnet
mkdir -p $G $G2 results/real models

# ---- 1. data ---------------------------------------------------------------------------------------------------
python scripts/data/prepare_gencode.py --raw $D/raw --out $D
python scripts/data/prepare_refseq.py --raw $D/raw --data $E1
python scripts/data/make_sets.py --data $E1 --draws 10
python scripts/data/homology_screen.py --data $E1
bash scripts/real_assemblies/build_sets.sh work/real $D
python scripts/data/homology_check.py --data $D
SETS="$(ls $E1/sets/*_seqs.txt) $D/real/wtc11.fa $D/real/chess.fa"

# ---- 2. standard schedule, PyTorch [GPU] ------------------------------------------------------------------------
for spec in "frcgr_k5 25520 1" "frcgr_k6 25520 1" "frcgr_k7 25520 1 2" "resnet18_k7 25520 1" "seq_net 25520 1" "seq_orf_net 25520 1"; do
  set -- $spec; m=$1; shift
  for s in "$@"; do python scripts/genodirnet/train_torch.py --model $m --seed $s --data $E1 --out $G --models $M --sets $SETS; done
done

# ---- 3. standard schedule, NumPy (paired comparison of fragment-aware training, window distributions, controls) ----
np_train() { python scripts/genodirnet/train_numpy.py --data $E1 --out $G --models $M --threads $THREADS "$@"; }
for s in 25520 1 2; do np_train --model frcgr_k5 --seed $s; done
for s in 25520 1; do np_train --model frcgr_k5 --no-fat --seed $s; done
np_train --model seq_net; np_train --model seq_net --no-fat
for s in 25520 1 2; do np_train --model seq_orf_net --seed $s; done
np_train --model seq_orf_net --no-fat
for m in frcgr_k5 seq_orf_net; do np_train --model $m --windows absolute; done
for m in frcgr_k5_summed frcgr_k5_partition frcgr_k5_permuted frcgr_k5_onekernel frcgr_k5_noneighbour plaincnn_k5; do
  for s in 25520 1 2; do np_train --model $m --seed $s; done
done
RUNS=$(ls $M/*_np*_best.npz | xargs -n1 basename | sed 's/_best.npz//')
python scripts/genodirnet/score_numpy.py --data $E1 --models $M --out $G --runs $RUNS --sets $SETS --threads $THREADS
python scripts/genodirnet/fuse.py --results $G --models $M --data $E1 --runs seq_orf_net_np frcgr_k5_np
python scripts/genodirnet/fuse.py --results $G --models $M --data $E1 --runs seq_orf_net_np_absolute frcgr_k5_np_absolute \
    --name genodirnet_absolute

# ---- 4. compared methods ----------------------------------------------------------------------------------------
WIN=$(for w in 100 200 300 500 1000; do echo $E1/sets/test_w${w}_seqs.txt; done)
python scripts/comparisons/feature_models.py --data $E1 --out $G --sets $SETS
python scripts/comparisons/feature_models.py --data $E1 --out $G --sets $SETS --no-fat --models wordfreq_lr framemlp_k5 orf_lr
python scripts/comparisons/feature_models.py --data $E1 --out $G --sets $SETS --windows absolute --models wordfreq_lr framemlp_k5 orf_lr
python scripts/comparisons/markov_chain.py --data $E1 --out $G --sets $SETS
python scripts/comparisons/markov_chain.py --data $E1 --out $G --sets $WIN --order 4
python scripts/comparisons/released_tools.py --data $E1 --out $G --sets $SETS --cpc2 $TOOLS/CPC2_standalone/bin/CPC2.py \
    --cpat-hexamer $TOOLS/Human_Hexamer.tsv --cpat-logit $TOOLS/Human_logitModel.RData
for fat in "" "--no-fat"; do
  python scripts/comparisons/retrain_rnasamba.py --data $E1 --models models/rnasamba $fat
  python scripts/comparisons/retrain_cpat.py --data $E1 --out $G --models models/cpat --sets $SETS $fat
  python scripts/comparisons/retrain_cpc2.py --data $E1 --out $G --models models/cpc2 --cpc2 $TOOLS/CPC2_standalone/bin/CPC2.py --sets $SETS $fat
done
python scripts/comparisons/score_rnasamba.py --out $G --sets $E1/val_seqs.txt $E1/test_seqs.txt $SETS \
    --model rnasamba=$TOOLS/RNAsamba/data/full_length_weights.hdf5 \
    rnasamba_partial=$TOOLS/RNAsamba/data/partial_length_weights.hdf5 rnasamba_retrained=models/rnasamba/rnasamba_retrained.hdf5 \
    rnasamba_retrained_nofat=models/rnasamba/rnasamba_retrained_nofat.hdf5
NT="python scripts/comparisons/nucleotide_transformer.py --data $E1 --out $G"                                    # [GPU]
$NT --sets $WIN $E1/sets/test_w300i?_seqs.txt $E1/sets/test_w1000i?_seqs.txt $E1/sets/refseq_*_seqs.txt
$NT --seed 1 --sets $WIN
$NT --no-fat --sets $WIN $E1/sets/refseq_human_seqs.txt

# ---- 5. extended schedule: GenoDirNet-SO, single additions, GenoDirNet-2 [GPU] ----------------------------------
T="python scripts/genodirnet2/train.py --data $E2"
python scripts/genodirnet2/search_first.py --data $E2 --out $G2/search1
for c in coding noncoding; do python scripts/genodirnet2/autoregressive.py --mode train --class $c --data $E2 --out $G2/lm; done
python scripts/genodirnet2/autoregressive.py --mode score --data $E2 --out $G2/lm
python scripts/genodirnet2/pretrain.py --method rnd --data $E2 --out $G2/encoder_rnd.pt
python scripts/genodirnet2/pretrain.py --method masked --data $E2 --out $G2/encoder_masked.pt
python scripts/genodirnet2/pretrain.py --method rnd --data $E2 --arch $G2/search1/best_arch.json --out $G2/encoder_rnd_deeper.pt
for s in 0 1; do
  $T --recipe so --seed $s --out $G2/so_s$s.json
  $T --recipe llr --llr-dir $G2/lm --seed $s --out $G2/llr_s$s.json
  $T --recipe erroraware --seed $s --out $G2/erroraware_s$s.json
  $T --recipe pretrain --init-encoder $G2/encoder_rnd.pt --seed $s --out $G2/pretrain_s$s.json
  $T --recipe masked --init-encoder $G2/encoder_masked.pt --seed $s --out $G2/masked_s$s.json
  $T --recipe deeper --arch $G2/search1/best_arch.json --seed $s --out $G2/deeper_s$s.json
  for tau in 0.02 0.05 0.1; do $T --recipe lengthweighted --tau $tau --seed $s --out $G2/lengthweighted_tau${tau}_s$s.json; done
done
TAU=$(python - $G2 <<'EOF'
import json, sys
val = {t: sum(json.load(open(f'{sys.argv[1]}/lengthweighted_tau{t}_s{s}.json'))['val_auroc'] for s in (0, 1)) for t in ('0.02', '0.05', '0.1')}
print(max(val, key=val.get))                         # temperature with the highest mean validation AUROC
EOF
)
python scripts/genodirnet2/autoregressive.py --mode sample --data $E2 --out $G2/lm --filter-model $G2/so_s0.pt
GDN2="--arch $G2/search1/best_arch.json --init-encoder $G2/encoder_rnd_deeper.pt --llr-dir $G2/lm --tau $TAU"
for s in 0 1; do
  $T --recipe synthetic --synthetic $G2/lm/synthetic.tsv --seed $s --out $G2/synthetic_s$s.json
  $T --recipe gdn2 $GDN2 --synthetic $G2/lm/synthetic.tsv --seed $s --out $G2/gdn2_s$s.json
done
python scripts/genodirnet2/search_second.py --data $E2 --first $G2/search1 --out $G2/search2 --selected-out $G2/compact

# ---- 6. real assembled transcripts -------------------------------------------------------------------------------
for set in wtc11 chess; do
  python scripts/genodirnet2/predict.py --model $G2/gdn2_s0.pt --fasta $D/real/$set.fa --out results/real/gdn2_$set.tsv
done
bash scripts/real_assemblies/run_tools.sh $D results $TOOLS/LncDC $THREADS
python scripts/real_assemblies/collect_scores.py --data $D --results results

# ---- 7. scoring time, tables, numbers and figures ---------------------------------------------------------------
python scripts/comparisons/runtime.py --data $E1 --models $M --out $G --cpc2 $TOOLS/CPC2_standalone/bin/CPC2.py --threads 1
python analysis/make_tables.py --results results --data $D --out paper
python analysis/make_figures.py --results results --data $D --out paper/figures --tables paper/tables
