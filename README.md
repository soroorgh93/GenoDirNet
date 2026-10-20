# GenoDirNet

Coding potential prediction for complete and fragmented transcripts.

GenoDirNet classifies a transcript, or a fragment of one, as protein-coding or long noncoding RNA. It fuses a
network on the frame-resolved chaos game representation (FR-CGR) of the sequence with a network on the nucleotide
sequence and five open reading frame features, and it is trained on random windows of transcripts
(fragment-aware training). GenoDirNet-2 adds likelihood-ratio features of two autoregressive models and
error-aware training for inputs with insertions and deletions.

> S. Ghandali. GenoDirNet: fragment-aware training for coding potential prediction on complete and
> fragmented transcripts. *NAR Genomics and Bioinformatics* (submitted).

## Installation

Linux or macOS, Python 3.10 or later and a C compiler:

    pip install -r requirements.txt

PyTorch 2.3 or later and a CUDA-capable GPU are needed only to train the PyTorch networks, GenoDirNet-2
(`scripts/genodirnet2`) and the Nucleotide Transformer. The compared tools (CPC2, CPAT, RNAsamba, LncFinder,
LncDC, TransDecoder) are installed separately.

## Prediction

    python scripts/genodirnet/predict.py --fasta transcripts.fa --models models/genodirnet --out predictions.tsv

Input: a FASTA file. Output: a tab-separated table with the identifier and the coding probability of each
sequence. The trained networks in `models/genodirnet` run on one processor core.

| Option | Effect |
| --- | --- |
| `--prior 0.2` | shifts the calibrated probability to the coding fraction expected in the input |
| `--fdr 0.05` | adds a call column: sequences are called coding while the posterior expected false discovery rate stays below the level |
| `--fusion genodirnet_absolute` | networks trained with absolute windows of 100 to 1,000 nucleotides, for inputs of a few hundred nucleotides |

## Reproducing the article

`reproduce.sh` runs every experiment in order; steps marked [GPU] need a GPU.

| Step | Scripts |
| --- | --- |
| 1. Data sets | `scripts/data/`, `scripts/real_assemblies/build_sets.sh`, `label_models.py` |
| 2. Standard schedule, PyTorch [GPU] | `scripts/genodirnet/train_torch.py` |
| 3. Standard schedule, NumPy | `scripts/genodirnet/train_numpy.py`, `score_numpy.py`, `fuse.py` |
| 4. Compared methods | `scripts/comparisons/` |
| 5. GenoDirNet-SO and GenoDirNet-2 [GPU] | `scripts/genodirnet2/` |
| 6. Real assembled transcripts | `scripts/genodirnet2/predict.py`, `scripts/real_assemblies/run_tools.sh`, `collect_scores.py` |
| 7. Scoring time, tables and figures | `scripts/comparisons/runtime.py`, `analysis/make_tables.py`, `analysis/make_figures.py` |

Run outputs are written to `results/`, trained models to `models/`, and every table, figure and number of the
article to `paper/` (`paper/tables/<name>.tsv`, `paper/figures/<name>.pdf`, `paper/numbers.tsv`). The transcript
lists of the chromosome splits are in `splits/`. The version used in the article and its run outputs are archived
at https://osf.io/rjuat.

## Licence

MIT (`LICENSE`). `scripts/comparisons/rnasamba_numpy.py` reimplements the inference of
[RNAsamba](https://github.com/apcamargo/RNAsamba) and is distributed under the GNU General Public License version 3.
