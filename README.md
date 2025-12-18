# NanoMod-tRNA v0.9.6

NanoMod-tRNA is a tRNA modification detection tool based on Multiple Instance Learning (MIL) with per-read classification + Noisy-OR pooling, Adaptive Training Strategy, and Structure-Aware Balancing.

## Key Features

- **MIL with Noisy-OR pooling**: Keeps all reads for each site (30 by default), predicts per-read modification probabilities, and aggregates them into a site-level probability using Noisy-OR pooling.
- **Adaptive Training Strategy**: Automatically selects the optimal training mode based on mismatch rates.
  - **Mode A (high mismatch rate)**: Uses Bayesian soft labels; applied when the mismatch rate at candidate sites is > 1.5 × the global mismatch rate.
  - **Mode B (low mismatch rate)**: Uses hard labels; applied when mismatch rate differences are not significant.
- **Structure-Aware Balancing (added in v0.9.6)**: Structure-aware balanced sampling strategy.
  - Performs balancing within each tRNA structure.
    * If the number of positive sites > negative sites: keep all sites in that structure.
    * If the number of positive sites ≤ negative sites: perform 1:1 positive/negative balancing.
  - For structures without any modified sites, keep 10% of sites as negative examples (at least 1 site).
  - Avoids the model learning trivial position bias (e.g. "structure=3 means modified").
- **Automatic mode selection**: Chooses training strategy automatically based on mismatch rate analysis; no manual tuning required.

## Installation

Install from source code:

```bash
# Clone repository
git clone https://github.com/jingkun12137/NanoMod-tRNA.git
cd NanoMod-tRNA

# Install package
pip install -e .
```

## Usage (four-step workflow)

After installation, NanoMod-tRNA can be run in four steps:

### Get Help

Show top-level help:

```bash
NanoMod -h
# or
NanoMod --help
```

Show subcommand help:

```bash
NanoMod train -h
NanoMod predict -h
NanoMod mistake -h
NanoMod modification -h
```

### 1) Generate modification site file (modification)

Given modified/unmodified tRNA FASTA files and a reference FASTA, generate a TSV of modification sites and a PDF of their distribution:

```bash
NanoMod modification \
  --modified NanoMod_data/modified_tRNA_all_all_rna_sequences.fasta \
  --unmodified NanoMod_data/unmodified_tRNA_all_all_rna_sequences.fasta \
  --reference NanoMod_data/yeast.tRNA.ext.fa \
  --mod-type D \
  --output NanoMod_data/tRNA_D_modification_sites.tsv \
  --pdf-output NanoMod_data/tRNA_D_modification_sites.pdf
```

Arguments:
- `--modified` Modified tRNA FASTA (required).
- `--unmodified` Unmodified tRNA FASTA (required).
- `--reference` Reference tRNA FASTA (required).
- `--output` Output TSV file for modification sites.
- `--mod-type` Modification type (default: `D`).
- `--pdf-output` Output PDF file for modification site distribution.
- `--identity-threshold` Minimum identity threshold for alignment (optional).
- `--no-predict` Skip prediction of modification sites for unannotated tRNAs (optional).

### 2) Detect base mismatches in feature file (mistake)

Detect mismatches in the feature file, output summary statistics and a visualization PDF, and write a `mistake` column back into the original feature TSV:

```bash
NanoMod mistake \
  --fasta NanoMod_data/yeast.tRNA.ext.fa \
  --features NanoMod_data/yeast_tRNA_features.tsv \
  --output NanoMod_mis/tRNA_mismatch.csv \
  --pdf-output NanoMod_mis/tRNA_mismatch.pdf
```

Arguments:
- `--features` Feature TSV file (required).
- `--fasta` Reference FASTA (default: `yeast.tRNA.ext.fa`).
- `--output` Output CSV file for mismatch statistics.
- `--pdf-output` Output PDF file for mismatch visualization.

### 3) Model training (train)

MIL (per-read classifier + Noisy-OR pooling) + Adaptive Training Strategy + Structure-Aware Balancing:
- **Input features**: 30 reads per site are kept; each read has 6 electrical signal features.
- **Noisy-OR pooling**: Converts per-read probabilities into a site-level probability.
- **tRNA structure embedding**: 16-dimensional positional encoding.
- **Adaptive Training Strategy**:
  - **Mismatch rate analysis**: Automatically computes mismatch rates for candidate sites and globally.
  - **Mode A (high mismatch rate)**: When candidate mismatch rate > threshold × global mismatch rate, use Bayesian soft labels for training.
  - **Mode B (low mismatch rate)**: When mismatch rate difference is not significant, use hard labels for training.
- **Structure-Aware Balancing (v0.9.6)**:
  - Perform balancing within each tRNA structure.
    * If the number of positive sites > negative sites: keep all sites in that structure.
    * If the number of positive sites ≤ negative sites: perform 1:1 positive/negative balancing.
  - For structures without modified sites, keep 10% of sites as negative examples (at least 1 site).
  - Prevent the model from overfitting to position information and ignoring signal features.
- **Outputs**: Trained model, training curves, and AUROC/AUPRC curves.

```bash
NanoMod train \
  --train-file NanoMod_data/yeast_tRNA_features.tsv \
  --val-file   NanoMod_data/yeast_tRNA_features_2.tsv \
  --mod-site-file NanoMod_data/tRNA_D_modification_sites.tsv \
  --mod-type D \
  --num-reads 30 \
  --mismatch-threshold 1.5 \
  --learning-rate 0.003 \
  --epochs 300 \
  --batch-size 256
```

Arguments:
- `--train-file` Training feature TSV (required).
- `--val-file` Validation feature TSV (required).
- `--mod-site-file` Modification site TSV (required).
- `--output-dir` Directory to save model and results (default: `NanoMod_model`).
- `--mod-type` Modification type (default: `D`).
- `--num-reads` Number of reads per site (default: 30).
- `--mismatch-threshold` Mismatch rate threshold for mode selection (default: 1.5).
- `--batch-size` Batch size (default: 256).
- `--epochs` Number of training epochs (default: 300).
- `--learning-rate` Learning rate (default: 0.003).
- `--dropout-rate` Dropout rate (default: 0.3).
- `--num-workers` Number of data loader worker threads (default: 128).
- `--gpu` GPU ID (default: 0; use -1 for CPU).
- `--seed` Random seed (default: 42).
- `--use-preprocessed` Use preprocessed data to speed up training (optional).
- `--kmer-nums` Compatibility parameter (default: 781).


### 4) Model prediction (predict)

Use a trained MIL model (per-read classifier + Noisy-OR pooling) to predict tRNA modifications:
- **Open-world prediction**: No candidate list is required; prediction is performed on all sites.
- **Outputs**: Site-level modification probability TSV and optional read-level probability TSV.

```bash
NanoMod predict \
  --features NanoMod_data/yeast_tRNA_features_3.tsv \
  --mod-type D \
  --output NanoMod_output/D_modification_predictions_1.tsv \
  --save-read-level \
  --read-output NanoMod_output/D_read_level_predictions_1.tsv
```

Arguments:
- `--features` Feature TSV file (required).
- `--mod-type` Modification type (default: `D`), used to auto-select the corresponding model.
- `--model` Model path (optional, default: auto-select `NanoMod_model/{MOD}_best_model.pt`).
- `--output` Full path to the site-level prediction TSV (default: `NanoMod_output/{MOD}_modification_predictions.tsv`).
- `--save-read-level` Whether to additionally output a read-level probability TSV (optional).
- `--read-output` Full path to the read-level prediction TSV (required when `--save-read-level` is enabled).
- `--num-instances` Number of reads per site (default: 30; should match training).
- `--num-workers` Number of data loader worker threads (default: 128).
- `--threshold` Deprecated (prediction uses fixed threshold 0.5).
