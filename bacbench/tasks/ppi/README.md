# Protein-Protein Interaction Prediction

This task benchmarks prediction of protein-protein interactions (PPIs) within bacterial genomes. Labels are derived from the [STRING database](https://string-db.org/) combined score.

The task is evaluated as binary classification. By default, STRING combined scores are thresholded at `0.6`.

## Data

The PPI scripts expect one parquet file with:

- `strain_name`
- `split` with `train`, `validation`, and `test` values
- `labels`: protein-pair labels per contig
- `embeddings`: per-protein embeddings per contig

The split is embedded in the parquet file. There is no separate train/test split JSON.

## Embedding Genomes

Run these commands from the repository root. The small dataset is useful for local checks; the large dataset is substantially larger.

```bash
# Protein model example: ESM-C on the small PPI dataset
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences-small \
    --output-filepath <output-dir>/ppi_esmc_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64 \
    --streaming

# Contextualized whole-genome protein model example: Bacformer
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences-small \
    --output-filepath <output-dir>/ppi_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --streaming \
    --max-n-proteins 9000
```

For the large dataset, use:

```bash
--dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences
```

The training and evaluation scripts consume a single parquet file. If you save embedding chunks, merge them into one parquet that keeps the `split` column before running the task scripts.

## Supervised MLP Training

Tune `--lr` on the validation set before reporting final test metrics.

The main training script is:

```bash
bacbench/tasks/ppi/run_train_mlp.py
```

Example:

```bash
python bacbench/tasks/ppi/run_train_mlp.py \
    --input-filepath <output-dir>/ppi_esmc_embeddings.parquet \
    --output-dir <model-output-dir> \
    --batch-size 256 \
    --max-epochs 10 \
    --score-threshold 0.6
```

By default, the script trains and validates only. Add `--test-after-train` after tuning validation settings to run test evaluation and write test prediction files.

For very large parquet files, use incremental split construction:

```bash
python bacbench/tasks/ppi/run_train_mlp.py \
    --input-filepath <output-dir>/ppi_esmc_embeddings.parquet \
    --output-dir <model-output-dir> \
    --use-incremental-parquet-read
```

## Unsupervised Evaluation

The unsupervised baseline scores protein pairs by cosine similarity between their embeddings and evaluates only the `test` split.

```bash
python bacbench/tasks/ppi/run_unsupervised_eval.py \
    --input-filepath <output-dir>/ppi_esmc_embeddings.parquet \
    --output-dir <eval-output-dir> \
    --model-name esmc \
    --score-threshold 0.6
```

## Embedding homology baseline

`run_embedding_homology_baseline.py` reads the flattened, one-contig-per-row ESM-C parquet hardcoded at
`/Users/maciejwiatrak/Downloads/esmc_ppi_subset.parquet`. It removes duplicated `(i, j)`/`(j, i)` interactions,
thresholds STRING scores at `0.6`, and represents each unordered pair by the normalized mean of its two normalized
protein embeddings. The script tunes `k` over `[1, 3, 5, 10]` using median per-genome validation AUPRC, then evaluates
the selected value once on the test split. Contigs are aggregated by `strain_name` before AUROC and AUPRC are
calculated.

Run the baseline without arguments from the repository root:

```bash
python bacbench/tasks/ppi/run_embedding_homology_baseline.py
```

The exact neighbor search runs on a CUDA GPU, while parquet parsing and pair construction remain on CPU. A CPU fallback
is used when CUDA is unavailable. The script reports mean, median, and sample standard deviation across test genomes
and writes validation tuning, per-genome test, and aggregate test CSV files under `/Users/maciejwiatrak/Downloads`.

## Output

`run_train_mlp.py` always writes checkpoints, logs, and `args.json` under `--output-dir`. When `--test-after-train` is set, it also writes `test_predictions.csv` and `test_predictions_by_genome.csv`.

`run_unsupervised_eval.py` writes:

```text
<output-dir>/unsupervised_eval_<model-name>.csv
```
