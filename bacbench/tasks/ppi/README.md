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

## Output

`run_train_mlp.py` writes checkpoints, logs, `args.json`, `test_predictions.csv`, and `test_predictions_by_genome.csv` under `--output-dir`.

`run_unsupervised_eval.py` writes:

```text
<output-dir>/unsupervised_eval_<model-name>.csv
```
