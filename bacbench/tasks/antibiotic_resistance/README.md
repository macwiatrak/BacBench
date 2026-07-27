# Antibiotic Resistance Prediction

This task benchmarks prediction of antibiotic resistance phenotypes from whole bacterial genome embeddings. The data are collated from the [NCBI AST Browser](https://www.ncbi.nlm.nih.gov/pathogens/ast).

The task supports two settings:

- **Binary classification**: predict resistant vs. susceptible calls for each antibiotic.
- **Regression**: predict minimum inhibitory concentration (MIC) values.

The evaluation script trains one MLP per antibiotic and reports validation metrics across three random seeds.

## Data

Input embeddings are whole-genome parquet files produced by the BacBench embedding scripts. The evaluation script reads:

- `genome_name`
- the embedding column named by `--embeddings-col`

If you use the embedding scripts without changing their defaults, the embedding column is `embeddings`, so pass `--embeddings-col embeddings`.

Label CSV files are available from the Hugging Face dataset repository:

- Binary labels: `binary_labels.csv`
- MIC regression labels: `mic_regression_labels.csv`

The labels file must contain `genome_name` plus one column per antibiotic.

## Embedding Genomes

Run these commands from the repository root. Whole-genome tasks should use `--agg-whole-genome` and a genome pooling method.

```bash
# Protein model example: ESM-C
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-protein-sequences \
    --output-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming

# Contextualized whole-genome protein model example: Bacformer Large
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-protein-sequences \
    --output-filepath <output-dir>/amr_bacformer_genome_embeddings.parquet \
    --model-path macwiatrak/bacformer-large-masked-complete-genomes \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming \
    --max-n-proteins 9000

# DNA model example: Nucleotide Transformer
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-dna \
    --output-filepath <output-dir>/amr_nucleotide_transformer_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32 \
    --agg-whole-genome \
    --genome-pooling-method mean \
    --streaming
```

## Model Training And Evaluation

Tune `--lr` on the validation set before reporting final test metrics.

The main script is:

```bash
bacbench/tasks/antibiotic_resistance/train_and_predict_linear.py
```

Binary classification:

```bash
python bacbench/tasks/antibiotic_resistance/train_and_predict_linear.py \
    --input-genomes-df-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/binary_labels.csv \
    --output-dir <output-dir> \
    --model-name embeddings \
    --lr 0.005
```

MIC regression:

```bash
python bacbench/tasks/antibiotic_resistance/train_and_predict_linear.py \
    --input-genomes-df-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/mic_regression_labels.csv \
    --output-dir <output-dir> \
    --model-name embeddings \
    --lr 0.005 \
    --regression
```

Useful options:

- `--limit-n-drugs <N>`: run only the first `N` antibiotics for debugging.
- `--test-after-train`: also report test metrics after validation.
- `--total-min-samples` and `--min-class-samples`: filter low-support antibiotics.

### Multi-model genus-split evaluation

Use `train_and_evaluate_genus_split.py` to evaluate every embedding column in a single wide parquet file. The script
keeps genera disjoint across the 70/10/20 train/validation/test partitions, tunes one learning rate per model using
the mean validation AUPRC across the first 20 eligible antibiotics on seed 1, and evaluates the selected learning
rate across all eligible antibiotics on seeds 1, 2, and 3. Seed values control both the genus partition and
linear-probe initialization. Override the tuning cap with `--tuning-n-drugs` if needed.

```bash
python bacbench/tasks/antibiotic_resistance/train_and_evaluate_genus_split.py
```

The default input and label paths point to the Bacformer RDS AMR model directory. They can be overridden explicitly:

```bash
python bacbench/tasks/antibiotic_resistance/train_and_evaluate_genus_split.py \
    --input-genomes-df-filepath <input-dir>/merged_all_embeddings.parquet \
    --labels-df-filepath <input-dir>/binary_labels.csv \
    --output-dir <output-dir>
```

The defaults retain the existing binary AMR probe settings: 100 epochs, early-stopping patience 10, batch size 256,
dropout 0.1, AdamW, at least 500 measurements per drug, and at least 50 examples from each class. Test metrics are
written as one row per model, drug, and seed. Separate CSV files preserve detailed LR-sweep results and model-level
summaries. Binary label `1` is treated as the resistant/positive class. Use
`--limit-n-models 1 --limit-n-drugs 1 --max-epochs 1` for a small end-to-end smoke run.

## Output

The script writes a CSV file:

```text
<output-dir>/amr_preds_regression_<True-or-False>_split_<split>_<model-name>_<date>.csv
```
