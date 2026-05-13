# Essential Genes Prediction

This task benchmarks prediction of essential genes in bacterial genomes. The dataset is collated from the [Database of Essential Genes (DEG)](http://origin.tubic.org/deg/public/index.php/browse/bacteria).

Each genome contains gene-level labels and gene/protein embeddings. The main evaluation script trains a linear classifier on gene embeddings and reports per-genome metrics.

## Data

Input parquet files should contain:

- `genome_name`
- `split` with `train`, `validation`, and `test` values
- `essential` gene labels
- an embedding column, defaulting to `embeddings`

The script explodes each genome into gene-level rows internally.

## Embedding Genomes

Run these commands from the repository root.

```bash
# Protein model example: ESM-C
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-essential-genes-protein-sequences \
    --output-filepath <output-dir>/essential_genes_esmc_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64

# Contextualized whole-genome protein model example: Bacformer Large
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-essential-genes-protein-sequences \
    --output-filepath <output-dir>/essential_genes_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-large-masked-complete-genomes \
    --batch-size 64 \
    --max-n-proteins 9000

# DNA model example: Nucleotide Transformer
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-essential-genes-dna \
    --output-filepath <output-dir>/essential_genes_nt_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32
```

## Model Training And Evaluation

Tune `--lr` on the validation set before reporting final test metrics.

The main linear-probe script is:

```bash
bacbench/tasks/essential_genes/run_train_cls.py
```

Example:

```bash
python bacbench/tasks/essential_genes/run_train_cls.py \
    --input-df-file-path <output-dir>/essential_genes_esmc_embeddings.parquet \
    --output-dir <output-dir> \
    --lr 0.005 \
    --max-epochs 100 \
    --model-name esmc \
    --embeddings-col embeddings
```

The CLI uses `--input-df-file-path` because the parser field is `input_df_file_path`.

## Optional End-To-End Finetuning Scripts

The directory also contains task-specific finetuning scripts for sequence models:

- `finetune_plm.py`: protein language models.
- `finetune_dna_lm.py`: DNA language models.
- `finetune_glm.py`: mixed DNA/protein gLM-style models.

These scripts load Hugging Face datasets directly and write `test_results.csv` plus checkpoints under `--output-dir`.

## Output

The linear-probe script writes:

```text
<output-dir>/finetune_results_<model-name>.parquet
```
