# Phenotypic Traits Prediction

This task benchmarks prediction of broad bacterial phenotypic traits from whole-genome embeddings. The labels combine multiple phenotype resources spanning a wide variety of traits [1,2,3].

The script trains one MLP per phenotype and reports validation metrics across three random seeds.

## Data

Input embeddings are whole-genome parquet files produced by the BacBench embedding scripts. The evaluation script reads:

- `genome_name`
- the embedding column named by `--embeddings-col`

If you use the embedding scripts without changing their defaults, the embedding column is `embeddings`, which is also the default value of `--embeddings-col`.

The labels CSV is available from the Hugging Face dataset repository for the phenotypic traits task - https://huggingface.co/datasets/macwiatrak/bacbench-phenotypic-traits-protein-sequences/blob/main/labels.csv .

## Embedding Genomes

Run these commands from the repository root. Whole-genome tasks should use `--agg-whole-genome` and a genome pooling method.

```bash
# Protein model example: ESM-C
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-protein-sequences \
    --output-filepath <output-dir>/pheno_esmc_genome_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming

# Contextualized whole-genome protein model example: Bacformer Large
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-protein-sequences \
    --output-filepath <output-dir>/pheno_bacformer_genome_embeddings.parquet \
    --model-path macwiatrak/bacformer-large-masked-complete-genomes \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming \
    --max-n-proteins 9000

# DNA model example: Nucleotide Transformer
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-dna \
    --output-filepath <output-dir>/pheno_nucleotide_transformer_embeddings.parquet \
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
bacbench/tasks/phenotypic_traits/train_and_predict_linear.py
```

Example:

```bash
python bacbench/tasks/phenotypic_traits/train_and_predict_linear.py \
    --input-genomes-df-filepath <output-dir>/pheno_bacformer_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/labels.csv \
    --output-dir <output-dir> \
    --embeddings-col embeddings \
    --model-name bacformer \
    --lr 0.01
```

Useful options:

- `--embeddings-col <column>`: embedding column in the input parquet, by default `embeddings`.
- `--model-name <name>`: label used in the output CSV filename and metrics table.
- `--limit-n-phenotypes <N>`: run only the first `N` phenotypes for debugging.
- `--test-after-train`: also report test metrics after validation.
- `--min-class-samples`: filter rare phenotype classes.
- `--split genus`: use a group-aware split column when present in the merged dataframe. If the column is not present, the script falls back to a random split.

## Output

The script writes:

```text
<output-dir>/phenotypic_traits_preds_<model-name>_<date>.csv
```

## Notes

Some source traits are quantitative. For consistency, this benchmark treats them as categorical traits and filters rare classes.

## References

```text
[1] Madin, Joshua S., et al. "A synthesis of bacterial and archaeal phenotypic trait data." Scientific Data 7.1 (2020): 170.

[2] Weimann, Aaron, et al. "From genomes to phenotypes: Traitar, the microbial trait analyzer." mSystems 1.6 (2016): 10-1128.

[3] Brbic, Maria, et al. "The landscape of microbial phenotypic traits and associated genes." Nucleic Acids Research (2016): gkw964.
```
