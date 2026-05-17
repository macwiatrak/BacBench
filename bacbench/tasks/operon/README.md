# Operon Identification

This task benchmarks zero-shot operon identification from gene embeddings. The main dataset is collated from [OperonDB known operons](https://operondb.jp/known).

The evaluation computes pairwise cosine similarity among genes in known operons and compares those scores with randomly sampled negative gene sets.

## Data

Input parquet files for `run_evaluation_operondb.py` should contain:

- `contig_name`
- `operon_protein_indices`
- `operon_names`
- an embedding column, defaulting to `embeddings`

The script also preserves metadata such as `taxid` in its output.

## Embedding Genomes

Run these commands from the repository root.

```bash
# Protein model example: ProtBert
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-operon-identification-protein-sequences \
    --output-filepath <output-dir>/operon_identification_protbert_embeddings.parquet \
    --model-path Rostlab/prot_bert \
    --batch-size 64

# Contextualized whole-genome protein model example: Bacformer
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-operon-identification-protein-sequences \
    --output-filepath <output-dir>/operon_identification_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --max-n-proteins 9000

# DNA model example: Mistral-DNA
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-operon-identification-dna \
    --output-filepath <output-dir>/operon_identification_mistral_embeddings.parquet \
    --model-path Raphaelmourad/Mistral-DNA-v1-138M-bacteria \
    --batch-size 256 \
    --max-seq-len 512 \
    --dna-seq-overlap 16
```

## Model Evaluation

The main evaluation script is:

```bash
bacbench/tasks/operon/run_evaluation_operondb.py
```

Example:

```bash
python bacbench/tasks/operon/run_evaluation_operondb.py \
    --input-df-filepath <output-dir>/operon_identification_bacformer_embeddings.parquet \
    --output-dir <output-dir> \
    --model-name bacformer \
    --embedding-col embeddings \
    --n-negatives 10
```

## Long-Read RNA-Seq Evaluation

The directory also includes:

```bash
bacbench/tasks/operon/run_evaluation_long_read_rna_seq.py
```

It evaluates adjacent-gene operon boundaries from long-read RNA-seq style inputs:

```bash
python bacbench/tasks/operon/run_evaluation_long_read_rna_seq.py \
    --input-filepath <input-dir>/operon_long_read_embeddings.parquet \
    --output-filepath <output-dir>/long_read_operon_eval.csv
```

## Output

`run_evaluation_operondb.py` writes:

```text
<output-dir>/operon_identification_results_<model-name>.parquet
```
