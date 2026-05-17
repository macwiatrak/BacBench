# Strain Clustering

This task is deprecated in the main BacBench benchmark but remains available for reproducibility.

It evaluates whether whole-genome embeddings cluster metagenome-assembled genomes (MAGs) by `species`, `genus`, and `family`. The dataset was collated from [MGnify](https://www.ebi.ac.uk/metagenomics).

## Data

Input parquet files should contain:

- whole-genome embeddings in `embeddings` or `genome_embedding`
- taxonomic metadata columns `species`, `genus`, and `family`

The current script auto-selects `embeddings` when present, otherwise `genome_embedding`.

## Embedding Genomes

Run these commands from the repository root.

```bash
# Protein model example: ESM-2
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-protein-sequences \
    --output-filepath <output-dir>/strain_clustering_esm2_embeddings.parquet \
    --model-path facebook/esm2_t12_35M_UR50D \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming

# MAG-specific Bacformer checkpoint
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-protein-sequences \
    --output-filepath <output-dir>/strain_clustering_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-MAG \
    --batch-size 64 \
    --genome-pooling-method mean \
    --agg-whole-genome \
    --streaming \
    --max-n-proteins 9000

# DNA model example: Nucleotide Transformer
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-dna \
    --output-filepath <output-dir>/strain_clustering_nucleotide_transformer_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32 \
    --agg-whole-genome \
    --genome-pooling-method mean \
    --streaming
```

## Evaluation

The evaluation script runs Leiden clustering over embeddings and reports ARI, NMI, and silhouette score over bootstrap samples.

```bash
python bacbench/tasks/strain_clustering/run_evaluation.py \
    --input-df-filepath <output-dir>/strain_clustering_esm2_embeddings.parquet \
    --output-dir <output-dir> \
    --model-name esm2 \
    --leiden-resolutions 0.1 0.25 0.5 \
    --k-neighbors 5 10 15 \
    --n-bootstraps 10 \
    --proportion 0.8
```

## Output

The script writes:

```text
<output-dir>/results_<model-name>.parquet
```
