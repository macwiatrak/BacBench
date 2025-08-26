# Operon identification from whole bacterial genomes

A benchmark for evaluating strain clustering using different pre-trained models. The dataset has been collated from [MGnify](https://www.ebi.ac.uk/metagenomics).

## Task description

The input to the model is a whole bacterial genome, specifically, metagenome assembled genome (MAG). Each genome is embedded with pre-trained models.
We then use the genome embeddings to cluster them and evaluate whether they cluster by `species`, `genus` and `family`.

The task is formulated as an unsupervised clustering problem, where we look at the distance of bacterial strains to each other
and their taxonomic labels. We evaluate the clustering performance using the adjusted Rand index (ARI), normalized mutual information (NMI) metrics
and average silhouette width (ASW).

## Embedding genomes

The first step is to embed genes in bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) DNA LMs, 2) protein LMs, and 3) contextualized protein LM.

```bash
# embed and save the genomes using the ESM-2 model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-protein-sequences \
    --output-filepath <output-dir>/strain_clustering_esm2_embeddings.parquet \
    --model-path facebook/esm2_t12_35M_UR50D \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-protein-sequences \
    --output-filepath <output-dir>/strain_clustering_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming \
    --max-n-proteins 9000  # max nr of proteins in a genome, default value


# embed and save the genomes using the Nucleotide Transformer model
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-strain-clustering-dna \
    --output-filepath <output-dir>/strain_clustering_nucleotide_transformer_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32 \
    --agg-whole-genome \
    --streaming
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide a script evaluate the pre-trained models. The models are evaluated using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository.

```bash
python bacbench/tasks/operon/run_evaluation_operondb.py \
    --input-df-filepath <input-dir>/strain_clustering_esm2_embeddings.parquet \
    --output-dir <output-dir> \
    --model-name esm2
```
