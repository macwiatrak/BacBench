# Gene essentiality prediction from whole bacterial genomes

A benchmark for predicting essential genes in bacterial genomes. The dataset has been collated from the [Database of Essential Genes (DEG)](http://origin.tubic.org/deg/public/index.php/browse/bacteria).

## Task description

The input to the model is a whole bacterial genome which consists of `N` genes. Each gene is embedded with pre-trained models.
We then use the gene embeddings to predict gene essentiality across diverse bacterial genomes.

The task is formulated as a binary classification problem, where the model predicts whether a gene is essential or not.

## Embedding genomes

The first step is to embed genes in bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) DNA LMs, 2) protein LMs, and 3) contextualized protein LM.

```bash
# embed and save the genomes using the ESM-C model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-essential-genes-protein-sequences \
    --output-filepath <output-dir>/essential_genes_esmc_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-essential-genes-protein-sequences \
    --output-filepath <output-dir>/essential_genes_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --max-n-proteins 9000  # max nr of proteins in a genome

# embed and save the genomes using the Nucleotide Transformer model
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-essential-genes-dna \
    --output-filepath <output-dir>/essential_genes_nt_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32  # overlap between the sequences when the gene length is higher than --max-seq-len, default value
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide a script to train and evaluate the pre-trained models. The models can be trained using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository.

```bash
python bacbench/tasks/essential_genes/run_train_cls.py \
    --input-df-filepath <output-dir>/essential_genes_esmc_embeddings.parquet \
    --output-dir <output-dir> \
    --model-name esmc
```
