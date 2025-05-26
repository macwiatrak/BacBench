# Operon identification from whole bacterial genomes

A benchmark for identifying operons in bacterial genomes. The dataset has been collated from the [Operon DB (known operons)](https://operondb.jp/known).

## Task description

The input to the model is a whole bacterial genome which consists of `N` genes. Each gene is embedded with pre-trained models.
We then use the gene embeddings to identify genes which form operons across diverse bacterial genomes in a completely
unsupervised manner.

The task is formulated as a zero-shot binary classification problem, where the model predicts whether a set of `N` genes
form an operon and compare the score to the negative operon sets, which are sampled at random.

## Embedding genomes

The first step is to embed genes in bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) DNA LMs, 2) protein LMs, and 3) contextualized protein LM.

```bash
# embed and save the genomes using the ProtBert model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-operon-identification-protein-sequences \
    --output-filepath <output-dir>/operon_identification_protbert_embeddings.parquet \
    --model-path Rostlab/prot_bert  \
    --model-type protbert \
    --batch-size 64

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-operon-identification-protein-sequences \
    --output-filepath <output-dir>/operon_identification_bacformer_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --model-type bacformer \
    --batch-size 64 \
    --max-n-proteins 9000  # max nr of proteins in a genome, default value


# embed and save the genomes using the Mistral-DNA model
python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-operon-identification-dna \  # name of the dataset
    --output-filepath <output-dir>/operon_identification_mistral_embeddings.parquet \
    --model-path Raphaelmourad/Mistral-DNA-v1-138M-bacteria \
    --model-type mistral_dna \
    --batch-size 256 \
    --max-seq-len 512 \
    --dna-seq-overlap 16
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide a script evaluate the pre-trained models. The models are evaluated using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository.

```bash
python bacbench/tasks/operon/run_evaluation.py \
    --input-df-filepath <input-dir>/operon_identification_bacformer_embeddings.parquet \
    --output-dir <output-dir> \
    --model-name bacformer
```
