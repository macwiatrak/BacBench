# Antibiotic resistance prediction from whole bacterial genomes

A benchmark for antibiotic resistance prediction from whole bacterial genomes collated from the
[NCBI AST Browset](https://www.ncbi.nlm.nih.gov/pathogens/ast).

## Task description

The input to the model is a whole bacterial genome which is embedded with pre-trained models.
We then use the genome embeddings to predict antibiotic resistance phenotypes across diverse antibiotics.
We train and evaluate the models in two settings:
* **Binary classification**: Predicting whether a bacterium is resistant or susceptible to a given antibiotic.
* **Regression**: Predicting the minimum inhibitory concentration (MIC) of a given antibiotic for a bacterium.

We train a separate MLP model for each of the phenotypic traits. The model is trained to predict the phenotypic traits based on the genome embeddings.

## Embedding genomes

The first step is to embed the whole bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) DNA LMs, 2) protein LMs, and 3) contextualized protein LM.

```bash
# embed and save the genomes using the ESM-C model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-protein-sequences \
    --output-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming # use streaming to avoid memory issues

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-protein-sequences \
    --output-filepath <output-dir>/amr_bacformer_genome_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming \
    --max-n-proteins 9000  # max nr of proteins in a genome

python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-antibiotic-resistance-dna \
    --output-filepath <output-dir>/amr_nucleotide_transformer_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32 \
    --agg-whole-genome \
    --streaming
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide scripts to train and evaluate models for both binary classification and regression tasks. The models can be trained using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository.

The antibiotic resistance labels file is available to download from the [Hugging Face dataset repository](https://huggingface.co/datasets/macwiatrak/bacbench-antibiotic-resistance-dna/tree/main).

```bash
# Binary classification task
python bacbench/tasks/antibiotic_resistance/train_and_predict_linear.py \
    --input-genomes-df-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/binary_labels.csv \
    --output-dir <output-dir> \
    --model-name esmc \
    --lr 0.001

# Regression minimum inhibtion concentration (MIC) task
python bacbench/tasks/antibiotic_resistance/train_and_predict_linear.py \
    --input-genomes-df-filepath <output-dir>/amr_esmc_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/mic_regression_labels.csv \
    --output-dir <output-dir> \
    --model-name esmc \
    --lr 0.001 \
    --regression
```
