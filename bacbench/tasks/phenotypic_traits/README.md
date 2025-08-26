# Antibiotic resistance prediction from whole bacterial genomes

A benchmark for phenotypic traits prediction from whole bacterial genomes. The dataset has been collated by
combining multpiple sources spanning a wide variety of phenotypes [1,2,3].

## Task description

The input to the model is a whole bacterial genome which is embedded with pre-trained models.
We then use the genome embeddings to predict phenotypic traits across diverse bacterial genomes spanning thousands of species.

We train a separate MLP model for each of the phenotypic traits. The model is trained to predict the phenotypic traits based on the genome embeddings.

## Embedding genomes

The first step is to embed the whole bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) DNA LMs, 2) protein LMs, and 3) contextualized protein LM.

```bash
# embed and save the genomes using the ESM-C model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-protein-sequences \
    --output-filepath <output-dir>/pheno_esmc_genome_embeddings.parquet \
    --model-path esmc_300m \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming # use streaming to avoid memory issues

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-protein-sequences \
    --output-filepath <output-dir>/pheno_bacformer_genome_embeddings.parquet \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --batch-size 64 \
    --genome-pooling-method mean \
    --streaming \
    --max-n-proteins 9000  # max nr of proteins in a genome

python bacbench/modeling/run_embed_dna.py \
    --dataset-name macwiatrak/bacbench-phenotypic-traits-dna \
    --output-filepath <output-dir>/pheno_nucleotide_transformer_embeddings.parquet \
    --model-path InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --batch-size 128 \
    --max-seq-len 2048 \
    --dna-seq-overlap 32 \
    --agg-whole-genome \
    --streaming
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide scripts to train and evaluate models. The models can be trained using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository.

The phenotypic traits labels file is available to download from the [Hugging Face dataset repository](https://huggingface.co/datasets/macwiatrak/bacbench-phenotypic-traits-protein-sequences/tree/main).

```bash
python bacbench/tasks/phenotypic_traits/run_train_mlp.py \
    --input-genomes-df-filepath <output-dir>/pheno_bacformer_genome_embeddings.parquet \
    --labels-df-filepath <input-dir>/labels.csv \
    --output-dir <output-dir> \
    --model-name bacformer
```

## References
```
[1] Madin, Joshua S., et al. "A synthesis of bacterial and archaeal phenotypic trait data." Scientific data 7.1 (2020): 170.

[2] Weimann, Aaron, et al. "From genomes to phenotypes: Traitar, the microbial trait analyzer." MSystems 1.6 (2016): 10-1128.

[3] Brbić, Maria, et al. "The landscape of microbial phenotypic traits and associated genes." Nucleic acids research (2016): gkw964.
```
