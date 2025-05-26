# Predicting protein-protein interactions (PPIs) between bacterial proteins in whole bacterial genomes

A benchmark for predicting protein-protein interactions between the proteins in the bacterial genome.
The dataset has been collated from the [STRING database](https://string-db.org/). We use the bacterial organisms
present in the STRING database to create a dataset of whole bacterial genomes with protein sequences. We leverage the
`Combined` score

## Task description

The input to the model is a whole bacterial genome which is embedded with pre-trained models.
We then use the pairs of protein embeddings to predict whether two proteins interact with each other.

The task is formulated as a binary classification problem, where the model predicts whether two proteins interact with each other.
The threshold for the interaction is set to 0.6, which has shown to be a good balance between AUROC and AUPRC metrics.

## Embedding genomes

The first step is to embed the whole bacterial genomes using pre-trained models. Below, we show examples on how to do it using
1) protein LMs, and 2) contextualized protein LM. We do not run the embedding for DNA sequences, as the PPI task is based on protein sequences.

```bash
# embed and save the genomes using the ESM-C model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences \
    --output-dir <prot-embeds-output-dir> \
    --model-path esmc_300m \
    --model-type esmc \
    --batch-size 64 \
    --save-every-n-rows 500 \
    --streaming # use streaming to avoid memory issues

# embed and save the genomes using the Bacformer model
python bacbench/modeling/run_embed_prot_seqs.py \
    --dataset-name macwiatrak/bacbench-ppi-stringdb-protein-sequences \
    --output-dir <prot-embeds-output-dir> \
    --model-path macwiatrak/bacformer-masked-complete-genomes \
    --model-type bacformer \
    --batch-size 64 \
    --save-every-n-rows 500 \
    --streaming \
    --max-n-proteins 9000  # max nr of proteins in a genome
```

For more info on supported models see the README in the root directory.

## Model training and evaluation

We provide scripts to train and evaluate models. The models can be trained using the embeddings generated from the pre-trained models (see step above).

This script should be executed in the root directory of the repository. Use the `<ouput-dir>` from the embedding step above as the `<input-dir>`.

### Train the MLP model for PPI prediction
```bash
python bacbench/tasks/ppi/run_train_mlp.py \
    --input-dir <prot-embeds-output-dir> \
    --output-dir <model-output-dir>
```

### Evaluate the MLP model for PPI prediction

#### Supervised evaluation
```bash
python bacbench/tasks/ppi/run_supervised_eval.py
    --input-dir <prot-embeds-output-dir> \
    --output-dir <supervised-eval-output-dir> \
    --model-name <model-name> \
    --ckpt-path <model-output-dir>/best_model.ckpt
```

### Unsupervised evaluation
```bash
python bacbench/tasks/ppi/run_unsupervised_eval.py
    --input-dir <prot-embeds-output-dir> \
    --output-dir <unsupervised-eval-output-dir> \
    --model-name <model-name>
```
