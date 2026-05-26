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

## AAI-Based Species Clustering

You can also cluster genomes directly from their protein sequences using average amino acid identity (AAI). This path expects one row per genome with a `genome_name` identifier, a `protein_sequence` column containing either a flat list of proteins or a contig-grouped list of lists/arrays, plus a `species` label used for evaluation. It reads parquet input incrementally with `pyarrow.parquet.ParquetFile.iter_batches`, uses MMseqs2 for the all-vs-all protein search, computes reciprocal-best-hit AAI, runs Leiden clustering on the precomputed AAI distance matrix, and reports ARI, NMI, homogeneity, completeness, V-measure, and silhouette.

```bash
python bacbench/tasks/strain_clustering/run_aai_clustering.py \
    --input-df-filepath <input-dir>/genome_proteins.parquet \
    --output-dir <output-dir>/aai_species_clustering \
    --genome-col genome_name \
    --proteins-col protein_sequence \
    --species-col species \
    --label-col species \
    --leiden-resolutions 0.1 0.25 1.0 \
    --k-neighbors 5 10 15 \
    --input-batch-size 100 \
    --mmseqs-split-memory-limit 110G \
    --mmseqs-max-seqs 10 \
    --mmseqs-min-seq-id 0.5 \
    --n-bootstraps 10 \
    --proportion 0.8 \
    --threads 32
```

The default hit filters are `evalue <= 1e-5`, query and target coverage `>= 0.5`, minimum sequence identity `>= 0.5`, and minimum alignment fraction `>= 0.2`. These e-value, identity, and coverage thresholds are passed into MMseqs before Python post-processing. MMseqs defaults to `--max-seqs 10` to keep the hit table bounded, and Python streams `mmseqs_hits.tsv` into a SQLite-backed best-hit reduction instead of loading the full file into memory. The MMseqs search defaults to `--split-memory-limit 110G` to reduce OOM risk on large all-vs-all runs; lower this value if your SLURM allocation is smaller. The script reports progress for streamed input passes, MMseqs processing, AAI construction, and Leiden evaluation; pass `--disable-progress` for quieter batch logs. If `mmseqs_hits.tsv` already exists in the output directory, the script reuses it unless `--force` is set.

If a run fails after writing `genome_index.csv`, `pairwise_aai.parquet`, `aai_matrix.csv`, and `distance_matrix.npy`, rerun the same command without `--force`. The script will reuse those existing artifacts and resume from the Leiden evaluation step.

Use `<output-dir>/final_metrics.csv` for the headline AAI result. It contains the best Leiden setting selected by mean bootstrap ARI, with mean NMI, mean V-measure, and mean silhouette as tie-breakers. Use `<output-dir>/metrics.csv` to audit the bootstrap metric summary for every Leiden resolution and neighbor setting, and `<output-dir>/bootstrap_metrics.csv` for the individual bootstrap samples.

## Output

The embedding-based script writes:

```text
<output-dir>/results_<model-name>.parquet
```

The AAI-based script writes:

```text
<output-dir>/pairwise_aai.parquet
<output-dir>/aai_matrix.csv
<output-dir>/distance_matrix.npy
<output-dir>/genome_index.csv
<output-dir>/protein_index.csv
<output-dir>/aai_hits.sqlite
<output-dir>/clusters.csv
<output-dir>/metrics.csv
<output-dir>/bootstrap_metrics.csv
<output-dir>/full_dataset_metrics.csv
<output-dir>/final_metrics.csv
<output-dir>/final_clusters.csv
```

AAI plots are skipped by default; pass `--make-plots` to also write `aai_heatmap.png`, `aai_dendrogram.png`, and `aai_mds.png`.

For large datasets, `pairwise_aai.parquet` contains genome pairs with reciprocal-best-hit evidence. Pairs absent from that file are treated as invalid/no-hit pairs and receive distance `1.0` in `distance_matrix.npy`.
