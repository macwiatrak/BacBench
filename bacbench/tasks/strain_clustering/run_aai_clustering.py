from __future__ import annotations

from tap import Tap

from bacbench.tasks.strain_clustering.aai import run_aai_clustering


class ArgumentParser(Tap):
    """Arguments for AAI-based species clustering."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_df_filepath: str
    output_dir: str
    genome_col: str = "genome_name"
    proteins_col: str = "protein_sequence"
    species_col: str = "species"
    min_species_count: int = 2
    max_evalue: float = 1e-5
    min_query_coverage: float = 0.5
    min_target_coverage: float = 0.5
    min_alignment_fraction: float = 0.2
    mmseqs_binary: str = "mmseqs"
    threads: int | None = None
    force: bool = False
    make_plots: bool = True
    random_seed: int = 42
    input_batch_size: int = 100
    leiden_resolutions: list[float] = [0.1, 0.25, 1.0]
    k_neighbors: list[int] = [5, 10, 15]
    label_col: str = "species"
    disable_progress: bool = False


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run_aai_clustering(
        input_df_filepath=args.input_df_filepath,
        output_dir=args.output_dir,
        genome_col=args.genome_col,
        proteins_col=args.proteins_col,
        species_col=args.species_col,
        min_species_count=args.min_species_count,
        max_evalue=args.max_evalue,
        min_query_coverage=args.min_query_coverage,
        min_target_coverage=args.min_target_coverage,
        min_alignment_fraction=args.min_alignment_fraction,
        mmseqs_binary=args.mmseqs_binary,
        threads=args.threads,
        force=args.force,
        make_plots=args.make_plots,
        random_seed=args.random_seed,
        input_batch_size=args.input_batch_size,
        leiden_resolutions=args.leiden_resolutions,
        k_neighbors=args.k_neighbors,
        label_col=args.label_col,
        show_progress=not args.disable_progress,
    )
