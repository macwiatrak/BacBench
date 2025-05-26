import os

import pandas as pd
import torch
from datasets import tqdm
from tap import Tap
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision

from bacbench.tasks.ppi.data_reader import collate_ppi, get_datasets_ppi
from bacbench.tasks.ppi.run_train_mlp import PpiLightningModule


def run(
    input_dir: str,
    output_dir: str,
    ckpt_path: str,
    model_name: str,
    score_threshold: float = 0.6,
    max_n_proteins: int = 6000,
    max_n_ppi_pairs: float = 2 * 1e6,
    dataloader_num_workers: int = 6,
    embeddings_col: str = "embeddings",
):
    """Evaluate the model on the PPI dataset in an unsupervised manner."""
    # Prepare datasets
    _, _, test_dataset = get_datasets_ppi(
        input_dir=input_dir,
        max_n_proteins=max_n_proteins,
        test=True,
        score_threshold=score_threshold,
        max_n_ppi_pairs=max_n_ppi_pairs,
        embeddings_col=embeddings_col,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = PpiLightningModule.load_from_checkpoint(ckpt_path).eval().to(device).to(torch.float16)

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=dataloader_num_workers, collate_fn=collate_ppi)

    output = []
    for _, batch in tqdm(enumerate(test_loader)):
        protein_embeddings = batch.pop("protein_embeddings").to(device)
        labels = batch.pop("ppi_labels").to(device)
        loss, logits = model(protein_embeddings, labels=labels)
        labels = labels[:, 2].type(torch.long).squeeze(0)

        auroc_val = auroc(logits, labels, task="binary").item()
        auprc_val = average_precision(logits, labels, task="binary").item()

        output.append(
            {
                "genome_name": batch["genome_name"][0],
                "loss": loss.item(),
                "auroc": auroc_val,
                "auprc": auprc_val,
                "n_proteins": protein_embeddings.shape[1],
                "n_ppi_pairs": len(labels),
                "n_pos_triples": (labels == 1).sum().item(),
            }
        )

        if _ % 100 == 0:
            print(f"Processed {_} batches, {len(output)} samples.")
            # Save intermediate results
            pd.DataFrame(output).to_csv(os.path.join(output_dir, f"supervised_eval_{model_name}.csv"), index=False)

    output = pd.DataFrame(output)
    output.to_csv(os.path.join(output_dir, f"supervised_eval_{model_name}.csv"), index=False)


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    output_dir: str
    ckpt_path: str
    model_name: str = "esmc"
    score_threshold: float = 0.6
    max_n_proteins: int = 6000
    max_n_ppi_pairs: float = 3e6


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    args = parser.parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        score_threshold=args.score_threshold,
        max_n_proteins=args.max_n_proteins,
        max_n_ppi_pairs=args.max_n_ppi_pairs,
    )
