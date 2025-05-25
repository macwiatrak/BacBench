import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tap import Tap
from tqdm import tqdm


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) model."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the MLP."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def fit_predict_mlp(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    device=0,
    hidden_dim=32,
    num_epochs=2000,
    learning_rate=0.01,
    patience=50,
    class_weight=None,
):
    """Train and evaluate MLP model."""
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Initialize model, loss function, and optimizer
    model = MLP(input_dim, hidden_dim, output_dim)
    weight = None
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    # Training loop
    early_stop_counter = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            if epoch == 0:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        model.load_state_dict(best_model_state)
                        break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device)).cpu().numpy()

    return {"test_outputs": test_outputs}


def calculate_classification_performances(
    df: pd.DataFrame,
    phenotypes: list[str],
    embeddings: np.ndarray,
    n_seeds: int = 5,
    min_class_samples: int = 50,
    balance_loss: bool = True,
    model_name: str = "Unknown",
):
    """Calculate classification performances for different methods and phenotypes."""
    results_df = pd.DataFrame(
        columns=[
            "phenotype",
            "model_name",
            "seed",
            "f1_score_weighted",
            "f1_score_macro",
            "precision_weighted",
            "precision_macro",
            "precision_weighted",
            "precision_macro",
            "recall_weighted",
            "recall_macro",
            "accuracy",
            "balanced_accuracy",
            "auc_weighted",
            "auc_macro",
            "n_classes",
        ]
    )

    for i, phenotype in enumerate(phenotypes):
        print(f"Phenotype: {phenotype} ({i + 1}/{len(phenotypes)})")
        keep_mask = df[phenotype].notnull()
        df_phenotype = df.loc[keep_mask, phenotype]

        # Filter out classes with less than min_class_samples samples
        class_counts = df_phenotype.value_counts()
        for class_label, count in class_counts.items():
            if count < min_class_samples:
                keep_mask = keep_mask & (df.loc[:, phenotype] != class_label)
        df_phenotype = df.loc[keep_mask, phenotype]

        # Skip phenotypes with less than 2 classes
        if len(df_phenotype.unique()) < 2:
            continue

        # Run methods for each seed
        for seed in tqdm(range(n_seeds)):
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings[keep_mask], df_phenotype, random_state=seed, test_size=0.2, stratify=df_phenotype
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, random_state=seed, test_size=1 / 8, stratify=y_train
            )

            # Encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            y_test = le.transform(y_test)

            # Train methods
            class_weight = None
            if balance_loss:
                class_weight = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
            out = fit_predict_mlp(X_train, X_val, X_test, y_train, y_val, y_test, class_weight=class_weight)

            # Calculate metrics
            y_test_pred = out["test_outputs"]
            report = classification_report(y_test, np.argmax(y_test_pred, axis=1), output_dict=True)

            # Calculate AUC
            y_test_oh = np.eye(len(np.unique(y_train)))[y_test]
            auc_weighted = roc_auc_score(y_test_oh, y_test_pred, average="weighted", multi_class="ovr")
            auc_macro = roc_auc_score(y_test_oh, y_test_pred, average="macro", multi_class="ovr")

            # Calculate balanced accuracy
            balanced_accuracy = balanced_accuracy_score(y_test, np.argmax(y_test_pred, axis=1))

            # Store results
            results_df.loc[len(results_df)] = pd.Series(
                {
                    "phenotype": phenotype,
                    "model_name": model_name,
                    "seed": seed,
                    "f1_score_weighted": report["weighted avg"]["f1-score"],
                    "f1_score_macro": report["macro avg"]["f1-score"],
                    "precision_weighted": report["weighted avg"]["precision"],
                    "precision_macro": report["macro avg"]["precision"],
                    "recall_weighted": report["weighted avg"]["recall"],
                    "recall_macro": report["macro avg"]["recall"],
                    "accuracy": report["accuracy"],
                    "balanced_accuracy": balanced_accuracy,
                    "auc_weighted": auc_weighted,
                    "auc_macro": auc_macro,
                    "n_classes": len(np.unique(y_train)),
                }
            )

    return results_df


class ArgumentParser(Tap):
    """Argument parser for training Bacformer (Lightning version)."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_filepath: str
    labels_filepath: str
    output_dir: str
    n_seeds: int = 5
    model_name: str = "unknown_model"


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.input_filepath)
    assert df.columns[:2].tolist() == ["genome_name", args.model_name], (
        "genome_name and model_name should be the first two columns"
    )
    labels_df = pd.read_parquet(args.labels_filepath)
    merged_df = pd.merge(df, labels_df, on="genome_name", how="inner")

    embeddings = np.concatenate(df[args.model_name].tolist(), axis=1)

    # Run classification
    phenotypes = merged_df.iloc[:, len(df) :].columns
    results_df = calculate_classification_performances(
        df, phenotypes, embeddings, n_seeds=args.n_seeds, model_name=args.model_name
    )
    results_df.to_csv(os.path.join(args.output_dir, f"results_{args.model_name}.csv"), index=False)
