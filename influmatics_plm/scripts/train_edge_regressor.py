"""Train distance regression on frozen embeddings and pre-split edge CSVs.

CLI embeddings use an NPZ archive: each key is a string node ID and each value
is a (1280,) array. The Python API also accepts dictionaries of torch tensors.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EdgeDataset(Dataset):
    def __init__(self, edge_df, embeddings_dict):
        required = {"node_A", "node_B", "distance"}
        if not edge_df.columns.is_unique or not required.issubset(edge_df.columns):
            raise ValueError("edge_df needs unique columns node_A, node_B, distance")
        if edge_df[["node_A", "node_B"]].isna().to_numpy().any():
            raise ValueError("Node IDs must not be missing")
        targets = edge_df["distance"].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("Distances must be finite and nonnegative")

        nodes = list(dict.fromkeys(
            edge_df["node_A"].tolist() + edge_df["node_B"].tolist()
        ))
        lookup = {node: index for index, node in enumerate(nodes)}
        self.embeddings = torch.empty((len(nodes), 1280), dtype=torch.float32)
        for index, node in enumerate(nodes):
            if node not in embeddings_dict:
                raise KeyError(f"Missing embedding for node {node!r}")
            value = embeddings_dict[node]
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Embedding for {node!r} must be ndarray or Tensor")
            if value.shape != (1280,):
                raise ValueError(f"Embedding for {node!r} must have shape (1280,)")
            if (torch.is_complex(value) if isinstance(value, torch.Tensor)
                    else np.iscomplexobj(value)):
                raise ValueError(f"Embedding for {node!r} must be real")
            # Own a frozen CPU copy; never train or mutate source embeddings.
            tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Embedding for {node!r} contains NaN or infinity")
            self.embeddings[index].copy_(tensor)
        self.node_a = torch.tensor([lookup[n] for n in edge_df["node_A"]], dtype=torch.long)
        self.node_b = torch.tensor([lookup[n] for n in edge_df["node_B"]], dtype=torch.long)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        diff = torch.abs(self.embeddings[self.node_a[index]]
                         - self.embeddings[self.node_b[index]])
        return diff, self.targets[index]


class DistanceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), nn.Softplus(),
        )

    def forward(self, diff):
        return self.network(diff).squeeze(-1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.SmoothL1Loss()
    total_loss, count = 0.0, 0
    for diff, target in loader:
        diff, target = diff.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(diff), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.numel()
        count += target.numel()
    if not count:
        raise ValueError("Training loader is empty")
    return total_loss / count


@torch.inference_mode()
def evaluate(model, loader, device):
    """Compute dataset-wide metrics; undefined Spearman rho is NaN."""
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="sum")
    predictions, targets = [], []
    total_loss = 0.0
    for diff, target in loader:
        diff, target = diff.to(device), target.to(device)
        pred = model(diff)
        loss = criterion(pred, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        total_loss += loss.item()
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    if not targets:
        raise ValueError("Evaluation loader is empty")
    pred = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    error = pred - target
    rho = float("nan")
    if len(target) >= 2 and np.ptp(target) > 0 and np.ptp(pred) > 0:
        rho = float(spearmanr(target, pred).statistic)
    return {
        "loss": total_loss / len(target),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "spearman_rho": rho,
        "n_samples": len(target),
    }


def fit(train_df, val_df, embeddings_dict, *, checkpoint_path="best_edge_model.pt",
        epochs=100, batch_size=256, patience=10, min_delta=0.0, seed=42,
        device=None, num_workers=0):
    """Train, save the lowest validation-loss checkpoint, and restore it.

    min_delta controls patience resets; every strict validation-loss minimum
    is still saved. Splits are supplied by the caller and are never randomized.
    The checkpoint supports inference; it is not a training-resumption snapshot.
    """
    if epochs < 1 or batch_size < 1 or patience < 1 or num_workers < 0:
        raise ValueError("epochs, batch_size, patience must be positive; workers >= 0")
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    if train_df.empty or val_df.empty:
        raise ValueError("Train and validation splits must both be nonempty")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_data = EdgeDataset(train_df, embeddings_dict)
    val_data = EdgeDataset(val_df, embeddings_dict)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    model = DistanceRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, patience_reference = float("inf"), float("inf")
    best_epoch, stale_epochs = 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in metrics.items()}})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} "
              f"val_loss={metrics['loss']:.6f} val_rmse={metrics['rmse']:.6f}")
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch
            torch.save({
                "model_state_dict": {k: v.detach().cpu()
                                     for k, v in model.state_dict().items()},
                "epoch": epoch, "val_loss": best_loss,
                "embedding_dim": 1280, "seed": seed,
            }, checkpoint_path)
        if metrics["loss"] < patience_reference - min_delta:
            patience_reference, stale_epochs = metrics["loss"], 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "best_epoch": best_epoch, "epochs_run": len(history),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "validation": evaluate(model, val_loader, device), "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--embeddings", required=True, help="NPZ with string node-ID keys")
    parser.add_argument("--checkpoint", default="best_edge_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="cpu, cuda, or mps; default: CUDA if available, else CPU")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    def read_edges(path):
        return pd.read_csv(path, dtype={"node_A": str, "node_B": str})

    with np.load(args.embeddings, allow_pickle=False) as archive:
        embeddings = {node: archive[node] for node in archive.files}
    train_df, val_df = read_edges(args.train_csv), read_edges(args.val_csv)
    test_df = read_edges(args.test_csv) if args.test_csv else None
    # Preserve the preceding temporal splitter's test-node holdout invariant.
    if test_df is not None:
        test_nodes = set(test_df.node_A) | set(test_df.node_B)
        for name, frame in (("train", train_df), ("validation", val_df)):
            if test_nodes & (set(frame.node_A) | set(frame.node_B)):
                raise ValueError(f"Test nodes overlap with {name} nodes")
        test_data = EdgeDataset(test_df, embeddings)
        if not len(test_data):
            raise ValueError("Test split must be nonempty when supplied")
    model, result = fit(
        train_df, val_df, embeddings, checkpoint_path=args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        min_delta=args.min_delta, seed=args.seed, device=args.device,
        num_workers=args.num_workers,
    )
    if test_df is not None:
        result["test"] = evaluate(
            model, DataLoader(test_data, batch_size=args.batch_size, shuffle=False),
            next(model.parameters()).device,
        )
    # Represent undefined correlations as JSON null rather than nonstandard NaN.
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    metrics_path = Path(args.checkpoint).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False)
                            + "\n", encoding="utf-8")
    print(json.dumps(json_safe({k: v for k, v in result.items() if k != "history"}),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
