from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.serialization import safe_globals
from torch_geometric.data import Data

from _common import PROJECT_ROOT, resolve_project_path
from src.models.gnn.encoder import GNNEncoder


class GNNPlacementRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_channels, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x, edge_index)
        return torch.sigmoid(self.head(self.dropout(embeddings)))


def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


def load_gnn_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def graph_paths_from_args(paths: list[str], graph_glob: str | None) -> list[Path]:
    resolved = [resolve_project_path(path) for path in paths]
    if graph_glob:
        resolved.extend(sorted(PROJECT_ROOT.glob(graph_glob)))
    unique = []
    seen = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise ValueError("Provide at least one graph path or --graph-glob.")
    return unique


def graph_features(graph: Data, include_positions: bool) -> torch.Tensor:
    x = graph.x.float()
    if include_positions:
        x = torch.cat([x, graph.pos.float()], dim=1)
    return x


def feature_stats(graphs: list[Data], include_positions: bool) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat([graph_features(graph, include_positions) for graph in graphs], dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalized_features(graph: Data, include_positions: bool, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (graph_features(graph, include_positions) - mean) / std


def evaluate(model: nn.Module, graphs: list[Data], include_positions: bool, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for graph in graphs:
            x = normalized_features(graph, include_positions, mean, std).to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.pos.float().to(device)
            pred = model(x, edge_index)
            losses.append(F.mse_loss(pred, target).item())
            maes.append(F.l1_loss(pred, target).item())
    return {"mse": float(sum(losses) / len(losses)), "mae": float(sum(maes) / len(maes))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GNN macro-position regressor.")
    parser.add_argument("graphs", nargs="*", help="Training graph .pt files.")
    parser.add_argument("--graph-glob", default=None, help="Project-root glob for additional training graphs.")
    parser.add_argument("--gnn-config", default="configs/gnn.yaml", help="Path to GNN YAML config.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay.")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Override hidden channels.")
    parser.add_argument("--out-channels", type=int, default=None, help="Override encoder output channels.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout.")
    parser.add_argument("--include-position-features", action="store_true", help="Append current positions to node features.")
    parser.add_argument("--device", default="cpu", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint", default="checkpoints/gnn_position_regressor.pt", help="Output checkpoint path.")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save final metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = load_gnn_config(args.gnn_config)
    architecture = config.get("encoder", {}).get("architecture", {})
    training = config.get("training", {})

    paths = graph_paths_from_args(args.graphs, args.graph_glob)
    graphs = [load_dataset(path)[0] for path in paths]
    mean, std = feature_stats(graphs, args.include_position_features)
    in_channels = int(mean.numel())
    hidden_channels = int(args.hidden_channels or architecture.get("hidden_channels", 128))
    out_channels = int(args.out_channels or architecture.get("out_channels", 256))
    learning_rate = float(args.learning_rate or training.get("learning_rate", 3e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else training.get("weight_decay", 0.0))
    dropout = float(args.dropout if args.dropout is not None else training.get("dropout", 0.0))
    grad_clip = training.get("gradient_clip_norm", 0.5)

    model = GNNPlacementRegressor(in_channels, hidden_channels, out_channels, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for graph in graphs:
            x = normalized_features(graph, args.include_position_features, mean, std).to(device)
            edge_index = graph.edge_index.to(device)
            target = graph.pos.float().to(device)

            pred = model(x, edge_index)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            epoch_losses.append(loss.item())

        record = {"epoch": epoch, "train_mse": float(sum(epoch_losses) / len(epoch_losses))}
        history.append(record)
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"epoch={epoch} train_mse={record['train_mse']:.6f}")

    metrics = evaluate(model, graphs, args.include_position_features, mean, std, device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "dropout": dropout,
            "include_position_features": args.include_position_features,
            "feature_mean": mean,
            "feature_std": std,
            "history": history,
            "metrics": metrics,
            "graph_paths": [str(path) for path in paths],
        },
        checkpoint_path,
    )

    if args.metrics_json:
        metrics_path = resolve_project_path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"metrics": metrics, "history": history}, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"GNN checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
