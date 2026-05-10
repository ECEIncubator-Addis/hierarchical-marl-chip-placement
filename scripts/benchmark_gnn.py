from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.serialization import safe_globals
from torch_geometric.data import Data

from _common import PROJECT_ROOT, resolve_project_path
from scripts.placement_benchmark import compute_metrics
from scripts.train_gnn import GNNPlacementRegressor, graph_features


def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


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


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[GNNPlacementRegressor, dict[str, Any]]:
    checkpoint = torch.load(resolve_project_path(checkpoint_path), map_location=device)
    model = GNNPlacementRegressor(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_graph(model: GNNPlacementRegressor, checkpoint: dict[str, Any], graph: Data, device: torch.device) -> torch.Tensor:
    include_positions = bool(checkpoint.get("include_position_features", False))
    mean = checkpoint["feature_mean"].to(device)
    std = checkpoint["feature_std"].to(device)
    x = graph_features(graph, include_positions).to(device)
    x = (x - mean) / std
    with torch.no_grad():
        return model(x, graph.edge_index.to(device)).cpu()


def evaluate_graph(model: GNNPlacementRegressor, checkpoint: dict[str, Any], graph: Data, device: torch.device) -> tuple[dict[str, float], Data]:
    target = graph.pos.float().cpu()
    pred = predict_graph(model, checkpoint, graph, device)
    pred_graph = graph.clone()
    pred_graph.pos = pred.clamp(0.0, 1.0)

    target_metrics = compute_metrics(graph)
    pred_metrics = compute_metrics(pred_graph)
    hpwl_gap = pred_metrics["hpwl"] - target_metrics["hpwl"]
    hpwl_gap_pct = hpwl_gap / target_metrics["hpwl"] * 100.0 if target_metrics["hpwl"] else 0.0

    return (
        {
            "position_mse": float(F.mse_loss(pred, target).item()),
            "position_mae": float(F.l1_loss(pred, target).item()),
            "target_hpwl": target_metrics["hpwl"],
            "predicted_hpwl": pred_metrics["hpwl"],
            "hpwl_gap": float(hpwl_gap),
            "hpwl_gap_pct": float(hpwl_gap_pct),
            "target_max_density": target_metrics["max_density"],
            "predicted_max_density": pred_metrics["max_density"],
        },
        pred_graph,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a trained GNN macro-position regressor.")
    parser.add_argument("--checkpoint", default="checkpoints/gnn_position_regressor.pt", help="GNN checkpoint path.")
    parser.add_argument("graphs", nargs="*", help="Graph .pt files to benchmark.")
    parser.add_argument("--graph-glob", default=None, help="Project-root glob for additional graphs.")
    parser.add_argument("--device", default="cpu", help="Benchmark device.")
    parser.add_argument("--json", default=None, help="Optional metrics JSON output path.")
    parser.add_argument("--save-predicted-graph", default=None, help="Optional path for the first predicted graph.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    paths = graph_paths_from_args(args.graphs, args.graph_glob)

    results = {}
    first_predicted_graph = None
    for path in paths:
        graph, _ = load_dataset(path)
        metrics, predicted_graph = evaluate_graph(model, checkpoint, graph, device)
        results[str(path)] = metrics
        if first_predicted_graph is None:
            first_predicted_graph = predicted_graph

    print(json.dumps(results, indent=2))

    if args.json:
        json_path = resolve_project_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.save_predicted_graph and first_predicted_graph is not None:
        graph_path = resolve_project_path(args.save_predicted_graph)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graph": first_predicted_graph, "source": "gnn_position_regressor", "metrics": next(iter(results.values()))}, graph_path)
        print(f"Predicted graph saved to: {graph_path}")


if __name__ == "__main__":
    main()
