from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.serialization import safe_globals
from torch_geometric.data import Data

from _common import resolve_project_path


def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


def compute_metrics(graph: Data, canvas_size: tuple[float, float] | None = None, grid_bins: int = 10) -> dict[str, float]:
    canvas = canvas_size or tuple(graph.canvas_size.tolist() if hasattr(graph, "canvas_size") else (400.0, 400.0))
    pos_norm = graph.pos.detach().cpu().numpy()
    pos_abs = pos_norm * np.array(canvas)
    sizes = graph.x[:, :2].detach().cpu().numpy()

    if graph.edge_index.numel() > 0:
        src, dst = graph.edge_index.detach().cpu().numpy()
        hpwl = float(np.sum(np.abs(pos_abs[src, 0] - pos_abs[dst, 0]) + np.abs(pos_abs[src, 1] - pos_abs[dst, 1])) / 2.0)
    else:
        hpwl = 0.0

    grid = np.zeros((grid_bins, grid_bins), dtype=np.float64)
    for idx in range(graph.num_nodes):
        bin_x = int(np.clip(np.floor(pos_norm[idx, 0] * grid_bins), 0, grid_bins - 1))
        bin_y = int(np.clip(np.floor(pos_norm[idx, 1] * grid_bins), 0, grid_bins - 1))
        grid[bin_y, bin_x] += float(sizes[idx, 0] * sizes[idx, 1])

    bin_area = (canvas[0] / grid_bins) * (canvas[1] / grid_bins)
    density = grid / bin_area
    return {
        "hpwl": hpwl,
        "max_density": float(np.max(density)),
        "avg_density": float(np.mean(density)),
        "total_macro_area": float(np.sum(sizes[:, 0] * sizes[:, 1])),
    }


def save_density_heatmap(graph: Data, save_path: str | Path, grid_bins: int = 10) -> None:
    canvas = tuple(graph.canvas_size.tolist() if hasattr(graph, "canvas_size") else (400.0, 400.0))
    pos_norm = graph.pos.detach().cpu().numpy()
    sizes = graph.x[:, :2].detach().cpu().numpy()
    grid = np.zeros((grid_bins, grid_bins), dtype=np.float64)
    for idx in range(graph.num_nodes):
        bin_x = int(np.clip(np.floor(pos_norm[idx, 0] * grid_bins), 0, grid_bins - 1))
        bin_y = int(np.clip(np.floor(pos_norm[idx, 1] * grid_bins), 0, grid_bins - 1))
        grid[bin_y, bin_x] += float(sizes[idx, 0] * sizes[idx, 1])
    density = grid / ((canvas[0] / grid_bins) * (canvas[1] / grid_bins))

    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(density.T, origin="lower", extent=[0, canvas[0], 0, canvas[1]], cmap="hot")
    plt.colorbar(label="Density")
    plt.title("Density Heatmap")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HPWL and density for placement graph files.")
    parser.add_argument("graphs", nargs="+", help="Graph .pt files to benchmark.")
    parser.add_argument("--grid-bins", type=int, default=10, help="Density grid resolution.")
    parser.add_argument("--json", default=None, help="Optional output metrics JSON path.")
    parser.add_argument("--heatmap", default=None, help="Optional density heatmap PNG path for the first graph.")
    args = parser.parse_args()

    results = {}
    first_graph = None
    for graph_path in args.graphs:
        resolved = resolve_project_path(graph_path)
        graph, _ = load_dataset(resolved)
        first_graph = first_graph or graph
        results[str(resolved)] = compute_metrics(graph, grid_bins=args.grid_bins)

    print(json.dumps(results, indent=2))

    if len(args.graphs) == 2:
        first_metrics, second_metrics = results.values()
        baseline = first_metrics["hpwl"]
        if baseline:
            improvement = (baseline - second_metrics["hpwl"]) / baseline * 100.0
            print(f"HPWL improvement from first to second graph: {improvement:.2f}%")

    if args.json:
        json_path = resolve_project_path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.heatmap and first_graph is not None:
        save_density_heatmap(first_graph, args.heatmap, grid_bins=args.grid_bins)
        print(f"Density heatmap saved to: {resolve_project_path(args.heatmap)}")


if __name__ == "__main__":
    main()
