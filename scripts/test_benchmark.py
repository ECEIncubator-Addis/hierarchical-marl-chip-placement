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


# ============================================================
# Loading
# ============================================================

def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    graph_path = resolve_project_path(graph_path)

    with safe_globals([Data]):
        data = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )

    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})

    if isinstance(data, Data):
        return data, {}

    raise ValueError(f"Unsupported graph format: {graph_path}")


# ============================================================
# Validation
# ============================================================

def validate_graph(graph: Data, graph_name: str) -> None:
    required_attrs = ["x", "pos", "edge_index"]

    for attr in required_attrs:
        if not hasattr(graph, attr):
            raise ValueError(f"{graph_name} missing required attribute: {attr}")

    if graph.pos is None:
        raise ValueError(f"{graph_name} has no node positions")

    if graph.x.shape[1] < 2:
        raise ValueError(
            f"{graph_name} node feature matrix must contain width/height"
        )


# ============================================================
# Metrics
# ============================================================

def get_canvas_size(
    graph: Data,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, float]:

    # --------------------------------------------------------
    # Case 1:
    # canvas embedded directly in graph
    # --------------------------------------------------------

    if hasattr(graph, "canvas_size"):

        canvas = graph.canvas_size

        if isinstance(canvas, torch.Tensor):
            return tuple(float(v) for v in canvas.tolist())

        return tuple(float(v) for v in canvas)

    # --------------------------------------------------------
    # Case 2:
    # metadata json
    # --------------------------------------------------------

    if metadata is not None:

        # Flat format
        if "canvas_size" in metadata:
            return tuple(float(v) for v in metadata["canvas_size"])

        # Width/height format
        if (
            "canvas_width" in metadata and
            "canvas_height" in metadata
        ):
            return (
                float(metadata["canvas_width"]),
                float(metadata["canvas_height"]),
            )

        # Nested known_info format
        known_info = metadata.get("known_info", {})

        if "canvas_size_um" in known_info:
            return tuple(
                float(v)
                for v in known_info["canvas_size_um"]
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return (400.0, 400.0)

def compute_hpwl(
    graph: Data,
    canvas_size: tuple[float, float],
) -> float:
    if graph.edge_index.numel() == 0:
        return 0.0

    pos = graph.pos.detach().cpu().numpy()
    pos_abs = pos * np.array(canvas_size)

    src, dst = graph.edge_index.detach().cpu().numpy()

    hpwl = np.sum(
        np.abs(pos_abs[src, 0] - pos_abs[dst, 0]) +
        np.abs(pos_abs[src, 1] - pos_abs[dst, 1])
    )

    return float(hpwl / 2.0)


def compute_density_map(
    graph: Data,
    canvas_size: tuple[float, float],
    grid_bins: int,
) -> np.ndarray:
    pos = graph.pos.detach().cpu().numpy()
    sizes = graph.x[:, :2].detach().cpu().numpy()

    grid = np.zeros((grid_bins, grid_bins), dtype=np.float64)

    for idx in range(graph.num_nodes):
        x_bin = int(np.clip(
            np.floor(pos[idx, 0] * grid_bins),
            0,
            grid_bins - 1,
        ))

        y_bin = int(np.clip(
            np.floor(pos[idx, 1] * grid_bins),
            0,
            grid_bins - 1,
        ))

        macro_area = float(sizes[idx, 0] * sizes[idx, 1])

        grid[y_bin, x_bin] += macro_area

    bin_area = (
        (canvas_size[0] / grid_bins) *
        (canvas_size[1] / grid_bins)
    )

    return grid / bin_area


def compute_overlap(graph: Data) -> float:
    """
    Approximate pairwise overlap area.
    """

    pos = graph.pos.detach().cpu().numpy()
    sizes = graph.x[:, :2].detach().cpu().numpy()

    overlap = 0.0

    for i in range(graph.num_nodes):
        xi, yi = pos[i]
        wi, hi = sizes[i]

        left_i = xi - wi / 2
        right_i = xi + wi / 2
        bottom_i = yi - hi / 2
        top_i = yi + hi / 2

        for j in range(i + 1, graph.num_nodes):
            xj, yj = pos[j]
            wj, hj = sizes[j]

            left_j = xj - wj / 2
            right_j = xj + wj / 2
            bottom_j = yj - hj / 2
            top_j = yj + hj / 2

            dx = min(right_i, right_j) - max(left_i, left_j)
            dy = min(top_i, top_j) - max(bottom_i, bottom_j)

            if dx > 0 and dy > 0:
                overlap += dx * dy

    return float(overlap)


def compute_metrics(
    graph: Data,
    grid_bins: int = 10,
) -> dict[str, float]:

    canvas_size = tuple(
        graph.canvas_size.tolist()
        if hasattr(graph, "canvas_size")
        else (400.0, 400.0)
    )

    density = compute_density_map(
        graph,
        canvas_size,
        grid_bins,
    )

    sizes = graph.x[:, :2].detach().cpu().numpy()

    return {
        "hpwl": compute_hpwl(graph, canvas_size),
        "max_density": float(np.max(density)),
        "avg_density": float(np.mean(density)),
        "std_density": float(np.std(density)),
        "overlap_area": compute_overlap(graph),
        "total_macro_area": float(
            np.sum(sizes[:, 0] * sizes[:, 1])
        ),
    }


# ============================================================
# Visualization
# ============================================================

def save_density_heatmap(
    density: np.ndarray,
    save_path: str | Path,
    title: str,
) -> None:

    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        density.T,
        origin="lower",
        cmap="hot",
    )

    plt.colorbar(label="Density")

    plt.title(title)

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Benchmark original vs RL-assisted placement graphs."
    )

    parser.add_argument(
        "--original",
        required=True,
        help="Original design graph (.pt)",
    )

    parser.add_argument(
        "--rl-placed",
        required=True,
        help="RL optimized placement graph (.pt)",
    )

    parser.add_argument(
        "--grid-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--json",
        default=None,
        help="Optional output JSON path",
    )

    parser.add_argument(
        "--heatmap-dir",
        default=None,
        help="Optional heatmap output directory",
    )

    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional metadata JSON path (for canvas size)",
    )

    args = parser.parse_args()

    original_graph, _ = load_dataset(args.original)
    rl_graph, _ = load_dataset(args.rl_placed)
    with open (resolve_project_path(args.metadata_json), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    validate_graph(original_graph, "original_graph")
    validate_graph(rl_graph, "rl_graph")

    if original_graph.num_nodes != rl_graph.num_nodes:
        raise ValueError(
            "Graphs are incompatible: node counts differ"
        )

    original_metrics = compute_metrics(
        original_graph,
        args.grid_bins,
    )

    rl_metrics = compute_metrics(
        rl_graph,
        args.grid_bins,
    )

    results = {
        "original": original_metrics,
        "rl_optimized": rl_metrics,
    }

    print(json.dumps(results, indent=2))

    # ========================================================
    # Improvements
    # ========================================================

    baseline_hpwl = original_metrics["hpwl"]

    if baseline_hpwl > 0:
        hpwl_improvement = (
            (baseline_hpwl - rl_metrics["hpwl"]) /
            baseline_hpwl
        ) * 100.0

        print(
            f"\nHPWL Improvement: "
            f"{hpwl_improvement:.2f}%"
        )

    baseline_overlap = original_metrics["overlap_area"]

    if baseline_overlap > 0:
        overlap_reduction = (
            (baseline_overlap - rl_metrics["overlap_area"]) /
            baseline_overlap
        ) * 100.0

        print(
            f"Overlap Reduction: "
            f"{overlap_reduction:.2f}%"
        )

    # ========================================================
    # Save JSON
    # ========================================================

    if args.json:
        json_path = resolve_project_path(args.json)

        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved metrics JSON to: {json_path}")

    # ========================================================
    # Heatmaps
    # ========================================================

    if args.heatmap_dir:

        heatmap_dir = resolve_project_path(args.heatmap_dir)

        heatmap_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        original_density = compute_density_map(
            original_graph,
            get_canvas_size(original_graph, metadata),
            args.grid_bins,
        )

        rl_density = compute_density_map(
            rl_graph,
            get_canvas_size(rl_graph, metadata),
            args.grid_bins,
        )

        save_density_heatmap(
            original_density,
            heatmap_dir / "original_density.png",
            "Original Placement Density",
        )

        save_density_heatmap(
            rl_density,
            heatmap_dir / "rl_density.png",
            "RL Placement Density",
        )

        print(f"\nHeatmaps saved to: {heatmap_dir}")


if __name__ == "__main__":
    main()