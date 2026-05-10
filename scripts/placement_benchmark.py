# benchmark_placement.py
import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from torch.serialization import safe_globals
def load_dataset(graph_path: str) -> Tuple[Data, dict]:
    with safe_globals([Data]):
        data = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False
        )
    if isinstance(data, dict) and "graph" in data:
        graph = data["graph"]
        metadata = data.get("metadata", {})
        return graph, metadata

    if isinstance(data, Data):
        return data, {}

    raise ValueError("Unsupported .pt file format")
def load_and_benchmark(graph_path: str, canvas_size: tuple = (400.0, 400.0)):
    graph, _ = load_dataset(graph_path)
    pos_norm = graph.pos.numpy()
    pos_abs = pos_norm * np.array(canvas_size)
    sizes = graph.x[:, :2].numpy()  # width, height

    # 1. HPWL
    if graph.edge_index.numel() > 0:
        src, dst = graph.edge_index.numpy()
        dx = np.abs(pos_abs[src, 0] - pos_abs[dst, 0])
        dy = np.abs(pos_abs[src, 1] - pos_abs[dst, 1])
        hpwl = np.sum(dx + dy) / 2
    else:
        hpwl = 0.0
    print(f"HPWL: {hpwl:.2f} μm")

    # 2. Simple Density (10x10 grid)
    grid_bins = 10
    grid = np.zeros((grid_bins, grid_bins))
    for i in range(graph.num_nodes):
        w, h = sizes[i]
        x, y = pos_norm[i]
        # Bin indices
        bin_x = np.clip(np.floor(x * grid_bins), 0, grid_bins-1).astype(int)
        bin_y = np.clip(np.floor(y * grid_bins), 0, grid_bins-1).astype(int)
        # Approximate area contribution
        macro_area = w * h
        grid[bin_y, bin_x] += macro_area  # Or +1 for count-based
    total_area = np.sum(sizes[:, 0] * sizes[:, 1])
    bin_area = (canvas_size[0] / grid_bins) * (canvas_size[1] / grid_bins)
    density = grid / bin_area
    max_density = np.max(density)
    avg_density = np.mean(density)
    print(f"Max Density: {max_density:.2f}x ({max_density*100:.1f}%)")
    print(f"Avg Density: {avg_density:.2f}x")

    # Visualize density heatmap
    plt.figure(figsize=(8,8))
    plt.imshow(density.T, origin='lower', extent=[0, canvas_size[0], 0, canvas_size[1]], cmap='hot')
    plt.colorbar(label='Density')
    plt.title("Density Heatmap")
    plt.show()

    return {"hpwl": hpwl, "max_density": max_density, "avg_density": avg_density}

# Usage
metrics_initial = load_and_benchmark("src/data/preprocessed/real-connection/ariane136/Nangate45/ariane136_Nangate45_graph.pt")
metrics_trained = load_and_benchmark("trained_placement_MemPool_tile.pt")
print(f"HPWL Improvement: {(metrics_initial['hpwl'] - metrics_trained['hpwl']) / metrics_initial['hpwl'] * 100:.1f}%")