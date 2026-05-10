# layout_visualizer.py
"""
Simple Macro Placement Layout Visualizer for your Hierarchical MARL Chip Placement Project

Usage:
  python layout_visualizer.py --graph-path preprocessed_dataset/ariane136/Nangate45/ariane136_Nangate45_graph.pt
  python layout_visualizer.py --graph-path path/to/your_graph.pt --title "Ariane136 Initial Placement" --save plot.png

Features:
- Loads your PyTorch Geometric .pt graph (direct Data object)
- Plots canvas as background rectangle
- Draws each macro as a scaled rectangle (real sizes from features)
- Draws real/synthetic connections as thin gray lines between macro centers
- Random initial positions (or placed if you update graph.pos in RL and re-save)
- Saves to PNG or shows interactively
- Works for uniform (Ariane) or heterogeneous (MemPool/BlackParrot) macros

Requirements: pip install torch torch-geometric matplotlib
"""

import argparse
from email import parser
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, Tuple
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

def visualize_layout(graph: Data, title: str = "Macro Placement Layout", save_path: Optional[str] = None):
    # Extract canvas
    canvas_w, canvas_h = graph.canvas_size.tolist() if hasattr(graph, 'canvas_size') else (400.0, 400.0)
    print(f"Canvas: {canvas_w} x {canvas_h} μm")

    # Positions: normalized [0,1] -> absolute μm
    pos = graph.pos.numpy()  # [N, 2] normalized
    abs_pos = pos * [canvas_w, canvas_h]  # Scale to canvas

    # Macro sizes from node features: x[:, :2] = width, height
    sizes = graph.x[:, :2].numpy()  # [N, 2] in μm

    # Centers for edges
    centers = abs_pos  # For simplicity, connect centers

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Typical EDA coord: origin bottom-left, y up
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Width (μm)")
    ax.set_ylabel("Height (μm)")

    # Background canvas
    ax.add_patch(patches.Rectangle((0, 0), canvas_w, canvas_h, linewidth=3, edgecolor='black', facecolor='none', label='Die Area'))

    # Draw edges (thin, semi-transparent for dense graphs)
    if graph.edge_index.numel() > 0:
        edge_index = graph.edge_index.numpy()
        for i in range(0, edge_index.shape[1], 2):  # Undirected
            src, dst = edge_index[0, i], edge_index[1, i]
            x_vals = [centers[src, 0], centers[dst, 0]]
            y_vals = [centers[src, 1], centers[dst, 1]]
            ax.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=0.5)

    # Draw macros as rectangles (colored randomly for distinction)
    colors = plt.cm.tab20(range(graph.num_nodes))  # Distinct colors
    for i in range(graph.num_nodes):
        w, h = sizes[i]
        x, y = abs_pos[i]
        # Bottom-left corner for rectangle
        bottom_left_x = x - w / 2
        bottom_left_y = y - h / 2
        rect = patches.Rectangle((bottom_left_x, bottom_left_y), w, h,
                                 linewidth=2, edgecolor='darkblue', facecolor=colors[i], alpha=0.8,
                                 label=f"Macro {i}" if i < 10 else None)  # Label only first few
        ax.add_patch(rect)
        # Optional: Label macro index
        ax.text(x, y, str(i), ha='center', va='center', fontsize=8, color='white')

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layout saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize macro placement layout from your preprocessed .pt graph")
    parser.add_argument("--graph-path", type=str, required=True, help="Path to your _graph.pt file")
    parser.add_argument("--design-name", type=str, default="Macro Placement Layout", help="Plot title")
    parser.add_argument("--title", type=str, default="Macro Placement Layout", help="Plot title")
    parser.add_argument("--save", type=str, default=None, help="Save plot to PNG file (e.g., initial.png)")
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    graph, _ = load_dataset(str(graph_path))
    visualize_layout(graph, title=args.title, save_path=f"{args.save}/{args.design_name}.png" if args.save else None)

if __name__ == "__main__":
    main()