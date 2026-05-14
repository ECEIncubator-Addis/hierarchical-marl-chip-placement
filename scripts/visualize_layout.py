from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.serialization import safe_globals
from torch_geometric.data import Data

from _common import load_agent, make_env, resolve_project_path
from src.rl import TrainerConfig, rollout_placement


def load_dataset(graph_path: str | Path) -> tuple[Data, dict[str, Any]]:
    with safe_globals([Data]):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"], data.get("metadata", {})
    if isinstance(data, Data):
        return data, {}
    raise ValueError(f"Unsupported .pt file format: {graph_path}")


def visualize_layout(graph: Data, title: str = "Macro Placement Layout", save_path: str | Path | None = None) -> None:
    canvas_w, canvas_h = graph.canvas_size.tolist() if hasattr(graph, "canvas_size") else (400.0, 400.0)
    pos = graph.pos.detach().cpu().numpy()
    abs_pos = pos * [canvas_w, canvas_h]
    sizes = graph.x[:, :2].detach().cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Width (um)")
    ax.set_ylabel("Height (um)")
    ax.add_patch(patches.Rectangle((0, 0), canvas_w, canvas_h, linewidth=2, edgecolor="black", facecolor="none"))

    if graph.edge_index.numel() > 0:
        edge_index = graph.edge_index.detach().cpu().numpy()
        for edge_idx in range(edge_index.shape[1]):
            src, dst = edge_index[:, edge_idx]
            ax.plot(
                [abs_pos[src, 0], abs_pos[dst, 0]],
                [abs_pos[src, 1], abs_pos[dst, 1]],
                color="0.55",
                alpha=0.25,
                linewidth=0.5,
            )

    colors = plt.cm.tab20(range(graph.num_nodes))
    for idx in range(graph.num_nodes):
        width, height = sizes[idx]
        x, y = abs_pos[idx]
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=1.5,
            edgecolor="black",
            facecolor=colors[idx % len(colors)],
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, color="white")

    if save_path:
        save_path = resolve_project_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Layout saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a graph or a trained RL rollout placement.")
    parser.add_argument("--graph-path", default=None, help="Graph .pt file to visualize.")
    parser.add_argument("--checkpoint", default=None, help="Optional agent checkpoint for rollout visualization.")
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "dqn"], default="ppo", help="Checkpoint algorithm.")
    parser.add_argument("--rl-config", default="configs/rl.yaml", help="Path to RL YAML config.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to environment YAML config.")
    parser.add_argument("--training-config", default="configs/training.yaml", help="Path to training YAML config.")
    parser.add_argument("--device", default=None, help="Rollout device; defaults to trainer config.")
    parser.add_argument("--title", default="Macro Placement Layout", help="Plot title.")
    parser.add_argument("--save", default=None, help="Optional PNG output path.")
    args = parser.parse_args()

    if args.checkpoint:
        config = TrainerConfig.from_yaml(args.rl_config, args.env_config, args.training_config)
        if args.graph_path:
            config.graph_path = str(resolve_project_path(args.graph_path))
        if args.device:
            config.device = args.device
        agent = load_agent(args.checkpoint, args.algorithm, device=config.device)
        env = make_env(config, eval_env=True)
        graph, result = rollout_placement(agent, env, deterministic=True)
        print(f"Rollout reward: {result.mean_reward:.4f}, HPWL: {result.best_hpwl}")
    else:
        if not args.graph_path:
            parser.error("--graph-path is required unless --checkpoint is provided.")
        graph, _ = load_dataset(resolve_project_path(args.graph_path))

    visualize_layout(graph, title=args.title, save_path=args.save)


if __name__ == "__main__":
    main()
