from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
import yaml

from _common import PROJECT_ROOT, resolve_project_path
from src.rl import HierarchicalRLTrainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a hierarchical RL macro-placement agent.")
    parser.add_argument("--rl-config", default="configs/rl.yaml", help="Path to RL YAML config.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to environment YAML config.")
    parser.add_argument("--training-config", default="configs/training.yaml", help="Path to training YAML config.")
    parser.add_argument("--graph-path", default=None, help="Override graph path from config.")
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "dqn"], default=None, help="Override algorithm.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory.")
    parser.add_argument("--device", default=None, help="Override training device.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--use-gnn-encoder", action=argparse.BooleanOptionalAction, default=None, help="Use the GNN representation encoder for PPO/A2C.")
    parser.add_argument("--save-final-graph", default=None, help="Optional path for final rollout graph.")
    parser.add_argument("--history-json", default=None, help="Optional path to save training history as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainerConfig.from_yaml(args.rl_config, args.env_config, args.training_config)

    if args.graph_path:
        config.graph_path = str(resolve_project_path(args.graph_path))
    if args.algorithm:
        config.algorithm = args.algorithm
        with resolve_project_path(args.rl_config).open("r", encoding="utf-8") as handle:
            rl_config = yaml.safe_load(handle) or {}
        config.algorithm_config = dict(rl_config.get("algorithms", {}).get(args.algorithm, config.algorithm_config))
    if args.use_gnn_encoder is not None:
        config.algorithm_config = dict(config.algorithm_config)
        config.algorithm_config["use_gnn_encoder"] = args.use_gnn_encoder
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.checkpoint_dir:
        config.checkpoint_dir = str(resolve_project_path(args.checkpoint_dir))
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    trainer = HierarchicalRLTrainer(config)
    history = trainer.train()

    final_model_path = trainer.checkpoint_dir / f"final_{config.algorithm}.pt"
    trainer.agent.save(final_model_path)

    if args.save_final_graph:
        graph_path = resolve_project_path(args.save_final_graph)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graph": trainer.env.graph,
                "algorithm": config.algorithm,
                "trainer_config": asdict(config),
                "history": history,
            },
            graph_path,
        )

    if args.history_json:
        history_path = resolve_project_path(args.history_json)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training complete from {PROJECT_ROOT}")
    print(f"Final model saved to: {final_model_path}")
    print(f"History records: {len(history)}")
    if trainer.best_eval_reward > -float("inf"):
        print(f"Best eval reward: {trainer.best_eval_reward:.4f}")


if __name__ == "__main__":
    main()
