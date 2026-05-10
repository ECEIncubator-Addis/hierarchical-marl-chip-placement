from __future__ import annotations

import argparse
import json

import torch

from _common import load_agent, make_env, resolve_project_path
from src.rl import TrainerConfig, evaluate_policy, rollout_placement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained hierarchical RL placement agent.")
    parser.add_argument("--checkpoint", default="checkpoints/best_ppo.pt", help="Agent checkpoint path.")
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "dqn"], default="ppo", help="Checkpoint algorithm.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--device", default=None, help="Evaluation device; defaults to trainer config.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-config", default="configs/rl.yaml", help="Path to RL YAML config.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to environment YAML config.")
    parser.add_argument("--training-config", default="configs/training.yaml", help="Path to training YAML config.")
    parser.add_argument("--graph-path", default=None, help="Override graph path from config.")
    parser.add_argument("--save-graph", default=None, help="Optional path to save a one-episode rollout graph.")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save metrics as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainerConfig.from_yaml(args.rl_config, args.env_config, args.training_config)
    if args.graph_path:
        config.graph_path = str(resolve_project_path(args.graph_path))
    if args.device:
        config.device = args.device

    agent = load_agent(args.checkpoint, args.algorithm, device=config.device)
    env = make_env(config, eval_env=True)
    result = evaluate_policy(agent, env, episodes=args.episodes, deterministic=args.deterministic)
    metrics = result.to_dict()

    print(json.dumps(metrics, indent=2))

    if args.metrics_json:
        metrics_path = resolve_project_path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_graph:
        graph, rollout_result = rollout_placement(agent, env, deterministic=args.deterministic)
        graph_path = resolve_project_path(args.save_graph)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graph": graph, "metrics": rollout_result.to_dict(), "algorithm": args.algorithm}, graph_path)
        print(f"Rollout graph saved to: {graph_path}")


if __name__ == "__main__":
    main()
