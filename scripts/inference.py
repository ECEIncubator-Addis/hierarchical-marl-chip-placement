from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from _common import resolve_project_path, load_agent
from src.rl.evaluator import evaluate_policy

try:
    from configs.rl_envs import MacroPlacementEnv
except ImportError:  # pragma: no cover
    from src.models.macro_placement_env import MacroPlacementEnv


ALGORITHM_CHOICES = ["ppo", "a2c", "dqn"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained hierarchical RL placement agent on a new placement graph."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained RL checkpoint file.")
    parser.add_argument(
        "--graph-path",
        required=True,
        help="Path to the target graph file to optimize with the trained model.",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default=None,
        help="Algorithm type for the checkpoint if it cannot be inferred from the filename.",
    )
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of inference episodes to run.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic actions during inference when supported.",
    )
    parser.add_argument(
        "--output-graph",
        default=None,
        help="Optional path to save the final optimized placement graph as a .pt file.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to save the inference metrics as JSON.",
    )
    return parser


def _infer_algorithm_from_path(checkpoint_path: Path) -> str:
    name = checkpoint_path.name.lower()
    for choice in ALGORITHM_CHOICES:
        if choice in name:
            return choice
    raise ValueError(
        "Could not infer algorithm from checkpoint filename. Please provide --algorithm explicitly."
    )


def _load_env(graph_path: str, device: str | None = None) -> Any:
    target_path = resolve_project_path(graph_path)
    return MacroPlacementEnv(str(target_path))


def _validate_agent_env(agent: Any, env: Any) -> None:
    env_obs_dim = getattr(getattr(env, 'observation_space', None), 'shape', (None,))[0]
    env_action_dim = getattr(getattr(env, 'action_space', None), 'n', None)

    model = getattr(agent, 'model', agent)
    agent_obs_dim = getattr(model, 'obs_dim', None) or getattr(agent, 'obs_dim', None)
    agent_action_dim = getattr(agent, 'action_dim', None)

    if env_obs_dim is not None and agent_obs_dim is not None and env_obs_dim != agent_obs_dim:
        raise ValueError(
            f"Checkpoint is incompatible with this environment: agent expects observation dim {agent_obs_dim}, "
            f"but environment provides observation dim {env_obs_dim}. "
            "This usually means the checkpoint was trained on a different graph or macro count."
        )

    if env_action_dim is not None and agent_action_dim is not None and env_action_dim != agent_action_dim:
        raise ValueError(
            f"Checkpoint action space mismatch: agent expects {agent_action_dim} actions, "
            f"but environment has {env_action_dim} actions. "
            "Use a checkpoint trained on the same placement graph or a matching action space."
        )

    if hasattr(agent, 'num_macros') and hasattr(env, 'num_macros') and agent.num_macros != env.num_macros:
        raise ValueError(
            f"Checkpoint node count mismatch: agent trained for {agent.num_macros} macros, "
            f"but environment graph has {env.num_macros} macros. "
            "Graph-structured models are not compatible across different node counts."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    algorithm = args.algorithm or _infer_algorithm_from_path(Path(checkpoint_path))
    print(f"Loading agent from {checkpoint_path} using algorithm '{algorithm}'")

    agent = load_agent(checkpoint_path, algorithm, device=args.device)
    env = _load_env(args.graph_path, args.device)
    _validate_agent_env(agent, env)

    print(f"Running inference for {args.episodes} episode(s) on graph {args.graph_path}")
    result = evaluate_policy(agent, env, episodes=args.episodes, deterministic=args.deterministic)
    summary = {
        "checkpoint": str(checkpoint_path),
        "algorithm": algorithm,
        "graph_path": str(resolve_project_path(args.graph_path)),
        "device": args.device,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "metrics": result.to_dict(),
    }

    print(json.dumps(summary, indent=2))

    if args.output_graph:
        output_path = resolve_project_path(args.output_graph)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graph": env.graph,
                "metrics": result.to_dict(),
                "algorithm": algorithm,
                "checkpoint": str(checkpoint_path),
            },
            output_path,
        )
        print(f"Saved optimized graph to: {output_path}")

    if args.metrics_json:
        metrics_path = resolve_project_path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved inference metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
