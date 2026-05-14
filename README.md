# Hierarchical MARL Chip Placement

This project trains and evaluates hierarchical reinforcement learning agents for macro placement.

The runnable entry points live in `scripts/`:

- `scripts/train.py` trains an agent using the `src.rl` trainer stack.
- `scripts/evaluate.py` evaluates a saved agent checkpoint.
- `scripts/placement_benchmark.py` computes HPWL and density metrics for placement graphs.
- `scripts/visualize_layout.py` renders a placement graph or a trained-agent rollout.
- `scripts/train_gnn.py` trains the GraphSAGE-based GNN position regressor.
- `scripts/benchmark_gnn.py` benchmarks a trained GNN checkpoint.

## Setup

Run commands from the repository root:

```bash
cd hierarchical-marl-chip-placement
```

Use the project virtual environment because the system Python may not have `torch` installed:

```bash
.venv/bin/python --version
```

If you use plotting commands on a machine where Matplotlib cannot write to its default cache directory, prefix visualization and benchmarking commands with:

```bash
MPLCONFIGDIR=/tmp
```

## Training

Train with the default YAML configs:

```bash
.venv/bin/python scripts/train.py
```

The defaults are read from:

- `configs/rl.yaml`
- `configs/env.yaml`
- `configs/training.yaml`

Common overrides:

```bash
.venv/bin/python scripts/train.py \
  --algorithm ppo \
  --timesteps 100000 \
  --checkpoint-dir checkpoints \
  --history-json results/metrics/train_history.json \
  --save-final-graph results/placements/final_placement.pt
```

Train on another graph:

```bash
.venv/bin/python scripts/train.py \
  --graph-path src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt
```

Supported algorithms:

```bash
--algorithm ppo
--algorithm a2c
--algorithm dqn
```

To train PPO or A2C with the GNN representation encoder, set the algorithm config in `configs/rl.yaml`:

```yaml
algorithms:
  ppo:
    use_gnn_encoder: true
    gnn_hidden_channels: 128
    gnn_embedding_dim: 256
    gnn_layers: 2
    gnn_dropout: 0.1
```

With `use_gnn_encoder: true`, macro selection is scored from per-node GNN embeddings, while direction selection and value prediction use the pooled graph embedding.

You can also enable it from the training CLI:

```bash
.venv/bin/python scripts/train.py \
  --algorithm ppo \
  --use-gnn-encoder \
  --timesteps 100000 \
  --checkpoint-dir checkpoints/ppo_gnn
```

Training writes:

- `checkpoints/final_<algorithm>.pt`
- `checkpoints/best_<algorithm>.pt` when evaluation improves during training
- optional final placement graph if `--save-final-graph` is provided
- optional JSON history if `--history-json` is provided

## Evaluation

Evaluate a saved PPO checkpoint:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/best_ppo.pt \
  --algorithm ppo \
  --episodes 5
```

Save evaluation metrics and a rollout graph:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/best_ppo.pt \
  --algorithm ppo \
  --episodes 5 \
  --metrics-json results/metrics/eval_metrics.json \
  --save-graph results/placements/eval_rollout.pt
```

Evaluate a checkpoint against a different graph:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/best_ppo.pt \
  --algorithm ppo \
  --graph-path src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt
```

The evaluation output includes mean reward, episode length, best reward, best HPWL, and final HPWL mean.

## Benchmarking

Benchmark one graph:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/placement_benchmark.py \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt
```

Benchmark an initial graph and a trained graph:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/placement_benchmark.py \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  results/placements/eval_rollout.pt
```

Save metrics and a density heatmap:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/placement_benchmark.py \
  results/placements/eval_rollout.pt \
  --json results/metrics/benchmark.json \
  --heatmap results/plots/density_heatmap.png
```

Benchmark metrics include:

- `hpwl`
- `max_density`
- `avg_density`
- `total_macro_area`

## GNN Training

Train the GNN placement regressor on one graph:

```bash
.venv/bin/python scripts/train_gnn.py \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  --epochs 200 \
  --checkpoint checkpoints/gnn_position_regressor.pt \
  --metrics-json results/metrics/gnn_train_metrics.json
```

Train on all preprocessed graph files:

```bash
.venv/bin/python scripts/train_gnn.py \
  --graph-glob "src/data/preprocessed/**/*_graph.pt" \
  --epochs 200 \
  --checkpoint checkpoints/gnn_position_regressor.pt
```

The script reads defaults from `configs/gnn.yaml`, but it infers the actual input feature dimension from the graph data. Useful overrides:

```bash
.venv/bin/python scripts/train_gnn.py \
  --graph-glob "src/data/preprocessed/real-connection/**/*_graph.pt" \
  --epochs 300 \
  --hidden-channels 128 \
  --out-channels 256 \
  --learning-rate 0.0003 \
  --dropout 0.1 \
  --checkpoint checkpoints/gnn_real_connection.pt
```

By default, the model predicts normalized macro positions from `graph.x` and `edge_index`. You can append current positions to the input features with `--include-position-features`, but that makes the task much easier and is usually better for encoder experimentation than for placement prediction.

## GNN Benchmarking

Benchmark a trained GNN checkpoint on one graph:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/benchmark_gnn.py \
  --checkpoint checkpoints/gnn_position_regressor.pt \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt
```

Benchmark across all preprocessed graphs and save JSON:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/benchmark_gnn.py \
  --checkpoint checkpoints/gnn_position_regressor.pt \
  --graph-glob "src/data/preprocessed/**/*_graph.pt" \
  --json results/metrics/gnn_benchmark.json
```

Save the first predicted placement graph for visualization or regular placement benchmarking:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/benchmark_gnn.py \
  --checkpoint checkpoints/gnn_position_regressor.pt \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  --save-predicted-graph results/placements/gnn_predicted.pt
```

GNN benchmark metrics include:

- `position_mse`
- `position_mae`
- `target_hpwl`
- `predicted_hpwl`
- `hpwl_gap`
- `hpwl_gap_pct`
- `target_max_density`
- `predicted_max_density`

## Visualization

Visualize a graph file:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/visualize_layout.py \
  --graph-path src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  --title "Initial MemPool Tile Placement" \
  --save results/plots/initial_layout.png
```

Visualize a graph produced by evaluation:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/visualize_layout.py \
  --graph-path results/placements/eval_rollout.pt \
  --title "Evaluated Placement" \
  --save results/plots/eval_layout.png
```

Visualize a rollout directly from a trained checkpoint:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/visualize_layout.py \
  --checkpoint checkpoints/best_ppo.pt \
  --algorithm ppo \
  --title "PPO Rollout Placement" \
  --save results/plots/ppo_rollout_layout.png
```

## Quick Smoke Test

Run a tiny training pass into `/tmp` to confirm the scripts are wired correctly:

```bash
.venv/bin/python scripts/train.py \
  --timesteps 1 \
  --checkpoint-dir /tmp/hmarl-check \
  --history-json /tmp/hmarl-history.json \
  --save-final-graph /tmp/hmarl-final-graph.pt
```

Then evaluate and visualize the smoke checkpoint:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint /tmp/hmarl-check/final_ppo.pt \
  --algorithm ppo \
  --episodes 1 \
  --save-graph /tmp/hmarl-rollout.pt

MPLCONFIGDIR=/tmp .venv/bin/python scripts/visualize_layout.py \
  --graph-path /tmp/hmarl-rollout.pt \
  --save /tmp/hmarl-layout.png
```

Run a tiny GNN smoke test:

```bash
.venv/bin/python scripts/train_gnn.py \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  --epochs 1 \
  --hidden-channels 16 \
  --out-channels 16 \
  --checkpoint /tmp/hmarl-gnn.pt

MPLCONFIGDIR=/tmp .venv/bin/python scripts/benchmark_gnn.py \
  --checkpoint /tmp/hmarl-gnn.pt \
  src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt \
  --save-predicted-graph /tmp/hmarl-gnn-pred.pt
```

## Useful Help Commands

Each script exposes its own CLI help:

```bash
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
MPLCONFIGDIR=/tmp .venv/bin/python scripts/placement_benchmark.py --help
MPLCONFIGDIR=/tmp .venv/bin/python scripts/visualize_layout.py --help
.venv/bin/python scripts/train_gnn.py --help
MPLCONFIGDIR=/tmp .venv/bin/python scripts/benchmark_gnn.py --help
```
