
from stable_baselines3 import PPO
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.macro_placement_env import MacroPlacementEnv

env = MacroPlacementEnv("src/data/preprocessed/real-connection/MemPool_tile/Nangate45/MemPool_tile_Nangate45_graph.pt")
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)
model.learn(total_timesteps=100000)

# Save best placement
obs, _ = env.reset()
for _ in range(env.max_steps):
    action, _ = model.predict(obs)
    obs, _, term, _, _ = env.step(action)
    if term:
        break

torch.save(env.graph, "trained_placement_MemPool_tile.pt")
print("Trained graph saved!")