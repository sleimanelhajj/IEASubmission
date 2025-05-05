import numpy as np
import os
import pickle
from app import move_agents_deep_qlearning, DQN

# Example grid (15x15 empty)
grid_size = 4
grid = np.zeros((grid_size, grid_size), dtype=int)

# Example: add obstacles if needed
obstacle_positions = []  # e.g., [(5,5), (6,6)]

# All empty cells
empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in obstacle_positions]

# Directory to save models
model_dir = "trained_models"
os.makedirs(model_dir, exist_ok=True)

# Train for all pairs (or a subset for speed)
for start_pos in empty_cells:
    for target_pos in empty_cells:
        if start_pos == target_pos:
            continue
        model_path = os.path.join(model_dir, f"dqn_{start_pos[0]}_{start_pos[1]}_{target_pos[0]}_{target_pos[1]}.pkl")
        if os.path.exists(model_path):
            print(f"Model already exists for {start_pos} -> {target_pos}, skipping.")
            continue
        print(f"Training DQN for {start_pos} -> {target_pos} ...")
        agent_positions = [start_pos]
        target_positions = [target_pos]
        result = move_agents_deep_qlearning(
            grid, agent_positions, target_positions, obstacle_positions, episodes=200
        )
        with open(model_path, "wb") as f:
            pickle.dump(result["model_state"], f)
        print(f"Saved model to {model_path}")