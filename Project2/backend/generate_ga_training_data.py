import numpy as np
import random
import json
from app import run_genetic_algorithm


def random_grid(grid_size, num_agents, num_targets, num_obstacles):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    # Place obstacles
    obstacles = set()
    while len(obstacles) < num_obstacles:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if grid[pos] == 0:
            grid[pos] = 2
            obstacles.add(pos)
    # Place targets
    targets = set()
    while len(targets) < num_targets:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if grid[pos] == 0:
            grid[pos] = 1
            targets.add(pos)
    # Place agents
    agents = set()
    while len(agents) < num_agents:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if grid[pos] == 0:
            agents.add(pos)
    return grid, list(agents), list(targets), list(obstacles)


def extract_assignment(grid, agents, targets, obstacles):
    # Run GA and extract the assignment (chromosome) from the best solution
    # The best assignment is printed as "GA: Best assignment found: [..]" in run_genetic_algorithm
    # We'll slightly modify run_genetic_algorithm to return the best chromosome as well as steps
    result = run_genetic_algorithm(grid, agents, targets, obstacles)
    if isinstance(result, tuple):
        steps, best_chromosome = result
    else:
        steps = result
        # Try to infer assignment from the initial agent/target positions
        # This fallback assumes agent i is assigned to target i (not always true)
        best_chromosome = list(range(min(len(agents), len(targets))))
    # The chromosome is a list of target indices for each agent
    return best_chromosome, steps


data = []
for i in range(100):  # Generate 200 scenarios
    grid_size = 10
    num_agents = 9  # 3x3 square = 9 agents
    num_targets = 9  # 3x3 square = 9 targets
    num_obstacles = 8  # You can adjust this as needed

    grid, agents, targets, obstacles = random_grid(
        grid_size, num_agents, num_targets, num_obstacles
    )
    print(f"Scenario {i+1}:")
    assignment, steps = extract_assignment(grid, agents, targets, obstacles)
    data.append(
        {
            "grid": grid.tolist(),
            "agents": agents,
            "targets": targets,
            "obstacles": obstacles,
            "assignment": assignment,
        }
    )

with open("ga_training_data.json", "w") as f:
    json.dump(data, f)
