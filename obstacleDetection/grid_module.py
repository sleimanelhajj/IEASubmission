import numpy as np

def initialize_grid(rows, cols, obstacles=None):
    """Initialize a grid with agents and obstacles in a structured manner."""
    grid = np.zeros((rows, cols), dtype=int)

    if obstacles is None:
        obstacles = []

    for x, y in obstacles:
        grid[x, y] = -1  # ⛔ Mark obstacles

    # ✅ Place agents in a structured manner (bottom rows)
    num_agents = min(16, cols * 2)
    agent_positions = [(rows - 2, j) for j in range(num_agents // 2)] + \
                      [(rows - 1, j) for j in range(num_agents // 2)]
    
    # ✅ Sort agents to ensure they move predictably
    agent_positions = sorted(agent_positions)

    for x, y in agent_positions:
        grid[x, y] = 1  # 🟦 Place agents

    return grid, agent_positions, obstacles

def display_grid(grid, placeholder):
    """Render the grid as text with obstacles."""
    grid_str = ""
    for row in grid:
        row_str = "".join("🟦" if cell == 1 else "⛔" if cell == -1 else "⬜" for cell in row)
        grid_str += row_str + "\n"
    placeholder.text(grid_str)