import numpy as np

def initialize_grid(rows, cols, obstacles=None, num_agents=None):
    """Initialize a grid with agents and obstacles in a structured manner."""
    grid = np.zeros((rows, cols), dtype=int)

    if obstacles is None:
        obstacles = []

    for x, y in obstacles:
        grid[x, y] = -1  # ⛔ Mark obstacles

    # ✅ Place agents in a structured manner (bottom rows)
    if num_agents is None:
        num_agents = min(20, cols * 2)  # Default if not specified
        
    # Ensure we don't try to place more agents than can fit in the bottom rows
    max_possible = cols * 2
    num_agents = min(num_agents, max_possible)
    
    agent_positions = [(rows - 2, j) for j in range(num_agents // 2)] + \
                      [(rows - 1, j) for j in range(num_agents - num_agents // 2)]
    
    # ✅ Sort agents to ensure they move predictably
    agent_positions = sorted(agent_positions)

    for x, y in agent_positions:
        grid[x, y] = 1  # 🟦 Place agents

    return grid, agent_positions, obstacles

def display_grid(grid, placeholder):
    """Render the grid as text with obstacles and agents."""
    grid_str = ""
    for row in grid:
        row_str = ""
        for cell in row:
            if cell == -1:
                row_str += "⛔"  # Obstacle
            elif cell == 0:
                row_str += "⬜"  # Empty
            elif cell > 0:
                row_str += "🟦"  # Agent (no numbers, just blue squares)
            else:
                row_str += "⬜"  # Default for safety
        grid_str += row_str + "\n"
    placeholder.text(grid_str)
