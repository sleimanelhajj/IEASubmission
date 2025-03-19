import numpy as np

def initialize_grid(rows=20, cols=20, obstacles=None):
    """
    Creates and returns a grid of given dimensions with optional obstacle placements, 
    and then populates a number of agents at the bottom rows.

    rows: number of grid rows
    cols: number of grid columns
    obstacles: list of (x, y) tuples marking grid cells that contain obstacles

    Returns:
        grid: a 2D NumPy array representing the grid
        agent_positions: a list of (x, y) tuples where each agent is placed
        obstacles: the list of obstacles used (might be empty if none provided)
    """

    # --------------------------------------------------
    # Create a 2D grid initialized to zeros (empty cells)
    # grid[i, j] =  0 -> empty cell
    # grid[i, j] = -1 -> obstacle
    # grid[i, j] =  1 -> agent
    # --------------------------------------------------
    grid = np.zeros((rows, cols), dtype=int)

    # If no obstacles list is provided, default to an empty list.
    if obstacles is None:
        obstacles = []

    # --------------------------------------------------
    # Place obstacles in the grid:
    # Each obstacle is stored as a coordinate (x, y)
    # Setting grid[x, y] = -1 marks that cell as an obstacle
    # --------------------------------------------------
    for x, y in obstacles:
        grid[x, y] = -1  # ⛔ Mark obstacles

    # --------------------------------------------------
    # Determine how many agents to place.
    # We cap it at 50 max, or (cols * 2), whichever is smaller.
    # --------------------------------------------------
    num_agents = min(50, cols * 2)  # Increased from 16 to 50 max agents

    # --------------------------------------------------
    # Create 3 'rows' of agents at the bottom of the grid.
    # We'll push them into a list called agent_positions.
    # --------------------------------------------------
    agent_positions = []
    for i in range(3):  # 3 rows of agents at the bottom
        for j in range(min(cols, num_agents // 3 + 1)):
            # rows - i - 1 means we start from the bottom row
            # j is the column index
            agent_positions.append((rows - i - 1, j))

    # --------------------------------------------------
    # Sort the agent positions to maintain a predictable order.
    # This can help keep movement ordering consistent across runs.
    # --------------------------------------------------
    agent_positions = sorted(agent_positions)

    # --------------------------------------------------
    # Mark those positions in the grid array as containing agents.
    # --------------------------------------------------
    for x, y in agent_positions:
        grid[x, y] = 1  # 🟦 Place agents (represented by 1)

    # --------------------------------------------------
    # Return the updated grid, list of agent positions, and the obstacles.
    # --------------------------------------------------
    return grid, agent_positions, obstacles

def display_grid(grid, placeholder):
    """
    Converts the numeric grid representation into a string of emoji characters
    and displays it using the 'placeholder' (e.g., st.empty in Streamlit).
    
    grid: The 2D NumPy array storing cell states (-1 for obstacle, 0 for empty, 1 for agent).
    placeholder: A Streamlit UI placeholder element to display the textual grid layout.
    """

    # Start with an empty string that we'll build row by row
    grid_str = ""

    # --------------------------------------------------
    # Iterate over each row in the grid
    # --------------------------------------------------
    for row in grid:
        # We'll build each row as a string, then append to grid_str
        row_str = ""
        for cell in row:
            if cell == -1:
                # If the cell is -1, we show '⛔' for obstacle
                row_str += "⛔"
            elif cell == 0:
                # If the cell is 0, we show '⬜' for an empty cell
                row_str += "⬜"
            else:
                # Otherwise, we assume it's an agent (value 1),
                # so we show '🟦'
                row_str += "🟦"
        # After finishing the row, append a newline
        grid_str += row_str + "\n"

    # --------------------------------------------------
    # Finally, display this text-based grid in the placeholder
    # --------------------------------------------------
    placeholder.text(grid_str)
