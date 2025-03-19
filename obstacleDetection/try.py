import streamlit as st
import numpy as np
from collections import deque
import time

# Define grid size (10x10)
ROWS, COLS = 10, 10

def initialize_grid():
    """Initialize a 10x10 grid with agents placed at the bottom rows."""
    if "grid" not in st.session_state:
        st.session_state.grid = np.zeros((ROWS, COLS), dtype=int)

        # Place exactly 16 agents in rows 8 and 9 (columns 0-7)
        agent_positions = [(8, j) for j in range(8)] + [(9, j) for j in range(8)]
        st.session_state.agent_positions = agent_positions
        for x, y in agent_positions:
            st.session_state.grid[x, y] = 1



# Define target positions to form a **square** at the top-left
SQUARE_POSITIONS = [
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 7), (2, 7), (3, 7), (4, 7),
    (4, 6), (4, 5), (4, 4), (4, 3),
    (3, 3), (2, 3), (1, 3)
]

# Define target positions to form a **circle-like shape**
CIRCLE_POSITIONS = [
    (1, 4), (1, 5), (2, 3), (2, 6),
    (3, 2), (3, 7), (4, 1), (4, 8),
    (5, 1), (5, 8), (6, 2), (6, 7),
    (7, 3), (7, 6), (8, 4), (8, 5)
]

def bfs(start, goal, grid):
    """Performs BFS to find the shortest path from start to goal."""
    queue = deque([(start, [])])  # (position, path)
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path  # Return the movement path

        # Define Von Neumann movement (Up, Down, Left, Right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if within bounds and the cell is empty
            if 0 <= nx < ROWS and 0 <= ny < COLS and grid[nx, ny] == 0 and (nx, ny) not in visited:
                queue.append(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))

    return []  # Return empty if no path is found

def move_agents(target_positions):
    """Moves all agents step by step simultaneously while avoiding collisions."""

    # Compute BFS paths for all agents
    agents = st.session_state.agent_positions.copy()
    paths = {agent: bfs(agent, target, st.session_state.grid) for agent, target in zip(agents, target_positions)}

    # Ensure movement continues until all agents reach their destinations
    while any(paths.values()):
        new_grid = np.copy(st.session_state.grid)
        move_attempts = {}  # Store intended moves

        # Step 1: Determine where each agent wants to move
        for agent, path in paths.items():
            if path:  # If a path exists
                next_pos = path[0]  # Next position the agent wants to move to
                move_attempts[agent] = next_pos  # Store intended move

        # Step 2: Resolve Conflicts (Prevent Two Agents Moving to the Same Cell)
        occupied_positions = set(st.session_state.agent_positions)  # Current positions
        final_moves = {}

        for agent, next_pos in move_attempts.items():
            if next_pos not in occupied_positions and list(move_attempts.values()).count(next_pos) == 1:
                final_moves[agent] = next_pos  # Safe to move
            else:
                final_moves[agent] = agent  # Stay in place if conflict

        # Step 3: Move Agents and Update Grid
        updated_paths = {}
        for agent, new_pos in final_moves.items():
            x, y = agent
            nx, ny = new_pos

            # Move agent if it's not staying in place
            if new_pos != agent:
                new_grid[x, y] = 0  # Clear old position
                new_grid[nx, ny] = 1  # Move agent to new position

            updated_paths[new_pos] = paths[agent][1:] if paths[agent] else []  # Remove used step

        # Update session state
        st.session_state.grid = new_grid
        st.session_state.agent_positions = list(updated_paths.keys())
        paths = updated_paths  # Update paths for next step

        # Update UI in Streamlit
        display_grid()
        time.sleep(0.3)  # Smooth animation delay

    display_grid()  # Final update to ensure the shape is fully formed

    

# -------------------------- Streamlit App --------------------------
initialize_grid()

st.title("Programmable Matter - Agents Forming Shapes")

# Display the grid
grid_placeholder = st.empty()  # Create a placeholder for updating grid

def display_grid():
    """Renders the grid using Streamlit UI."""
    with grid_placeholder.container():
        for i in range(ROWS):
            cols = st.columns(COLS)
            for j in range(COLS):
                cell = "🟦" if st.session_state.grid[i][j] == 1 else "⬜"
                cols[j].write(cell)


display_grid()  # Ensure grid updates dynamically


if st.button("Form Square"):
    move_agents(SQUARE_POSITIONS)  # Move agents immediately

if st.button("Form Circle"):
    move_agents(CIRCLE_POSITIONS)  # Move agents immediately
