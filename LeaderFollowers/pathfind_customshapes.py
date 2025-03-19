import time
from grid_module import display_grid
import numpy as np
from collections import deque

def bfs_dynamic(start, goal, grid, other_agents):
    """
    BFS that treats other agent positions (and any -1 cells) as obstacles.
    other_agents: set of (r, c) for all agents except 'start' cell.
    Returns a path (list of cells) from just after 'start' to 'goal', or [] if none.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path  # the path includes 'goal' but excludes the original 'start'

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Blocked if cell is -1 (true obstacle) or an agent in other_agents
                if (grid[nr, nc] != -1) and ((nr, nc) not in other_agents) and ((nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return []

def move_agents_dynamic_bfs(grid, agent_positions, target_positions, grid_placeholder, speed=0.3):
    """
    Moves agents to target_positions using a step-by-step BFS that treats 
    other agents (and -1 cells) as obstacles. Recomputes BFS each timestep
    for collision-free shape formation.
    """
    # Assign permanent IDs to each agent (starting from 1)
    agent_ids = list(range(1, len(agent_positions) + 1))

    # Build agent states with IDs
    agents = []
    for i in range(len(agent_positions)):
        agents.append({
            'pos': agent_positions[i], 
            'target': target_positions[i],
            'id': agent_ids[i]  # Add ID for visualization
        })
        
    # Display initial state with numbered agents
    initial_grid = np.zeros_like(grid)
    for agent in agents:
        r, c = agent['pos']
        initial_grid[r, c] = agent['id']
    display_grid(initial_grid, grid_placeholder)
    time.sleep(1)  # Give time to see initial state

    while True:
        # Step 1: Build a fresh grid for the next movement step
        new_grid = np.zeros_like(grid)
        move_dict = {}
        conflict_positions = set()
        all_reached = True

        # Step 2: Sort agents by distance to target (optional, helps reduce deadlocks)
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        agents.sort(key=lambda ag: manhattan(ag['pos'], ag['target']))

        # Gather set of agent positions for BFS obstacles
        agent_set = set(a['pos'] for a in agents)

        # Step 3: For each agent, do a BFS with other agents as obstacles
        for ag in agents:
            current_pos = ag['pos']
            goal_pos = ag['target']

            # If already at target, stay put
            if current_pos == goal_pos:
                move_dict[current_pos] = current_pos
                continue
            else:
                all_reached = False

            # BFS ignoring this agent's current position so it can move out
            other_agents = agent_set - {current_pos}
            path = bfs_dynamic(current_pos, goal_pos, grid, other_agents)

            if path:
                # Attempt to move one step along BFS if not conflicting
                next_cell = path[0]
                if next_cell not in conflict_positions:
                    move_dict[current_pos] = next_cell
                    conflict_positions.add(next_cell)
                else:
                    # Conflict => stay
                    move_dict[current_pos] = current_pos
            else:
                # No path => stay
                move_dict[current_pos] = current_pos

        # Step 4: Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p:
                final_moves[old_p] = old_p
            else:
                final_moves[old_p] = new_p

        # Step 5: Update agent positions & new grid with NUMBERED agents
        updated_agents = []
        for ag in agents:
            old_pos = ag['pos']
            new_pos = final_moves[old_pos]
            updated_agents.append({
                'pos': new_pos, 
                'target': ag['target'],
                'id': ag['id']  # Preserve agent ID
            })
            new_grid[new_pos[0], new_pos[1]] = ag['id']  # Use agent ID in grid

        agents = updated_agents
        grid = new_grid

        # Step 6: Streamlit visualization
        display_grid(grid, grid_placeholder)
        time.sleep(speed)

        # If all agents are at their targets, end the loop
        if all_reached:
            break

    final_positions = [a['pos'] for a in agents]
    return grid, final_positions
