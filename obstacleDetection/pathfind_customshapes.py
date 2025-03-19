import time
import numpy as np
from collections import deque
from munkres import Munkres
from grid_module import display_grid  # or adapt to your environment

def manhattan(a, b):
    """
    Returns the Manhattan distance between two points (r1,c1) and (r2,c2).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_dynamic(start, goal, grid, other_agents):
    """
    BFS that treats -1 and other_agents as obstacles.
    Returns path from just AFTER 'start' to 'goal' or [] if no path found.

    UPDATED for 8-direction movement:
    Agents can now move diagonally, not just up/down/left/right.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}

    # 8 directions (N, S, E, W + 4 diagonals)
    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (-1, -1), # diag up-left
        (-1, 1),  # diag up-right
        (1, -1),  # diag down-left
        (1, 1)    # diag down-right
    ]

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Avoid obstacles (-1) and other agents
                if grid[nr, nc] != -1 and (nr, nc) not in other_agents:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
    return []

def move_agents_layered_fill(
    grid: np.ndarray,
    agent_positions: list[tuple[int,int]],
    target_positions: list[tuple[int,int]],
    grid_placeholder,
    speed: float=0.3
):
    """
    1) Finds the center of the shape (average of target_positions).
    2) Sorts target_positions by distance to center (inner -> outer).
    3) Uses Hungarian Algorithm to assign each agent to exactly one target cell.
    4) Step-by-step BFS movement (now 8-direction).
       Direct swaps are allowed if it unblocks a path.
    5) This inside-out approach helps reduce leaving a hole unfilled.

    :param grid:
        2D numpy array (environment): -1 => obstacle, 0 => free, 1 => agent
    :param agent_positions:
        list of (r,c) starting positions
    :param target_positions:
        list of (r,c) user-checked cells forming the shape
    :param grid_placeholder:
        placeholder or display function for visualization
    :param speed:
        seconds to sleep between animation steps
    :return:
        (final_grid, final_positions)
    """

    # --- 1) Basic checks
    if not target_positions:
        print("No target positions selected.")
        return grid, []

    if len(agent_positions) < len(target_positions):
        print("Warning: Not enough agents to fill all target cells!")
    elif len(agent_positions) > len(target_positions):
        print("Warning: More agents than target cells; some agents won't have a unique target.")

    # --- 2) Compute center of shape
    rows = [r for (r,c) in target_positions]
    cols = [c for (r,c) in target_positions]
    center_r = sum(rows)//len(rows)
    center_c = sum(cols)//len(cols)
    center_pos = (center_r, center_c)

    # --- 3) Sort target_positions from innermost to outermost
    # We measure "distance to center" by Manhattan; closer => fill first
    sorted_targets = sorted(target_positions, key=lambda t: manhattan(t, center_pos))

    # --- 4) Build cost matrix for Hungarian: agent -> sorted_targets
    cost_matrix = []
    for agent_pos in agent_positions:
        row_costs = []
        for tgt in sorted_targets:
            row_costs.append(manhattan(agent_pos, tgt))
        cost_matrix.append(row_costs)

    # Solve assignment
    m = Munkres()
    indexes = m.compute(cost_matrix)  # (agent_idx, target_idx) pairs
    # agent_targets[i] => the actual cell assigned to agent i
    agent_targets = [None]*len(agent_positions)
    for (agent_i, target_i) in indexes:
        agent_targets[agent_i] = sorted_targets[target_i]

    # --- 5) Create agent states
    agents = []
    for i in range(len(agent_positions)):
        agents.append({
            'id': i,
            'pos': agent_positions[i],
            'target': agent_targets[i]
        })

    # --- 6) Iterative BFS movement (with diagonals)
    while True:
        new_grid = np.zeros_like(grid)
        all_reached = True

        # For BFS collisions
        agent_set = set(a['pos'] for a in agents)
        move_dict = {}
        conflict_positions = set()

        # BFS each agent one step
        for ag in agents:
            cur = ag['pos']
            tgt = ag['target']
            if cur == tgt:
                move_dict[cur] = cur
                continue
            else:
                all_reached = False

            # BFS ignoring agent's own cell
            other_agents = agent_set - {cur}
            path = bfs_dynamic(cur, tgt, grid, other_agents)
            if path:
                next_cell = path[0]
                # If not claimed by another agent => move
                if next_cell not in conflict_positions:
                    move_dict[cur] = next_cell
                    conflict_positions.add(next_cell)
                else:
                    # conflict => stay put
                    move_dict[cur] = cur
            else:
                # No path => stuck => stay
                move_dict[cur] = cur

        # Resolve direct swaps by letting them actually swap if beneficial
        final_moves = {}
        checked = set()
        for old_p, new_p in move_dict.items():
            if old_p in checked:
                continue

            # If two agents want to swap:
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                # Let them swap
                final_moves[old_p] = new_p
                final_moves[new_p] = old_p
                checked.add(new_p)
            else:
                final_moves[old_p] = new_p
            checked.add(old_p)

        # Update agent positions & build new_grid
        updated_agents = []
        for ag in agents:
            old_pos = ag['pos']
            new_pos = final_moves[old_pos]
            updated_agents.append({
                'id': ag['id'],
                'pos': new_pos,
                'target': ag['target']
            })
            new_grid[new_pos[0], new_pos[1]] = 1

        agents = updated_agents
        grid = new_grid

        # Display
        display_grid(grid, grid_placeholder)
        time.sleep(speed)

        if all_reached:
            break

    # Done
    final_positions = [a['pos'] for a in agents]
    return grid, final_positions