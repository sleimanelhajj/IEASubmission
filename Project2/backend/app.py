from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
from collections import deque
from scipy.optimize import linear_sum_assignment
import random

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Shape definitions
SHAPES = {
    "square": [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 3],
        [2, 0],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
    ],
    "triangle": [[0, 2], [1, 1], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
    "circle": [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 4],
        [2, 0],
        [2, 4],
        [3, 0],
        [3, 4],
        [4, 1],
        [4, 2],
        [4, 3],
    ],
}

# ======================================================
# Utility Functions
# ======================================================


def manhattan_dist(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_dynamic(start, goal, grid, agents, obstacles=None):
    """
    BFS that treats other agents and obstacles as impassable cells.
    agents is a set of all current agent positions (except this agent's own position).
    obstacles is a set of impassable cells on the grid.
    Now considers 8 directions (including diagonals).
    Returns a list of steps from start->goal (excluding start, including goal).
    If no path, returns [].
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}

    # 8-direction movement: up, down, left, right, plus 4 diagonals
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # If obstacles is None, use an empty set
    if obstacles is None:
        obstacles = set()

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Treat agent cells and obstacles as impassable
                if (
                    (nr, nc) not in agents
                    and (nr, nc) not in visited
                    and (nr, nc) not in obstacles
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return []


def hungarian_assignment(agent_positions, target_positions):
    """
    Assign each agent to a unique target using the Hungarian Algorithm
    (via scipy's linear_sum_assignment).
    Returns list of (agent, target) pairs.
    """
    agent_positions = sorted(agent_positions)
    target_positions = sorted(target_positions)
    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    size = max(n_agents, n_targets)

    cost_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i < n_agents and j < n_targets:
                cost_matrix[i, j] = manhattan_dist(
                    agent_positions[i], target_positions[j]
                )
            else:
                # If there's a mismatch in counts, put a large cost so it's never chosen
                cost_matrix[i, j] = 999999

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for i in range(size):
        if row_ind[i] < n_agents and col_ind[i] < n_targets:
            agent = agent_positions[row_ind[i]]
            target = target_positions[col_ind[i]]
            assignments.append((agent, target))
    return assignments

def hungarian_furthest_assignment(agent_positions, target_positions):
    """
    Assign each agent to the furthest available target from its own position using the Hungarian algorithm.
    Returns list of (agent, target) pairs.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    size = max(n_agents, n_targets)
    cost_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i < n_agents and j < n_targets:
                # Negative distance for maximization
                cost_matrix[i, j] = -manhattan_dist(agent_positions[i], target_positions[j])
            else:
                cost_matrix[i, j] = 999999
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for i in range(size):
        if row_ind[i] < n_agents and col_ind[i] < n_targets:
            assignments.append((agent_positions[row_ind[i]], target_positions[col_ind[i]]))
    return assignments
    """
    Assign each agent to the furthest available target from its current position.
    Returns list of (agent, target) pairs.
    """
    assignments = []
    available_targets = set(target_positions)
    for agent in agent_positions:
        if not available_targets:
            break
        # Find the furthest target for this agent
        furthest_target = max(available_targets, key=lambda t: manhattan_dist(agent, t))
        assignments.append((agent, furthest_target))
        available_targets.remove(furthest_target)
    return assignments

def get_corners(target_positions):
    """Return a set of corner positions from the target positions."""
    rows = [r for r, _ in target_positions]
    cols = [c for _, c in target_positions]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    corners = set([
        (min_r, min_c),
        (min_r, max_c),
        (max_r, min_c),
        (max_r, max_c),
    ])
    # Only keep corners that are actually targets
    return corners & set(target_positions)
def hungarian_furthest_assignment_no_corner_priority(agent_positions, target_positions):
    """
    Assign each agent to the furthest available target from its own position using the Hungarian algorithm,
    but assign corners last (lowest priority).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    corners = get_corners(target_positions)
    # Sort targets: non-corners first, then corners
    sorted_targets = [t for t in target_positions if t not in corners] + [t for t in target_positions if t in corners]

    n_agents = len(agent_positions)
    n_targets = len(sorted_targets)
    size = max(n_agents, n_targets)
    cost_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i < n_agents and j < n_targets:
                # Negative distance for maximization
                cost_matrix[i, j] = -manhattan_dist(agent_positions[i], sorted_targets[j])
            else:
                cost_matrix[i, j] = 999999
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for i in range(size):
        if row_ind[i] < n_agents and col_ind[i] < n_targets:
            assignments.append((agent_positions[row_ind[i]], sorted_targets[col_ind[i]]))
    return assignments
# ======================================================
# 1. Inside-Out Algorithm (from your original code)
# ======================================================


def move_agents_inside_out(grid, agent_positions, target_positions, obstacle_positions):
    """
    Assign agents to the furthest available targets from the hole (empty space) inside the target area.
    Returns step-by-step data for the React frontend.
    """
    import numpy as np

    # Initialize metrics
    metrics = {
        "agent_steps": [0] * len(agent_positions),
        "waiting_time": [0] * len(agent_positions),
        "manhattan_distances": [0] * len(agent_positions),
        "path_length": [0] * len(agent_positions),
        "reached_target": [False] * len(agent_positions),
        "is_inner": [False] * len(agent_positions),
        "assigned_target": [None] * len(agent_positions),
        "inner_targets_count": 0,
        "outer_targets_count": 0,
        "obstacles_count": len(obstacle_positions),
        "conflicts": 0,
    }

    # -------- Find the "hole" (empty cell inside the target area) --------
    hole_position = None
    target_set = set(target_positions)
    for r, c in target_positions:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in target_set and (nr, nc) not in obstacle_positions:
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0:
                        hole_position = (nr, nc)
                        break
        if hole_position:
            break
    # If not found, just use the center of the targets
    if not hole_position:
        center_r = sum(r for r, _ in target_positions) // len(target_positions)
        center_c = sum(c for _, c in target_positions) // len(target_positions)
        hole_position = (center_r, center_c)

    # -------- Assign agents to targets maximizing distance from the hole --------
    all_assignments = hungarian_furthest_assignment(agent_positions, target_positions)

    # -------- Store agent state with target info --------
    agents = []
    for idx, (agent, target) in enumerate(all_assignments):
        metrics["assigned_target"][idx] = target
        metrics["manhattan_distances"][idx] = manhattan_dist(agent, target)
        initial_path = bfs_dynamic(agent, target, grid, set(), set(obstacle_positions))
        if initial_path:
            metrics["path_length"][idx] = len(initial_path)
        else:
            metrics["path_length"][idx] = metrics["manhattan_distances"][idx]
        agents.append(
            {
                "id": idx,
                "pos": agent,
                "target": target,
                "is_inner": False,  # You can update this if you want to distinguish
            }
        )
    # Track any agents without targets (happens when more agents than targets)
    unassigned_agents = [
        a for a in agent_positions if a not in [ag["pos"] for ag in agents]
    ]
    for agent in unassigned_agents:
        agent_idx = agent_positions.index(agent)
        agents.append(
            {"id": agent_idx, "pos": agent, "target": None, "is_inner": False}
        )

    # -------- PHASE: MOVEMENT ALGORITHM WITH STEP TRACKING --------
    # For React frontend - collect all steps
    simulation_steps = []

    # Add initial state
    initial_agents = []
    for ag in agents:
        initial_agents.append(
            {
                "id": ag["id"] + 1,  # 1-indexed for display
                "x": ag["pos"][0],
                "y": ag["pos"][1],
            }
        )

    # Create path visualization matrix for the frontend
    path_visualization = {}
    for agent in agents:
        if agent["target"] is not None:
            current = agent["pos"]
            target = agent["target"]
            path = bfs_dynamic(current, target, grid, set(), set(obstacle_positions))
            for cell in path:
                r, c = cell
                if (r, c) != current and (r, c) != target:
                    if r not in path_visualization:
                        path_visualization[r] = {}
                    path_visualization[r][c] = True

    simulation_steps.append(
        {"step": 0, "agents": initial_agents, "paths": path_visualization}
    )

    # Main movement loop
    step_counter = 1
    max_steps = 100  # Limit steps for API response size

    while True and step_counter < max_steps:
        new_grid = np.zeros_like(grid)
        for r, c in obstacle_positions:
            new_grid[r, c] = -1

        move_dict = {}
        conflict_positions = set()
        all_reached = True

        agents_with_targets = [ag for ag in agents if ag["target"] is not None]
        agents_with_targets.sort(
            key=lambda ag: (
                -len(
                    bfs_dynamic(
                        ag["pos"],
                        ag["target"],
                        grid,
                        set(a["pos"] for a in agents if a != ag),
                        obstacle_positions,
                    )
                )
                if ag["target"] is not None
                else 0
            )
        )

        agent_positions_set = set(ag["pos"] for ag in agents)

        for ag in agents_with_targets:
            current = ag["pos"]
            target = ag["target"]
            agent_id = ag["id"]

            if current == target:
                move_dict[current] = current
                if not metrics["reached_target"][agent_id]:
                    metrics["reached_target"][agent_id] = True
                continue

            all_reached = False

            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(
                current, target, grid, agent_set_except_self, obstacle_positions
            )

            if path:
                next_step = path[0]
                if next_step not in conflict_positions:
                    move_dict[current] = next_step
                    conflict_positions.add(next_step)
                else:
                    move_dict[current] = current
                    metrics["waiting_time"][agent_id] += 1
                    metrics["conflicts"] += 1
            else:
                move_dict[current] = current
                metrics["waiting_time"][agent_id] += 1

        for ag in agents:
            if ag["target"] is None:
                move_dict[ag["pos"]] = ag["pos"]

        for ag in agents:
            if ag["pos"] not in move_dict:
                move_dict[ag["pos"]] = ag["pos"]

        # Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                final_moves[old_p] = old_p
                for ag in agents:
                    if ag["pos"] == old_p or ag["pos"] == new_p:
                        metrics["waiting_time"][ag["id"]] += 1
                metrics["conflicts"] += 2
            else:
                final_moves[old_p] = new_p

        updated_agents = []
        for ag in agents:
            old_pos = ag["pos"]
            agent_id = ag["id"]

            if old_pos in final_moves:
                new_pos = final_moves[old_pos]
                if new_pos in obstacle_positions:
                    new_pos = old_pos
                    metrics["waiting_time"][agent_id] += 1
            else:
                new_pos = old_pos
                metrics["waiting_time"][agent_id] += 1

            if old_pos != new_pos:
                metrics["agent_steps"][agent_id] += 1

            new_grid[new_pos[0], new_pos[1]] = 1

            updated_agents.append(
                {
                    "id": agent_id,
                    "pos": new_pos,
                    "target": ag["target"],
                    "is_inner": ag["is_inner"],
                }
            )

        agents = updated_agents
        grid = new_grid

        step_agents = []
        for ag in agents:
            step_agents.append(
                {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]}
            )

        simulation_steps.append(
            {"step": step_counter, "agents": step_agents, "paths": path_visualization}
        )

        step_counter += 1

        if all_reached:
            break

    return simulation_steps
    """
    Modified version of your move_agents_no_collision function to:
    1. Remove Streamlit dependencies
    2. Return step-by-step data for the React frontend
    3. Slightly simplified for API usage
    """
    # Initialize metrics
    metrics = {
        "agent_steps": [0] * len(agent_positions),
        "waiting_time": [0] * len(agent_positions),
        "manhattan_distances": [0] * len(agent_positions),
        "path_length": [0] * len(agent_positions),
        "reached_target": [False] * len(agent_positions),
        "is_inner": [False] * len(agent_positions),
        "assigned_target": [None] * len(agent_positions),
        "inner_targets_count": 0,
        "outer_targets_count": 0,
        "obstacles_count": 0,
        "conflicts": 0,
    }

    metrics["obstacles_count"] = len(obstacle_positions)

    # -------- Identify Inner and Outer Targets --------
    target_set = set(target_positions)

    inner_targets = []
    outer_targets = []

    # Directions for neighboring cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for target in target_positions:
        r, c = target
        is_boundary = False

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # If the neighbor is not in targets, this is a boundary target
            if (nr, nc) not in target_set:
                is_boundary = True
                break

        if is_boundary:
            outer_targets.append(target)
        else:
            inner_targets.append(target)

    # If no inner targets were found, use centrality as an alternative
    if not inner_targets:
        if target_positions:
            center_r = sum(r for r, _ in target_positions) / len(target_positions)
            center_c = sum(c for _, c in target_positions) / len(target_positions)

            # Sort by distance to center
            sorted_targets = sorted(
                target_positions,
                key=lambda pos: (pos[0] - center_r) ** 2 + (pos[1] - center_c) ** 2,
            )

            inner_count = max(1, len(target_positions) // 3)
            inner_targets = sorted_targets[:inner_count]
            outer_targets = sorted_targets[inner_count:]

    metrics["inner_targets_count"] = len(inner_targets)
    metrics["outer_targets_count"] = len(outer_targets)

    # Initialize grid
    new_grid = np.zeros_like(grid)
    for r, c in obstacle_positions:
        new_grid[r, c] = -1
    for a in agent_positions:
        new_grid[a[0], a[1]] = 1
    grid = new_grid

    # -------- PHASE 1: INNER TARGETS ASSIGNMENT --------
    # Phase 1: Assign agents to inner targets only
    remaining_agents = agent_positions.copy()
    inner_assignments = []

    # If we have inner targets, assign closest agents to them
    if inner_targets:
        # Create cost matrix for inner targets only
        inner_cost = []
        for agent in remaining_agents:
            agent_costs = []
            for target in inner_targets:
                dist = manhattan_dist(agent, target)
                agent_costs.append((dist, agent, target))
            agent_costs.sort()
            inner_cost.append(agent_costs)

        # Flatten and sort all agent-target pairs by distance
        all_costs = []
        for agent_costs in inner_cost:
            all_costs.extend(agent_costs)
        all_costs.sort()

        # Greedy assignment - take closest pairs first
        assigned_agents = set()
        assigned_targets = set()

        for dist, agent, target in all_costs:
            if agent not in assigned_agents and target not in assigned_targets:
                inner_assignments.append((agent, target))
                assigned_agents.add(agent)
                assigned_targets.add(target)

                # Find the agent index and update metrics
                agent_idx = agent_positions.index(agent)
                metrics["is_inner"][agent_idx] = True
                metrics["assigned_target"][agent_idx] = target
                metrics["manhattan_distances"][agent_idx] = dist

                # Calculate initial path length for efficiency metrics
                initial_path = bfs_dynamic(
                    agent, target, grid, set(), set(obstacle_positions)
                )
                if initial_path:
                    metrics["path_length"][agent_idx] = len(initial_path)
                else:
                    metrics["path_length"][
                        agent_idx
                    ] = dist  # Use Manhattan distance if no path found

                # Stop when all inner targets are assigned
                if len(assigned_targets) == len(inner_targets):
                    break

        # Remove assigned agents from remaining_agents
        remaining_agents = [a for a in remaining_agents if a not in assigned_agents]

    # -------- PHASE 2: OUTER TARGETS ASSIGNMENT --------
    # Phase 2: Assign remaining agents to outer targets
    outer_assignments = []

    if outer_targets and remaining_agents:
        # Create cost matrix for outer targets only
        outer_cost = []
        for agent in remaining_agents:
            agent_costs = []
            for target in outer_targets:
                dist = manhattan_dist(agent, target)
                agent_costs.append((dist, agent, target))
            agent_costs.sort()
            outer_cost.append(agent_costs)

        # Flatten and sort all agent-target pairs by distance
        all_costs = []
        for agent_costs in outer_cost:
            all_costs.extend(agent_costs)
        all_costs.sort()

        # Greedy assignment - take closest pairs first
        assigned_agents = set()
        assigned_targets = set()

        for dist, agent, target in all_costs:
            if agent not in assigned_agents and target not in assigned_targets:
                outer_assignments.append((agent, target))
                assigned_agents.add(agent)
                assigned_targets.add(target)

                # Find the agent index and update metrics
                agent_idx = agent_positions.index(agent)
                metrics["is_inner"][agent_idx] = False
                metrics["assigned_target"][agent_idx] = target
                metrics["manhattan_distances"][agent_idx] = dist

                # Calculate initial path length for efficiency metrics
                initial_path = bfs_dynamic(
                    agent, target, grid, set(), set(obstacle_positions)
                )
                if initial_path:
                    metrics["path_length"][agent_idx] = len(initial_path)
                else:
                    metrics["path_length"][
                        agent_idx
                    ] = dist  # Use Manhattan distance if no path found

                # Stop when all agents or targets are assigned
                if len(assigned_targets) == len(outer_targets) or len(
                    assigned_agents
                ) == len(remaining_agents):
                    break

    # Combine assignments (inner first, then outer)
    all_assignments = inner_assignments + outer_assignments

    # -------- Store agent state with target info --------
    agents = []
    for agent, target in all_assignments:
        is_inner = target in inner_targets
        agent_idx = agent_positions.index(agent)
        agents.append(
            {"id": agent_idx, "pos": agent, "target": target, "is_inner": is_inner}
        )

    # Track any agents without targets (happens when more agents than targets)
    unassigned_agents = [
        a for a in agent_positions if a not in [ag["pos"] for ag in agents]
    ]
    for agent in unassigned_agents:
        agent_idx = agent_positions.index(agent)
        agents.append(
            {"id": agent_idx, "pos": agent, "target": None, "is_inner": False}
        )

    # -------- PHASE 3: MOVEMENT ALGORITHM WITH STEP TRACKING --------
    # For React frontend - collect all steps
    simulation_steps = []

    # Add initial state
    initial_agents = []
    for ag in agents:
        initial_agents.append(
            {
                "id": ag["id"] + 1,  # 1-indexed for display
                "x": ag["pos"][0],
                "y": ag["pos"][1],
            }
        )

    # Create path visualization matrix for the frontend
    path_visualization = {}
    for agent in agents:
        if agent["target"] is not None:
            current = agent["pos"]
            target = agent["target"]
            path = bfs_dynamic(current, target, grid, set(), set(obstacle_positions))
            # Mark path cells for visualization
            for cell in path:
                r, c = cell
                if (r, c) != current and (r, c) != target:  # Skip start and end
                    if r not in path_visualization:
                        path_visualization[r] = {}
                    path_visualization[r][c] = True

    simulation_steps.append(
        {"step": 0, "agents": initial_agents, "paths": path_visualization}
    )

    # Main movement loop
    step_counter = 1
    max_steps = 100  # Limit steps for API response size

    while True and step_counter < max_steps:
        new_grid = np.zeros_like(grid)
        # Add obstacles back
        for r, c in obstacle_positions:
            new_grid[r, c] = -1

        move_dict = {}
        conflict_positions = set()
        all_reached = True

        # First sort by inner/outer, then by distance to target
        # This ensures inner targets get priority in movement
        agents_with_targets = [ag for ag in agents if ag["target"] is not None]
        agents_with_targets.sort(
            key=lambda ag: (not ag["is_inner"], manhattan_dist(ag["pos"], ag["target"]))
        )

        # Get positions of all agents for collision avoidance
        agent_positions_set = set(ag["pos"] for ag in agents)

        # First process agents with targets
        for ag in agents_with_targets:
            current = ag["pos"]
            target = ag["target"]
            agent_id = ag["id"]

            if current == target:
                move_dict[current] = current  # Already at target

                # If this is the first time reaching the target, update metrics
                if not metrics["reached_target"][agent_id]:
                    metrics["reached_target"][agent_id] = True
                continue

            all_reached = False  # At least one agent still moving

            # Calculate path avoiding other agents and obstacles
            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(
                current, target, grid, agent_set_except_self, obstacle_positions
            )

            if path:
                next_step = path[0]
                if next_step not in conflict_positions:
                    move_dict[current] = next_step
                    conflict_positions.add(next_step)
                else:
                    move_dict[current] = current  # Stay put due to conflict
                    metrics["waiting_time"][agent_id] += 1
                    metrics["conflicts"] += 1
            else:
                move_dict[current] = current  # No path, stay put
                metrics["waiting_time"][agent_id] += 1

        # Then process agents without targets (just stay in place)
        for ag in agents:
            if ag["target"] is None:
                move_dict[ag["pos"]] = ag["pos"]

        # Make sure ALL agents are accounted for
        for ag in agents:
            if ag["pos"] not in move_dict:
                move_dict[ag["pos"]] = ag["pos"]

        # Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                # No swapping - both agents stay put
                final_moves[old_p] = old_p

                # Find the agents involved in swap and increment their conflict counters
                for ag in agents:
                    if ag["pos"] == old_p or ag["pos"] == new_p:
                        metrics["waiting_time"][ag["id"]] += 1

                metrics["conflicts"] += 2  # Count as two conflicts
            else:
                final_moves[old_p] = new_p

        # Update agent positions
        updated_agents = []
        for ag in agents:
            old_pos = ag["pos"]
            agent_id = ag["id"]

            if old_pos in final_moves:
                new_pos = final_moves[old_pos]
                if new_pos in obstacle_positions:
                    new_pos = old_pos  # Safety check
                    metrics["waiting_time"][agent_id] += 1
            else:
                new_pos = old_pos  # Default to staying put if not in moves dict
                metrics["waiting_time"][agent_id] += 1

            # Count a step if the agent actually moved
            if old_pos != new_pos:
                metrics["agent_steps"][agent_id] += 1

            # Add agent to grid
            new_grid[new_pos[0], new_pos[1]] = 1

            # Update agent position
            updated_agents.append(
                {
                    "id": agent_id,
                    "pos": new_pos,
                    "target": ag["target"],
                    "is_inner": ag["is_inner"],
                }
            )

        agents = updated_agents
        grid = new_grid

        # Create step data for React frontend
        step_agents = []
        for ag in agents:
            step_agents.append(
                {
                    "id": ag["id"] + 1,  # 1-indexed for display
                    "x": ag["pos"][0],
                    "y": ag["pos"][1],
                }
            )

        simulation_steps.append(
            {"step": step_counter, "agents": step_agents, "paths": path_visualization}
        )

        step_counter += 1

        # Exit condition - all agents have reached their targets
        if all_reached:
            break

    return simulation_steps


# ======================================================
# 2. Leader-Follower Algorithm
# ======================================================


def move_agents_leader_follower(
    grid, agent_positions, target_positions, obstacle_positions
):
    """
    Leader-Follower movement approach adapted for the React frontend:
    - The first agent (leader) moves towards its target
    - Each follower moves step-by-step to the position previously occupied by the agent in front
    - After leader reaches target, other agents navigate independently to their targets
    """
    # Copy agent positions to avoid modifying the original list
    current_positions = agent_positions.copy()

    # Assign targets to agents (leader gets the first target)
    leader_pos = current_positions[0]

    # Sort targets by distance from the leader
    sorted_targets = sorted(
        target_positions, key=lambda t: manhattan_dist(leader_pos, t)
    )
    leader_target = sorted_targets[0]

    # Assign remaining targets using Hungarian algorithm
    if len(current_positions) > 1 and len(target_positions) > 1:
        follower_targets = hungarian_assignment(
            current_positions[1:], [t for t in sorted_targets if t != leader_target]
        )
        follower_assignments = {agent: target for agent, target in follower_targets}
    else:
        follower_assignments = {}

    # Find leader's path
    leader_path = bfs_dynamic(
        leader_pos, leader_target, grid, set(), obstacle_positions
    )
    if not leader_path:
        # If no path found, return at least the initial state
        initial_agents = []
        for i, pos in enumerate(current_positions):
            initial_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})
        return [{"step": 0, "agents": initial_agents, "paths": {}}]

    # Track position history for each agent
    position_history = [[] for _ in range(len(current_positions))]
    for i, pos in enumerate(current_positions):
        position_history[i].append(pos)

    # For React frontend - collect all steps
    simulation_steps = []

    # Add initial state
    initial_agents = []
    for i, pos in enumerate(current_positions):
        initial_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

    # Create path visualization for the React frontend
    path_visualization = {}

    # Highlight leader's path
    for cell in leader_path:
        r, c = cell
        if (r, c) != leader_pos and (r, c) != leader_target:  # Skip start and end
            if r not in path_visualization:
                path_visualization[r] = {}
            path_visualization[r][c] = True

    simulation_steps.append(
        {"step": 0, "agents": initial_agents, "paths": path_visualization}
    )

    # Insert initial position at beginning of path for smooth movement
    full_leader_path = [leader_pos] + leader_path

    # -------- PHASE 1: LEADER MOVEMENT --------
    # Leader movement phase - move leader step by step
    for step_idx in range(1, len(full_leader_path)):
        # Move leader to next position in path
        current_positions[0] = full_leader_path[step_idx]
        position_history[0].append(current_positions[0])

        # For each follower, calculate path to the previous position of the agent ahead
        for i in range(1, len(current_positions)):
            # Get position of agent ahead from previous step
            ahead_agent_previous_pos = position_history[i - 1][
                -2
            ]  # Position before its current one
            current_pos = current_positions[i]

            # If not already at that position, try to move one step toward it
            if current_pos != ahead_agent_previous_pos:
                # Build set of positions occupied by other agents
                others = set()
                for j, pos in enumerate(current_positions):
                    if j != i:  # Skip self
                        others.add(pos)

                # Find path avoiding other agents and obstacles
                mini_path = bfs_dynamic(
                    current_pos,
                    ahead_agent_previous_pos,
                    grid,
                    others,
                    obstacle_positions,
                )

                if mini_path and len(mini_path) > 0:
                    # Move just one step along the path
                    current_positions[i] = mini_path[0]

            # Record this position in history
            position_history[i].append(current_positions[i])

        # Create step data for React frontend
        step_agents = []
        for i, pos in enumerate(current_positions):
            step_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

        simulation_steps.append(
            {"step": step_idx, "agents": step_agents, "paths": path_visualization}
        )

    # -------- PHASE 2: FOLLOWER MOVEMENT --------
    # After leader reaches target, followers navigate independently to their targets
    step_counter = len(full_leader_path)
    max_steps = 100  # Limit total steps

    # Create agent objects for independent movement
    followers = []
    for i in range(1, len(current_positions)):
        target = None
        pos = current_positions[i]
        if pos in follower_assignments:
            target = follower_assignments[pos]
        else:
            # Try to find a target if not assigned
            for agent_pos, target_pos in follower_assignments.items():
                if manhattan_dist(agent_pos, pos) < 3:  # Close enough
                    target = target_pos
                    break

        if target is not None:
            followers.append({"id": i, "pos": pos, "target": target})

    # Continue until all followers reach targets or max steps
    while step_counter < max_steps:
        # Check if all followers reached targets
        all_reached = True
        for follower in followers:
            if follower["pos"] != follower["target"]:
                all_reached = False
                break

        if all_reached:
            break

        # Move followers independently
        agent_positions_set = {current_positions[0]}  # Leader position is fixed
        moves = {}

        for follower in followers:
            current = follower["pos"]
            target = follower["target"]

            if current == target:
                moves[current] = current
                agent_positions_set.add(current)
                continue

            # Find path avoiding other agents and obstacles
            others = agent_positions_set - {current}
            path = bfs_dynamic(current, target, grid, others, obstacle_positions)

            if path and len(path) > 0:
                next_pos = path[0]
                if next_pos not in agent_positions_set:
                    moves[current] = next_pos
                    agent_positions_set.add(next_pos)
                else:
                    # Collision - stay put
                    moves[current] = current
                    agent_positions_set.add(current)
            else:
                # No path - stay put
                moves[current] = current
                agent_positions_set.add(current)

        # Update positions
        for follower in followers:
            old_pos = follower["pos"]
            follower["pos"] = moves.get(old_pos, old_pos)

        # Update current_positions
        for i, follower in enumerate(followers, 1):
            current_positions[i] = follower["pos"]

        # Create step data for React frontend
        step_agents = []
        for i, pos in enumerate(current_positions):
            step_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

        simulation_steps.append(
            {"step": step_counter, "agents": step_agents, "paths": path_visualization}
        )

        step_counter += 1

    return simulation_steps


# ======================================================
# 3. Centralized Algorithm
# ======================================================


def move_agents_centralized(
    grid, agent_positions, target_positions, obstacle_positions
):
    """
    Centralized movement approach adapted for the React frontend:
    - The first agent (leader) visits all target positions one by one
    - As the leader visits each target, the closest agent is assigned to that position
    - After the leader's tour, other agents move one by one to their assigned targets
    """
    # Copy agent positions to avoid modifying the original list
    current_positions = agent_positions.copy()

    # Store assignments (target_position: agent_index)
    target_assignments = {}
    available_agents = list(
        range(1, len(current_positions))
    )  # All agents except leader

    # Leader starts at its current position
    leader_pos = current_positions[0]

    # For React frontend - collect all steps
    simulation_steps = []

    # Add initial state
    initial_agents = []
    for i, pos in enumerate(current_positions):
        initial_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

    # Create path visualization for the React frontend
    path_visualization = {}

    simulation_steps.append(
        {"step": 0, "agents": initial_agents, "paths": path_visualization}
    )

    # Set to track visited targets
    visited_targets = set()
    step_counter = 1

    # -------- PHASE 1: LEADER TOUR --------
    # Leader visits each target position one by one
    for target_idx, target in enumerate(target_positions):
        # Skip if target has already been visited
        if target in visited_targets:
            continue

        # Find path for leader to current target
        obstacles = set(current_positions[1:])  # All agents except leader
        leader_path = bfs_dynamic(
            leader_pos, target, grid, obstacles, obstacle_positions
        )

        if not leader_path:
            continue  # Skip if no path found

        # Add path to visualization
        for cell in leader_path:
            r, c = cell
            if (r, c) != leader_pos and (r, c) != target:  # Skip start and end
                if r not in path_visualization:
                    path_visualization[r] = {}
                path_visualization[r][c] = True

        # Move leader along path
        for step in leader_path:
            current_positions[0] = step

            # Create step data for React frontend
            step_agents = []
            for i, pos in enumerate(current_positions):
                step_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

            simulation_steps.append(
                {
                    "step": step_counter,
                    "agents": step_agents,
                    "paths": path_visualization,
                }
            )

            step_counter += 1

        # Mark the target as visited
        visited_targets.add(target)

        # Find the closest available agent to this target
        if available_agents:
            min_dist = float("inf")
            closest_agent_idx = -1

            for agent_idx in available_agents:
                dist = manhattan_dist(current_positions[agent_idx], target)
                if dist < min_dist:
                    min_dist = dist
                    closest_agent_idx = agent_idx

            # Assign target to closest agent
            if closest_agent_idx != -1:
                target_assignments[target] = closest_agent_idx
                available_agents.remove(closest_agent_idx)

        leader_pos = target  # Update leader position

    # -------- PHASE 2: AGENT MOVEMENT TO TARGETS --------
    # Move assigned agents to their targets
    for target, agent_idx in target_assignments.items():
        agent_pos = current_positions[agent_idx]

        # Skip if agent is already at target
        if agent_pos == target:
            continue

        # Create set of obstacles (all agents except the current one)
        obstacles_and_agents = set(obstacle_positions)
        for i, pos in enumerate(current_positions):
            if i != agent_idx:
                obstacles_and_agents.add(pos)

        # Find path avoiding other agents and obstacles
        agent_path = bfs_dynamic(agent_pos, target, grid, set(), obstacles_and_agents)

        if not agent_path:
            continue  # Skip if no path found

        # Move agent along path
        for step in agent_path:
            current_positions[agent_idx] = step

            # Create step data for React frontend
            step_agents = []
            for i, pos in enumerate(current_positions):
                step_agents.append({"id": i + 1, "x": pos[0], "y": pos[1]})

            simulation_steps.append(
                {
                    "step": step_counter,
                    "agents": step_agents,
                    "paths": path_visualization,
                }
            )

            step_counter += 1

    return simulation_steps


def shift_shape_coords(shape, grid_dims):
    """Center the shape in the grid"""
    rows, cols = grid_dims
    shape_rows = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    shape_cols = max(c for _, c in shape) - min(c for _, c in shape) + 1

    row_offset = (rows - shape_rows) // 2
    col_offset = (cols - shape_cols) // 2

    return [(r + row_offset, c + col_offset) for r, c in shape]

# ======================================================
# 4. Genetic Algorithm
# ======================================================

def run_genetic_algorithm(grid, agent_positions, target_positions, obstacle_positions):
    """
    Genetic Algorithm for agent-to-target assignment (minimize total path cost using movement logic).
    Each chromosome is a permutation of target indices.
    """
    import copy
    import random

    num_agents = len(agent_positions)
    num_targets = len(target_positions)
    population_size = 10
    generations = 10
    mutation_rate = 0.1

    # Each chromosome is a permutation of target indices (length = num_agents)
    def random_chromosome():
        indices = list(range(num_targets))
        random.shuffle(indices)
        # Ensure chromosome length matches the number of agents, even if fewer targets
        if num_agents > num_targets:
            # If more agents than targets, some agents won't have a target initially
            # The chromosome will assign targets to the first num_targets agents
            return indices + [None] * (num_agents - num_targets)
        else:
            # If more targets than agents, assign agents to a subset of targets
            return indices[:num_agents]

    def evaluate_fitness(chrom):
        # Assign agents to targets according to chromosome
        assignments = []
        valid_target_indices = [idx for idx in chrom if idx is not None] # Filter out None for agents without targets
        targets_to_assign = [target_positions[i] for i in valid_target_indices]

        # Use Hungarian for optimal assignment if needed, or direct mapping
        # For simplicity here, we map agent i to chrom[i] if chrom[i] is not None
        agents_for_sim = []
        assigned_target_indices = set()
        for i in range(num_agents):
            target_idx = chrom[i]
            if target_idx is not None and target_idx < num_targets:
                 # Ensure unique target assignment if needed, though GA might explore non-unique
                 # For this fitness evaluation, allow multiple agents to target the same initially
                 target_pos = target_positions[target_idx]
                 agents_for_sim.append({"id": i, "pos": agent_positions[i], "target": target_pos})
                 assigned_target_indices.add(target_idx)
            else:
                 # Agent has no target assigned by this chromosome
                 agents_for_sim.append({"id": i, "pos": agent_positions[i], "target": None})


        # Simulate movement (no animation, just count steps)
        sim_grid = copy.deepcopy(grid)
        sim_agents = copy.deepcopy(agents_for_sim)
        step_counter = 0
        max_steps = 100 # Limit simulation steps for fitness evaluation

        # --- Simulation loop for fitness evaluation ---
        while step_counter < max_steps:
            all_reached = True
            move_dict = {}
            conflict_positions = set() # Track intended next positions
            agent_positions_set = set(ag["pos"] for ag in sim_agents)

            # Determine intended moves and detect conflicts
            for ag in sim_agents:
                current = ag["pos"]
                target = ag["target"]
                if target is None or current == target:
                    move_dict[current] = current
                    conflict_positions.add(current) # Agent stays, occupies current cell
                    continue

                all_reached = False
                agent_set_except_self = agent_positions_set - {current}
                # Use sim_grid here
                path = bfs_dynamic(current, target, sim_grid, agent_set_except_self, obstacle_positions)

                if path:
                    next_step = path[0]
                    # Check against obstacles and other agents' intended moves
                    if next_step not in conflict_positions and next_step not in obstacle_positions:
                        move_dict[current] = next_step
                        conflict_positions.add(next_step) # Reserve this cell
                    else:
                        # Conflict: stay put
                        move_dict[current] = current
                        conflict_positions.add(current)
                else:
                    # No path: stay put
                    move_dict[current] = current
                    conflict_positions.add(current)

            # Resolve direct swaps (A->B, B->A should result in A->A, B->B)
            final_moves = {}
            processed_swaps = set()
            for old_p, new_p in move_dict.items():
                if old_p in processed_swaps:
                    continue
                if new_p != old_p and new_p in move_dict and move_dict[new_p] == old_p:
                    final_moves[old_p] = old_p
                    final_moves[new_p] = new_p
                    processed_swaps.add(old_p)
                    processed_swaps.add(new_p)
                elif old_p not in final_moves:
                     final_moves[old_p] = new_p

            # Update agent positions using final_moves
            updated_agents = []
            current_positions_next_step = set()
            for ag in sim_agents:
                old_pos = ag["pos"]
                new_pos = final_moves.get(old_pos, old_pos)

                # Safety check: ensure no two agents end up in the same final position or obstacle
                if new_pos in current_positions_next_step or new_pos in obstacle_positions:
                    new_pos = old_pos # If conflict persists or obstacle, stay put
                current_positions_next_step.add(new_pos)

                updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})

            sim_agents = updated_agents
            step_counter += 1
            if all_reached:
                break
        # --- End Simulation loop ---

        # Fitness is lower for faster completion (fewer steps)
        # Add penalty if not all reached? Or just use steps? Using steps for now.
        # Consider adding a penalty for agents not reaching target:
        penalty = 0
        for ag in sim_agents:
            if ag["target"] is not None and ag["pos"] != ag["target"]:
                penalty += max_steps # Add significant penalty for each agent not reaching target

        # print(f"GA Eval: Chrom {chrom} -> Steps: {step_counter}, Penalty: {penalty}")
        return step_counter + penalty # Lower is better

    def crossover(parent1, parent2):
        # Order crossover (OX) - suitable for permutation-based chromosomes
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size

        # Copy segment from parent1
        child[a:b] = parent1[a:b]
        parent1_segment_targets = set(filter(None, parent1[a:b])) # Targets used in the segment

        # Fill remaining slots from parent2, avoiding duplicates from segment
        fill_values = []
        for item in parent2:
            if item not in parent1_segment_targets:
                 fill_values.append(item)

        child_idx = 0
        fill_idx = 0
        while child_idx < size:
            if child[child_idx] is None:
                 # Ensure we don't run out of fill_values if lengths differ
                 if fill_idx < len(fill_values):
                     # Avoid adding a target already present elsewhere in the child
                     if fill_values[fill_idx] not in filter(None, child):
                         child[child_idx] = fill_values[fill_idx]
                     else:
                         # If target already used, try next fill value or leave None
                         # This simple approach might leave Nones, handle appropriately
                         pass # Or find another unused value if strict permutation needed
                 fill_idx += 1
            child_idx += 1

        # Fill any remaining Nones if possible (e.g., if duplicates were skipped)
        # This part needs careful handling depending on whether duplicates are allowed
        # or if all agents must have a unique target if available.
        # For now, leave as potentially having Nones.

        return child


    def mutate(chrom):
        # Swap mutation for permutation-based chromosomes
        size = len(chrom)
        if size < 2: return # Cannot mutate if less than 2 elements
        a, b = random.sample(range(size), 2)
        chrom[a], chrom[b] = chrom[b], chrom[a]


    # Initialize population
    population = [random_chromosome() for _ in range(population_size)]

    # --- Genetic Algorithm Evolution Loop ---
    for gen in range(generations):
        # Evaluate fitness of the current population
        fitness_scores = [evaluate_fitness(chrom) for chrom in population]

        # Sort population by fitness (lower is better)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]

        print(f"GA Generation {gen+1}/{generations}")
        print(f"  Best fitness this gen: {evaluate_fitness(sorted_population[0])}")

        next_gen = []

        # Elitism: Keep the best individuals
        elitism_count = max(1, population_size // 5) # Keep best 20% (at least 1)
        next_gen.extend(sorted_population[:elitism_count])

        # Selection and Crossover
        # Use tournament selection or roulette wheel - Tournament is simpler here
        while len(next_gen) < population_size:
            # Select parents using tournament selection
            tournament_size = 3
            p1_candidates = random.sample(sorted_population, tournament_size)
            p2_candidates = random.sample(sorted_population, tournament_size)
            parent1 = min(p1_candidates, key=evaluate_fitness)
            parent2 = min(p2_candidates, key=evaluate_fitness)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                mutate(child)

            next_gen.append(child)

        population = next_gen
    # --- End GA Evolution Loop ---

    # Final evaluation to find the best chromosome
    final_fitness_scores = [evaluate_fitness(chrom) for chrom in population]
    best_chrom_index = final_fitness_scores.index(min(final_fitness_scores))
    best_chromosome = population[best_chrom_index]

    print(f"GA: Best assignment chromosome found: {best_chromosome}")

    # Create final assignments based on the best chromosome
    final_assignments = []
    agents_final = []
    assigned_target_indices_final = set()
    for i in range(num_agents):
        target_idx = best_chromosome[i]
        if target_idx is not None and target_idx < num_targets:
            # Simple assignment: agent i gets target best_chromosome[i]
            # Could add logic here to prevent multiple agents assigned to same target if needed
            target_pos = target_positions[target_idx]
            final_assignments.append((agent_positions[i], target_pos))
            agents_final.append({"id": i, "pos": agent_positions[i], "target": target_pos})
            assigned_target_indices_final.add(target_idx)
        else:
            # Agent has no target
            agents_final.append({"id": i, "pos": agent_positions[i], "target": None})

    print(f"GA: Final Assignments derived: {final_assignments}")


    # --- Animate the best solution using the collision-aware logic ---
    agents = copy.deepcopy(agents_final) # Use the agents derived from the best chromosome
    simulation_steps = []
    initial_agents_state = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
    simulation_steps.append({"step": 0, "agents": initial_agents_state, "paths": {}})

    step_counter = 1
    max_animation_steps = 200 # Allow more steps for animation if needed
    while step_counter < max_animation_steps:
        all_reached = True
        move_dict = {}
        conflict_positions = set() # Keep track of intended next positions
        agent_positions_set = set(ag["pos"] for ag in agents)

        # Determine intended moves and detect conflicts
        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                conflict_positions.add(current) # Agent stays, occupies current cell
                continue

            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            # Use the original grid for pathfinding during animation
            path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)

            if path:
                next_step = path[0]
                # Check against obstacles and other agents' intended moves
                if next_step not in conflict_positions and next_step not in obstacle_positions:
                    move_dict[current] = next_step
                    conflict_positions.add(next_step) # Reserve this cell for next step
                else:
                    # Conflict: stay put
                    move_dict[current] = current
                    conflict_positions.add(current) # Stay put, occupy current cell
            else:
                # No path: stay put
                move_dict[current] = current
                conflict_positions.add(current) # Stay put, occupy current cell

        # Resolve direct swaps (A->B, B->A should result in A->A, B->B)
        final_moves = {}
        processed_swaps = set()
        for old_p, new_p in move_dict.items():
            if old_p in processed_swaps:
                continue
            # Check if the target cell's occupant wants to move to the current cell
            if new_p != old_p and new_p in move_dict and move_dict[new_p] == old_p:
                # Direct swap detected: both agents stay put
                final_moves[old_p] = old_p
                final_moves[new_p] = new_p
                processed_swaps.add(old_p)
                processed_swaps.add(new_p)
            elif old_p not in final_moves: # Ensure not already processed as part of a swap
                 final_moves[old_p] = new_p

        # Update agent positions using final_moves
        updated_agents = []
        current_positions_next_step = set() # Track positions for safety check
        for ag in agents:
            old_pos = ag["pos"]
            # Use final_moves, default to staying put if somehow missing
            new_pos = final_moves.get(old_pos, old_pos)

            # Safety check: ensure no two agents end up in the same final position or obstacle
            if new_pos in current_positions_next_step or new_pos in obstacle_positions:
                # print(f"GA Sim: Persistent conflict or obstacle at {new_pos} for agent {ag['id']}. Staying at {old_pos}.")
                new_pos = old_pos # If conflict persists or obstacle, stay put
            current_positions_next_step.add(new_pos)

            updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})

        agents = updated_agents

        # Record step for animation
        step_agents_state = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
        simulation_steps.append({"step": step_counter, "agents": step_agents_state, "paths": {}}) # Add paths if needed

        step_counter += 1
        if all_reached:
            print(f"GA: All agents reached targets in {step_counter-1} animation steps.")
            break
    # Add a check if max_steps was reached without completion
    if not all_reached and step_counter >= max_animation_steps:
         print(f"GA: Max animation steps ({max_animation_steps}) reached.")
         
    return simulation_steps
    """
    Genetic Algorithm for agent-to-target assignment (minimize total path cost using movement logic).
    Each chromosome is a permutation of target indices.
    """
    import copy

    num_agents = len(agent_positions)
    num_targets = len(target_positions)
    population_size = 10
    generations = 10
    mutation_rate = 0.2

    # Each chromosome is a permutation of target indices (length = num_agents)
    def random_chromosome():
        indices = list(range(num_targets))
        random.shuffle(indices)
        return indices[:num_agents]

    def evaluate_fitness(chrom):
        # Assign agents to targets according to chromosome
        assignments = [(agent_positions[i], target_positions[chrom[i]]) for i in range(min(num_agents, num_targets))]
        # Simulate movement (no animation, just count steps)
        agents = []
        for idx, (agent, target) in enumerate(assignments):
            agents.append({"id": idx, "pos": agent, "target": target})

        # Add agents without targets (if more agents than targets)
        assigned_positions = set(ag["pos"] for ag in agents)
        for idx, agent in enumerate(agent_positions):
            if agent not in assigned_positions:
                agents.append({"id": idx, "pos": agent, "target": None})

        sim_grid = copy.deepcopy(grid)
        sim_agents = copy.deepcopy(agents)
        step_counter = 0
        max_steps = 100
        while step_counter < max_steps:
            all_reached = True
            move_dict = {}
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            for ag in sim_agents:
                current = ag["pos"]
                target = ag["target"]
                if target is None or current == target:
                    move_dict[current] = current
                    continue
                all_reached = False
                agent_set_except_self = agent_positions_set - {current}
                path = bfs_dynamic(current, target, sim_grid, agent_set_except_self, obstacle_positions)
                if path:
                    move_dict[current] = path[0]
                else:
                    move_dict[current] = current
            updated_agents = []
            for ag in sim_agents:
                old_pos = ag["pos"]
                new_pos = move_dict[old_pos]
                updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
            sim_agents = updated_agents
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            step_counter += 1
            if all_reached:
                break
        if not all_reached:
            print(f"GA: Not all agents reached targets in max_steps ({max_steps}) for chrom {chrom}")
        else:
            print(f"GA: All agents reached targets in {step_counter} steps for chrom {chrom}")
        return step_counter  # Lower is better

    def crossover(parent1, parent2):
        # Order crossover (OX)
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]
        fill = [x for x in parent2 if x not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def mutate(chrom):
        a, b = random.sample(range(len(chrom)), 2)
        chrom[a], chrom[b] = chrom[b], chrom[a]

    # Initialize population
    population = [random_chromosome() for _ in range(population_size)]

    for gen in range(generations):
        print(f"GA Generation {gen+1}/{generations}")
        population.sort(key=evaluate_fitness)
        print(f"  Best fitness this gen: {evaluate_fitness(population[0])}")
        next_gen = population[:4]  # Elitism: keep best 4
        while len(next_gen) < population_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_rate:
                mutate(child)
            next_gen.append(child)
        population = next_gen

    # Best assignment
    best = min(population, key=evaluate_fitness)
    print(f"GA: Best assignment found: {best}")
    assignments = [(agent_positions[i], target_positions[best[i]]) for i in range(min(num_agents, num_targets))]
    print(f"GA: Assignments: {assignments}")

    # Now, animate using your existing movement logic (step-by-step)
    agents = []
    for idx, (agent, target) in enumerate(assignments):
        agents.append({"id": idx, "pos": agent, "target": target})

    # Add agents without targets (if more agents than targets)
    assigned_positions = set(ag["pos"] for ag in agents)
    for idx, agent in enumerate(agent_positions):
        if agent not in assigned_positions:
            agents.append({"id": idx, "pos": agent, "target": None})

    # Step-by-step movement (reuse your movement logic)
    simulation_steps = []
    initial_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
    simulation_steps.append({"step": 0, "agents": initial_agents, "paths": {}})

    step_counter = 1
    max_steps = 100
    while step_counter < max_steps:
        all_reached = True
        move_dict = {}
        agent_positions_set = set(ag["pos"] for ag in agents)
        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                continue
            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)
            if path:
                move_dict[current] = path[0]
            else:
                move_dict[current] = current
        updated_agents = []
        for ag in agents:
            old_pos = ag["pos"]
            new_pos = move_dict[old_pos]
            updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
        agents = updated_agents
        agent_positions_set = set(ag["pos"] for ag in agents)
        step_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
        simulation_steps.append({"step": step_counter, "agents": step_agents, "paths": {}})
        step_counter += 1
        if all_reached:
            print(f"GA: All agents reached targets in {step_counter} animation steps.")
            break

    return simulation_steps

    """
    Genetic Algorithm for agent-to-target assignment (minimize total path cost using movement logic).
    Each chromosome is a permutation of target indices.
    """
    import copy

    num_agents = len(agent_positions)
    num_targets = len(target_positions)
    population_size = 10
    generations = 10
    mutation_rate = 0.2

    # Each chromosome is a permutation of target indices (length = num_agents)
    def random_chromosome():
        indices = list(range(num_targets))
        random.shuffle(indices)
        return indices[:num_agents]

    def evaluate_fitness(chrom):
        # Assign agents to targets according to chromosome
        assignments = [(agent_positions[i], target_positions[chrom[i]]) for i in range(num_agents)]
        # Simulate movement (no animation, just count steps)
        agents = []
        for idx, (agent, target) in enumerate(assignments):
            agents.append({"id": idx, "pos": agent, "target": target})

        # Use a copy of the grid to avoid modifying the original
        sim_grid = copy.deepcopy(grid)
        sim_agents = copy.deepcopy(agents)
        step_counter = 0
        max_steps = 100
        while step_counter < max_steps:
            all_reached = True
            move_dict = {}
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            for ag in sim_agents:
                current = ag["pos"]
                target = ag["target"]
                if target is None or current == target:
                    move_dict[current] = current
                    continue
                all_reached = False
                agent_set_except_self = agent_positions_set - {current}
                path = bfs_dynamic(current, target, sim_grid, agent_set_except_self, obstacle_positions)
                if path:
                    move_dict[current] = path[0]
                else:
                    move_dict[current] = current
            updated_agents = []
            for ag in sim_agents:
                old_pos = ag["pos"]
                new_pos = move_dict[old_pos]
                updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
            sim_agents = updated_agents
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            step_counter += 1
            if all_reached:
                break
        if not all_reached:
            print(f"GA: Not all agents reached targets in max_steps ({max_steps}) for chrom {chrom}")
        else:
            print(f"GA: All agents reached targets in {step_counter} steps for chrom {chrom}")
        return step_counter  # Lower is better

    def crossover(parent1, parent2):
        # Order crossover (OX)
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]
        fill = [x for x in parent2 if x not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def mutate(chrom):
        a, b = random.sample(range(len(chrom)), 2)
        chrom[a], chrom[b] = chrom[b], chrom[a]

    # Initialize population
    population = [random_chromosome() for _ in range(population_size)]

    for gen in range(generations):
        print(f"GA Generation {gen+1}/{generations}")
        population.sort(key=evaluate_fitness)
        print(f"  Best fitness this gen: {evaluate_fitness(population[0])}")
        next_gen = population[:4]  # Elitism: keep best 4
        while len(next_gen) < population_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_rate:
                mutate(child)
            next_gen.append(child)
        population = next_gen

    # Best assignment
    best = min(population, key=evaluate_fitness)
    print(f"GA: Best assignment found: {best}")
    assignments = [(agent_positions[i], target_positions[best[i]]) for i in range(num_agents)]
    print(f"GA: Assignments: {assignments}")

    # Now, animate using your existing movement logic (step-by-step)
    agents = []
    for idx, (agent, target) in enumerate(assignments):
        agents.append({"id": idx, "pos": agent, "target": target})

    # Track any agents without targets (if more agents than targets)
    unassigned_agents = [a for a in agent_positions if a not in [ag["pos"] for ag in agents]]
    for agent in unassigned_agents:
        idx = agent_positions.index(agent)
        agents.append({"id": idx, "pos": agent, "target": None})

    # Step-by-step movement (reuse your movement logic)
    simulation_steps = []
    initial_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
    simulation_steps.append({"step": 0, "agents": initial_agents, "paths": {}})

    step_counter = 1
    max_steps = 100
    while step_counter < max_steps:
        all_reached = True
        move_dict = {}
        agent_positions_set = set(ag["pos"] for ag in agents)
        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                continue
            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)
            if path:
                move_dict[current] = path[0]
            else:
                move_dict[current] = current
        updated_agents = []
        for ag in agents:
            old_pos = ag["pos"]
            new_pos = move_dict[old_pos]
            updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
        agents = updated_agents
        agent_positions_set = set(ag["pos"] for ag in agents)
        step_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
        simulation_steps.append({"step": step_counter, "agents": step_agents, "paths": {}})
        step_counter += 1
        if all_reached:
            print(f"GA: All agents reached targets in {step_counter} animation steps.")
            break

    return simulation_steps

    """
    Genetic Algorithm for agent-to-target assignment (minimize total path cost using movement logic).
    Each chromosome is a permutation of target indices.
    """
    import copy

    num_agents = len(agent_positions)
    num_targets = len(target_positions)
    population_size = 10
    generations = 10
    mutation_rate = 0.1

    # Each chromosome is a permutation of target indices (length = num_agents)
    def random_chromosome():
        indices = list(range(num_targets))
        random.shuffle(indices)
        return indices[:num_agents]

    def evaluate_fitness(chrom):
        # Assign agents to targets according to chromosome
        assignments = [(agent_positions[i], target_positions[chrom[i]]) for i in range(num_agents)]
        # Simulate movement (no animation, just count steps)
        agents = []
        for idx, (agent, target) in enumerate(assignments):
            agents.append({"id": idx, "pos": agent, "target": target})

        # Use a copy of the grid to avoid modifying the original
        sim_grid = copy.deepcopy(grid)
        sim_agents = copy.deepcopy(agents)
        step_counter = 0
        max_steps = 100
        while step_counter < max_steps:
            all_reached = True
            move_dict = {}
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            for ag in sim_agents:
                current = ag["pos"]
                target = ag["target"]
                if target is None or current == target:
                    move_dict[current] = current
                    continue
                all_reached = False
                agent_set_except_self = agent_positions_set - {current}
                path = bfs_dynamic(current, target, sim_grid, agent_set_except_self, obstacle_positions)
                if path:
                    move_dict[current] = path[0]
                else:
                    move_dict[current] = current
            updated_agents = []
            for ag in sim_agents:
                old_pos = ag["pos"]
                new_pos = move_dict[old_pos]
                updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
            sim_agents = updated_agents
            agent_positions_set = set(ag["pos"] for ag in sim_agents)
            step_counter += 1
            if all_reached:
                break
        return step_counter  # Lower is better

    def crossover(parent1, parent2):
        # Order crossover (OX)
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]
        fill = [x for x in parent2 if x not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    def mutate(chrom):
        a, b = random.sample(range(len(chrom)), 2)
        chrom[a], chrom[b] = chrom[b], chrom[a]

    # Initialize population
    population = [random_chromosome() for _ in range(population_size)]

    for _ in range(generations):
        population.sort(key=evaluate_fitness)
        next_gen = population[:4]  # Elitism: keep best 4
        while len(next_gen) < population_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_rate:
                mutate(child)
            next_gen.append(child)
        population = next_gen

    # Best assignment
    best = min(population, key=evaluate_fitness)
    assignments = [(agent_positions[i], target_positions[best[i]]) for i in range(num_agents)]

    # Now, animate using your existing movement logic (step-by-step)
    # Reuse the movement loop from move_agents_no_collision or similar
    agents = []
    for idx, (agent, target) in enumerate(assignments):
        agents.append({"id": idx, "pos": agent, "target": target})

    # Track any agents without targets (if more agents than targets)
    unassigned_agents = [a for a in agent_positions if a not in [ag["pos"] for ag in agents]]
    for agent in unassigned_agents:
        idx = agent_positions.index(agent)
        agents.append({"id": idx, "pos": agent, "target": None})

    # Step-by-step movement (reuse your movement logic)
    simulation_steps = []
    initial_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
    simulation_steps.append({"step": 0, "agents": initial_agents, "paths": {}})

    step_counter = 1
    max_steps = 100
    while step_counter < max_steps:
        all_reached = True
        move_dict = {}
        agent_positions_set = set(ag["pos"] for ag in agents)
        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                continue
            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)
            if path:
                move_dict[current] = path[0]
            else:
                move_dict[current] = current
        updated_agents = []
        for ag in agents:
            old_pos = ag["pos"]
            new_pos = move_dict[old_pos]
            updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})
        agents = updated_agents
        agent_positions_set = set(ag["pos"] for ag in agents)
        step_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
        simulation_steps.append({"step": step_counter, "agents": step_agents, "paths": {}})
        step_counter += 1
        if all_reached:
            break

    return simulation_steps

# ======================================================
# API Endpoints
# ======================================================

# Update to the run_simulation function in app.py to modify agent placement


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    """API endpoint to receive grid data and return simulation steps."""
    try:
        # Get data from request
        data = request.json
        grid_data = data["gridData"]
        config_data = data["configData"]

        # Convert grid data to numpy array
        grid = np.array(grid_data)

        # Extract shape positions (where value is 1)
        target_positions = []
        for r in range(len(grid_data)):
            for c in range(len(grid_data[r])):
                if grid_data[r][c] == 1:  # Shape cells
                    target_positions.append((r, c))

        # Extract obstacle positions (where value is 2)
        obstacle_positions = []
        for r in range(len(grid_data)):
            for c in range(len(grid_data[r])):
                if grid_data[r][c] == 2:  # Obstacle cells
                    obstacle_positions.append((r, c))

        # Determine number of agents based on config
        agent_count = config_data.get("agentCount", 5)

        # Place agents in the bottom left two rows
        agent_positions = []
        grid_height = len(grid_data)
        grid_width = len(grid_data[0])

        # Bottom row
        for col in range(min(agent_count, grid_width)):
            pos = (grid_height - 1, col)
            if grid_data[pos[0]][pos[1]] == 0:  # Only place on empty cells
                agent_positions.append(pos)

        # Second from bottom row if needed
        if len(agent_positions) < agent_count:
            for col in range(min(agent_count - len(agent_positions), grid_width)):
                pos = (grid_height - 2, col)
                if grid_data[pos[0]][pos[1]] == 0:  # Only place on empty cells
                    agent_positions.append(pos)

        # If we still need more agents, find additional empty cells
        if len(agent_positions) < agent_count:
            empty_cells = []
            for r in range(len(grid_data)):
                for c in range(len(grid_data[r])):
                    if (
                        grid_data[r][c] == 0 and (r, c) not in agent_positions
                    ):  # Empty cells
                        empty_cells.append((r, c))

            # Sort empty cells by bottom-to-top, left-to-right
            empty_cells.sort(key=lambda pos: (pos[0], -pos[1]), reverse=True)

            # Add needed cells
            needed = agent_count - len(agent_positions)
            agent_positions.extend(empty_cells[:needed])

        # Determine which algorithm to use
        algorithm = config_data.get("algorithm", "inside-out")

        # Generate simulation steps using the selected algorithm
        if algorithm == "leader-follower":
            simulation_steps = move_agents_leader_follower(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "centralized":
            simulation_steps = move_agents_centralized(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "Genetic-Algorithm":
            simulation_steps = run_genetic_algorithm(
                grid, agent_positions, target_positions, obstacle_positions
            )
        else:  # Default to inside-out
            simulation_steps = move_agents_inside_out(
                grid, agent_positions, target_positions, obstacle_positions
            )

        # Return result
        return jsonify({"steps": simulation_steps})

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/get_predefined_shape", methods=["GET"])
def get_predefined_shape():
    """Return a predefined shape centered in the grid."""
    try:
        shape_name = request.args.get("shape", "square")
        grid_size = int(request.args.get("gridSize", 15))

        if shape_name not in SHAPES:
            return jsonify({"error": f"Shape {shape_name} not found"}), 404

        # Get the raw shape
        shape = SHAPES[shape_name]

        # Center it in the grid
        centered_shape = shift_shape_coords(shape, (grid_size, grid_size))

        # Validate shape is within bounds
        valid_shape = [
            (r, c)
            for r, c in centered_shape
            if 0 <= r < grid_size and 0 <= c < grid_size
        ]

        return jsonify({"shape": valid_shape})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
