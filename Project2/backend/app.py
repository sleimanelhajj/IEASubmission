from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
from collections import deque
from scipy.optimize import linear_sum_assignment
import random
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import os
import heapq

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

def astar(start, goal, grid, agents, obstacles=None):
    """
    A* pathfinding algorithm.
    - start: (row, col)
    - goal: (row, col)
    - grid: numpy array
    - agents: set of positions to treat as obstacles (other agents)
    - obstacles: set of positions to treat as obstacles
    Returns a list of steps from start->goal (excluding start, including goal).
    If no path, returns [].
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + manhattan_dist(start, goal), 0, start, []))
    closed_set = set()
    if obstacles is None:
        obstacles = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in closed_set:
            continue
        closed_set.add(current)
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and neighbor not in agents
                and neighbor not in obstacles
                and neighbor not in closed_set
            ):
                heapq.heappush(
                    open_set,
                    (
                        g + 1 + manhattan_dist(neighbor, goal),
                        g + 1,
                        neighbor,
                        path + [neighbor],
                    ),
                )
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
                cost_matrix[i, j] = -manhattan_dist(
                    agent_positions[i], target_positions[j]
                )
            else:
                cost_matrix[i, j] = 999999
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for i in range(size):
        if row_ind[i] < n_agents and col_ind[i] < n_targets:
            assignments.append(
                (agent_positions[row_ind[i]], target_positions[col_ind[i]])
            )
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



def get_neighbors(pos, grid_shape):
    r, c = pos
    rows, cols = grid_shape
    # 8-direction movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def is_adjacent(pos1, pos2):
    """Returns True if pos1 is adjacent (including diagonals) to pos2."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) == 1


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


# ======================================================
# 2. Leader-Follower Algorithm
# ======================================================


def move_agents_leader_follower(grid, agent_positions, target_positions, obstacle_positions):
    """
    Leader-Follower + Parallel movement for backend API.
    Phase 1: Leader moves to highest-priority target, followers follow.
    Phase 2: Remaining agents are assigned to remaining targets (Hungarian), move in parallel.
    Returns a list of simulation steps for frontend animation.
    """
    import numpy as np

    n = min(len(agent_positions), len(target_positions))
    agent_positions = agent_positions[:n]
    target_positions = target_positions[:n]

    # --- Phase 1: Leader-Follower to highest-priority target ---
    agent_centroid = (
        sum(pos[0] for pos in agent_positions) / len(agent_positions),
        sum(pos[1] for pos in agent_positions) / len(agent_positions)
    )
    def priority(t):
        return abs(t[0] - agent_centroid[0]) + abs(t[1] - agent_centroid[1])
    sorted_targets = sorted(target_positions, key=priority, reverse=True)
    leader_target = sorted_targets[0]

    # Track agent positions by index
    current_positions = agent_positions.copy()
    position_history = [[] for _ in range(len(current_positions))]
    for i, pos in enumerate(current_positions):
        position_history[i].append(pos)

    simulation_steps = []
    simulation_steps.append({
        "step": 0,
        "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(current_positions)],
        "paths": {},
    })

    # Leader's path
    leader_path = bfs_dynamic(
        current_positions[0], leader_target, grid, set(current_positions[1:]), set(obstacle_positions)
    )
    if not leader_path:
        return simulation_steps

    # Move leader step by step, followers follow
    full_leader_path = [current_positions[0]] + leader_path
    for step_idx in range(1, len(full_leader_path)):
        current_positions[0] = full_leader_path[step_idx]
        position_history[0].append(current_positions[0])

        for i in range(1, len(current_positions)):
            ahead_prev_pos = position_history[i-1][-2]
            curr_pos = current_positions[i]
            if curr_pos != ahead_prev_pos:
                others = set(current_positions)
                others.remove(curr_pos)
                path = bfs_dynamic(curr_pos, ahead_prev_pos, grid, others, set(obstacle_positions))
                if path and path[0] not in others:
                    current_positions[i] = path[0]
            position_history[i].append(current_positions[i])

        simulation_steps.append({
            "step": step_idx,
            "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(current_positions)],
            "paths": {},
        })

    # --- Phase 2: Parallel movement for remaining agents/targets ---
    filled_targets = {leader_target}
    remaining_targets = [t for t in sorted_targets if t not in filled_targets]
    follower_indices = list(range(1, len(current_positions)))
    followers = [current_positions[i] for i in follower_indices]

    # Hungarian assignment for remaining agents/targets
    agents = []
    if followers and remaining_targets:
        follower_assignments = hungarian_assignment(followers, remaining_targets)
        # Build a mapping from position to index for followers
        pos_to_idx = {pos: idx for idx, pos in zip(follower_indices, followers)}
        for agent_pos, target in follower_assignments:
            agent_id = pos_to_idx[agent_pos]
            agents.append({
                "id": agent_id,
                "pos": agent_pos,
                "target": target
            })

    max_iterations = 200
    step_counter = len(full_leader_path)
    while step_counter < max_iterations and agents:
        all_reached = True
        moves = {}
        occupied = {current_positions[0]}  # Leader's position

        for ag in agents:
            if ag["pos"] == ag["target"]:
                moves[ag["id"]] = ag["pos"]
                occupied.add(ag["pos"])
                continue
            all_reached = False
            others = occupied.copy()
            path = bfs_dynamic(ag["pos"], ag["target"], grid, others, set(obstacle_positions))
            if path and path[0] not in occupied:
                moves[ag["id"]] = path[0]
                occupied.add(path[0])
            else:
                moves[ag["id"]] = ag["pos"]
                occupied.add(ag["pos"])

        # Apply moves by agent index
        for ag in agents:
            ag["pos"] = moves[ag["id"]]
            current_positions[ag["id"]] = ag["pos"]

        simulation_steps.append({
            "step": step_counter,
            "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(current_positions)],
            "paths": {},
        })
        step_counter += 1
        if all_reached:
            break

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
# 4. A-star Algorithm
# ======================================================


def move_agents_astar(grid, agent_positions, target_positions, obstacle_positions):
    """
    Each agent moves to its assigned target using A* pathfinding.
    Assigns agents to targets in order (agent i -> target i).
    Returns step-by-step data for the React frontend.
    """
    agents = [{"id": i, "pos": agent_positions[i], "target": target_positions[i] if i < len(target_positions) else None} for i in range(len(agent_positions))]
    simulation_steps = []
    step_counter = 0
    max_steps = 100

    # Initial state
    simulation_steps.append({
        "step": 0,
        "agents": [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents],
        "paths": {},
    })

    while step_counter < max_steps:
        all_reached = True
        move_dict = {}
        conflict_positions = set()
        agent_positions_set = set(ag["pos"] for ag in agents)

        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                conflict_positions.add(current)
                continue

            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            path = astar(current, target, grid, agent_set_except_self, obstacle_positions)
            if path:
                next_step = path[0]
                if next_step not in conflict_positions:
                    move_dict[current] = next_step
                    conflict_positions.add(next_step)
                else:
                    move_dict[current] = current
                    conflict_positions.add(current)
            else:
                move_dict[current] = current
                conflict_positions.add(current)

        # Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                final_moves[old_p] = old_p
                final_moves[new_p] = new_p
            else:
                final_moves[old_p] = new_p

        updated_agents = []
        current_positions_next_step = set()
        for ag in agents:
            old_pos = ag["pos"]
            new_pos = final_moves.get(old_pos, old_pos)
            if new_pos in current_positions_next_step or new_pos in obstacle_positions:
                new_pos = old_pos
            current_positions_next_step.add(new_pos)
            updated_agents.append({"id": ag["id"], "pos": new_pos, "target": ag["target"]})

        agents = updated_agents
        step_agents = [{"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents]
        simulation_steps.append({"step": step_counter + 1, "agents": step_agents, "paths": {}})
        step_counter += 1
        if all_reached:
            break

    return simulation_steps


# ======================================================
# 4. Genetic Algorithm
# ======================================================


def run_genetic_algorithm(grid, agent_positions, target_positions, obstacle_positions):
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
        # Ensure chromosome length matches the number of agents, even if fewer targets
        if num_agents > num_targets:
            # If more agents than targets, some agents won't have a target initially
            # The chromosome will assign targets to the first num_targets agents
            return indices + [None] * (num_agents - num_targets)
        else:
            # If more targets than agents, assign agents to a subset of targets
            return indices[:num_agents]

    fitness_cache = {}  # Add this line

    def evaluate_fitness(chrom):
        key = tuple(chrom)
        if key in fitness_cache:
            return fitness_cache[key]
        # Assign agents to targets according to chromosome
        assignments = []
        valid_target_indices = [
            idx for idx in chrom if idx is not None
        ]  # Filter out None for agents without targets
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
                agents_for_sim.append(
                    {"id": i, "pos": agent_positions[i], "target": target_pos}
                )
                assigned_target_indices.add(target_idx)
            else:
                # Agent has no target assigned by this chromosome
                agents_for_sim.append(
                    {"id": i, "pos": agent_positions[i], "target": None}
                )

        # Simulate movement (no animation, just count steps)
        sim_grid = copy.deepcopy(grid)
        sim_agents = copy.deepcopy(agents_for_sim)
        step_counter = 0
        max_steps = 100  # Limit simulation steps for fitness evaluation

        # --- Simulation loop for fitness evaluation ---
        while step_counter < max_steps:
            all_reached = True
            move_dict = {}
            conflict_positions = set()  # Track intended next positions
            agent_positions_set = set(ag["pos"] for ag in sim_agents)

            # Determine intended moves and detect conflicts
            for ag in sim_agents:
                current = ag["pos"]
                target = ag["target"]
                if target is None or current == target:
                    move_dict[current] = current
                    conflict_positions.add(
                        current
                    )  # Agent stays, occupies current cell
                    continue

                all_reached = False
                agent_set_except_self = agent_positions_set - {current}
                # Use sim_grid here
                path = bfs_dynamic(
                    current, target, sim_grid, agent_set_except_self, obstacle_positions
                )

                if path:
                    next_step = path[0]
                    # Check against obstacles and other agents' intended moves
                    if (
                        next_step not in conflict_positions
                        and next_step not in obstacle_positions
                    ):
                        move_dict[current] = next_step
                        conflict_positions.add(next_step)  # Reserve this cell
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
                if (
                    new_pos in current_positions_next_step
                    or new_pos in obstacle_positions
                ):
                    new_pos = old_pos  # If conflict persists or obstacle, stay put
                current_positions_next_step.add(new_pos)

                updated_agents.append(
                    {"id": ag["id"], "pos": new_pos, "target": ag["target"]}
                )

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
                penalty += max_steps  # Add significant penalty for each agent not reaching target

        # print(f"GA Eval: Chrom {chrom} -> Steps: {step_counter}, Penalty: {penalty}")
        fitness = step_counter + penalty  # Lower is better
        fitness_cache[key] = fitness
        return fitness  # Lower is better

    def crossover(parent1, parent2):
        # Order crossover (OX) - suitable for permutation-based chromosomes
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size

        # Copy segment from parent1
        child[a:b] = parent1[a:b]
        parent1_segment_targets = set(
            filter(None, parent1[a:b])
        )  # Targets used in the segment

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
                        pass  # Or find another unused value if strict permutation needed
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
        if size < 2:
            return  # Cannot mutate if less than 2 elements
        a, b = random.sample(range(size), 2)
        chrom[a], chrom[b] = chrom[b], chrom[a]

    # Initialize population
    population = [random_chromosome() for _ in range(population_size)]

    # --- Genetic Algorithm Evolution Loop ---
    for gen in range(generations):
        # Evaluate fitness of the current population
        fitness_scores = [evaluate_fitness(chrom) for chrom in population]

        # Sort population by fitness (lower is better)
        sorted_population = [
            x
            for _, x in sorted(
                zip(fitness_scores, population), key=lambda pair: pair[0]
            )
        ]

        print(f"GA Generation {gen+1}/{generations}")
        print(f"  Best fitness this gen: {evaluate_fitness(sorted_population[0])}")

        next_gen = []

        # Elitism: Keep the best individuals
        elitism_count = max(1, population_size // 5)  # Keep best 20% (at least 1)
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
            agents_final.append(
                {"id": i, "pos": agent_positions[i], "target": target_pos}
            )
            assigned_target_indices_final.add(target_idx)
        else:
            # Agent has no target
            agents_final.append({"id": i, "pos": agent_positions[i], "target": None})

    print(f"GA: Final Assignments derived: {final_assignments}")

    # --- Animate the best solution using the collision-aware logic ---
    agents = copy.deepcopy(
        agents_final
    )  # Use the agents derived from the best chromosome
    simulation_steps = []
    initial_agents_state = [
        {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents
    ]
    simulation_steps.append({"step": 0, "agents": initial_agents_state, "paths": {}})

    step_counter = 1
    max_animation_steps = 200  # Allow more steps for animation if needed
    while step_counter < max_animation_steps:
        all_reached = True
        move_dict = {}
        conflict_positions = set()  # Keep track of intended next positions
        agent_positions_set = set(ag["pos"] for ag in agents)

        # Determine intended moves and detect conflicts
        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                conflict_positions.add(current)  # Agent stays, occupies current cell
                continue

            all_reached = False
            agent_set_except_self = agent_positions_set - {current}
            # Use the original grid for pathfinding during animation
            path = bfs_dynamic(
                current, target, grid, agent_set_except_self, obstacle_positions
            )

            if path:
                next_step = path[0]
                # Check against obstacles and other agents' intended moves
                if (
                    next_step not in conflict_positions
                    and next_step not in obstacle_positions
                ):
                    move_dict[current] = next_step
                    conflict_positions.add(next_step)  # Reserve this cell for next step
                else:
                    # Conflict: stay put
                    move_dict[current] = current
                    conflict_positions.add(current)  # Stay put, occupy current cell
            else:
                # No path: stay put
                move_dict[current] = current
                conflict_positions.add(current)  # Stay put, occupy current cell

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
            elif (
                old_p not in final_moves
            ):  # Ensure not already processed as part of a swap
                final_moves[old_p] = new_p

        # Update agent positions using final_moves
        updated_agents = []
        current_positions_next_step = set()  # Track positions for safety check
        for ag in agents:
            old_pos = ag["pos"]
            # Use final_moves, default to staying put if somehow missing
            new_pos = final_moves.get(old_pos, old_pos)

            # Safety check: ensure no two agents end up in the same final position or obstacle
            if new_pos in current_positions_next_step or new_pos in obstacle_positions:
                # print(f"GA Sim: Persistent conflict or obstacle at {new_pos} for agent {ag['id']}. Staying at {old_pos}.")
                new_pos = old_pos  # If conflict persists or obstacle, stay put
            current_positions_next_step.add(new_pos)

            updated_agents.append(
                {"id": ag["id"], "pos": new_pos, "target": ag["target"]}
            )

        agents = updated_agents

        # Record step for animation
        step_agents_state = [
            {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents
        ]
        simulation_steps.append(
            {"step": step_counter, "agents": step_agents_state, "paths": {}}
        )  # Add paths if needed

        step_counter += 1
        if all_reached:
            print(
                f"GA: All agents reached targets in {step_counter-1} animation steps."
            )
            break
    # Add a check if max_steps was reached without completion
    if not all_reached and step_counter >= max_animation_steps:
        print(f"GA: Max animation steps ({max_animation_steps}) reached.")

    return simulation_steps


# ======================================================
# Cellular Automata
# ======================================================


def move_agents_cellular_automata(
    grid, agent_positions, target_positions, obstacle_positions
):
    """
    Pure CA: No randomness, only local information.
    - Settled agents (on target) with empty target neighbors "invite" all unsettled agents.
    - Unsettled agents move toward the nearest invitation using only local moves.
    - Once an agent reaches a target, it locks/settles.
    """
    targets = set(target_positions)
    obstacles = set(obstacle_positions)
    simulation_steps = []
    step_counter = 0
    max_steps = 500

    # Agent state: id, pos, settled (on target)
    agents = [
        {"id": i, "pos": pos, "settled": False} for i, pos in enumerate(agent_positions)
    ]

    simulation_steps.append(
        {
            "step": 0,
            "agents": [
                {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]}
                for ag in agents
            ],
            "paths": {},
        }
    )

    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    neighbors.append((nr, nc))
        return neighbors

    while step_counter < max_steps:
        occupied = {ag["pos"] for ag in agents} | obstacles
        settled_positions = {ag["pos"] for ag in agents if ag["settled"]}

        # Step 1: Settled agents broadcast invitations for empty target neighbors
        invitations = set()
        for ag in agents:
            if ag["settled"]:
                for n in get_neighbors(ag["pos"]):
                    if n in targets and n not in occupied:
                        invitations.add(n)

        # Step 2: Each agent decides where to move (no randomness)
        intended_moves = {}
        for ag in agents:
            if ag["settled"]:
                intended_moves[ag["id"]] = ag["pos"]
                continue

            # If on a target, settle
            if ag["pos"] in targets:
                ag["settled"] = True
                intended_moves[ag["id"]] = ag["pos"]
                continue

            # If there are invitations, move toward the closest invitation
            if invitations:
                # Move toward the closest invitation
                min_dist = float("inf")
                best_neighbor = ag["pos"]
                for n in [ag["pos"]] + get_neighbors(ag["pos"]):
                    if n not in occupied and n not in intended_moves.values():
                        dist = min(
                            abs(n[0] - inv[0]) + abs(n[1] - inv[1])
                            for inv in invitations
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = n
                intended_moves[ag["id"]] = best_neighbor
            else:
                # No invitations: only allow up (north) movement toward the closest empty target
                empty_targets = [
                    t
                    for t in targets
                    if t not in occupied and t not in intended_moves.values()
                ]
                if empty_targets:
                    min_dist = float("inf")
                    best_neighbor = ag["pos"]
                    # Only consider staying or moving up
                    r, c = ag["pos"]
                    up_neighbor = (r - 1, c) if r > 0 else ag["pos"]
                    candidates = [ag["pos"]]
                    if (
                        up_neighbor not in occupied
                        and up_neighbor not in intended_moves.values()
                    ):
                        candidates.append(up_neighbor)
                    for n in candidates:
                        dist = min(
                            abs(n[0] - t[0]) + abs(n[1] - t[1]) for t in empty_targets
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = n
                    intended_moves[ag["id"]] = best_neighbor
                else:
                    intended_moves[ag["id"]] = ag["pos"]

        # Step 3: Resolve conflicts (if two agents want the same cell, both stay)
        move_counts = {}
        for move in intended_moves.values():
            move_counts[move] = move_counts.get(move, 0) + 1

        for ag in agents:
            desired = intended_moves[ag["id"]]
            if move_counts[desired] > 1:
                intended_moves[ag["id"]] = ag["pos"]

        # Step 4: Update agent positions and settled state
        for ag in agents:
            ag["pos"] = intended_moves[ag["id"]]
            if ag["pos"] in targets:
                ag["settled"] = True

        # Record step
        simulation_steps.append(
            {
                "step": step_counter + 1,
                "agents": [
                    {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]}
                    for ag in agents
                ],
                "paths": {},
            }
        )

        step_counter += 1

        # Stop if all targets are filled by agents
        if all(ag["settled"] for ag in agents) and len(
            {ag["pos"] for ag in agents if ag["settled"]}
        ) == len(targets):
            break

    return simulation_steps


# ======================================================
# Minimax
# ======================================================


def move_agents_minimax(grid, agent_positions, target_positions, obstacle_positions):
    """
    Assign agents to targets to minimize the maximum distance any agent must travel.
    Then move agents step by step toward their assigned targets.
    """
    import numpy as np

    # Step 1: Assign agents to targets to minimize the maximum distance
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
                cost_matrix[i, j] = 999999

    # Hungarian minimizes total cost, not max; so we brute-force for small n
    from itertools import permutations

    min_max_dist = float("inf")
    best_assignment = None
    for perm in permutations(range(n_targets), min(n_agents, n_targets)):
        max_dist = 0
        for i, j in enumerate(perm):
            max_dist = max(max_dist, cost_matrix[i, j])
        if max_dist < min_max_dist:
            min_max_dist = max_dist
            best_assignment = perm

    assignments = []
    if best_assignment:
        for i, j in enumerate(best_assignment):
            assignments.append((agent_positions[i], target_positions[j]))

    # Step 2: Move agents step by step toward their assigned targets
    agents = []
    for idx, (agent, target) in enumerate(assignments):
        agents.append({"id": idx, "pos": agent, "target": target})

    # Track any agents without targets
    unassigned_agents = [
        a for a in agent_positions if a not in [ag["pos"] for ag in agents]
    ]
    for agent in unassigned_agents:
        agent_idx = agent_positions.index(agent)
        agents.append({"id": agent_idx, "pos": agent, "target": None})

    simulation_steps = []
    initial_agents = [
        {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents
    ]
    simulation_steps.append({"step": 0, "agents": initial_agents, "paths": {}})

    step_counter = 1
    max_steps = 100
    while step_counter < max_steps:
        all_reached = True
        move_dict = {}
        conflict_positions = set()
        agent_positions_set = set(ag["pos"] for ag in agents)

        for ag in agents:
            current = ag["pos"]
            target = ag["target"]
            if target is None or current == target:
                move_dict[current] = current
                conflict_positions.add(current)
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
                    conflict_positions.add(current)
            else:
                move_dict[current] = current
                conflict_positions.add(current)

        # Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                final_moves[old_p] = old_p
                final_moves[new_p] = new_p
            else:
                final_moves[old_p] = new_p

        updated_agents = []
        current_positions_next_step = set()
        for ag in agents:
            old_pos = ag["pos"]
            new_pos = final_moves.get(old_pos, old_pos)
            if new_pos in current_positions_next_step or new_pos in obstacle_positions:
                new_pos = old_pos
            current_positions_next_step.add(new_pos)
            updated_agents.append(
                {"id": ag["id"], "pos": new_pos, "target": ag["target"]}
            )

        agents = updated_agents
        step_agents = [
            {"id": ag["id"] + 1, "x": ag["pos"][0], "y": ag["pos"][1]} for ag in agents
        ]
        simulation_steps.append(
            {"step": step_counter, "agents": step_agents, "paths": {}}
        )
        step_counter += 1
        if all_reached:
            break

    return simulation_steps


# ======================================================
# Expectimax
# ======================================================


def move_agents_expectimax(
    grid,
    agent_positions,
    target_positions,
    obstacle_positions,
    max_depth=2,
    obstacle_prob=0.2,
):
    """
    Expectimax-based agent movement with random obstacle appearance/disappearance.
    Only supports a single agent for simplicity.
    Returns obstacle positions at each step for visualization.
    """
    import copy

    agent_pos = agent_positions[0]
    target_pos = target_positions[0]
    simulation_steps = []
    step_counter = 0
    max_steps = 100

    static_obstacles = set(obstacle_positions)  # <-- Always include these

    def evaluation(pos):
        return -manhattan_dist(pos, target_pos)

    def get_neighbors(pos):
        r, c = pos
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                neighbors.append((nr, nc))
        return neighbors

    def expectimax(pos, dynamic_obstacles, depth, is_chance):
        # Always combine static and dynamic obstacles
        obstacles = static_obstacles | dynamic_obstacles
        if depth == 0 or pos == target_pos:
            return evaluation(pos)
        if is_chance:
            neighbors = get_neighbors(pos)
            expected = 0
            for n in neighbors:
                new_dynamic = set(dynamic_obstacles)
                if n in dynamic_obstacles:
                    new_dynamic.remove(n)
                else:
                    new_dynamic.add(n)
                expected += obstacle_prob * expectimax(
                    pos, new_dynamic, depth - 1, False
                )
            expected += (1 - obstacle_prob * len(neighbors)) * expectimax(
                pos, dynamic_obstacles, depth - 1, False
            )
            return expected
        else:
            best = float("-inf")
            for n in get_neighbors(pos) + [pos]:
                if n not in obstacles:
                    val = expectimax(n, dynamic_obstacles, depth - 1, True)
                    if val > best:
                        best = val
            return best

    current_pos = agent_pos
    current_dynamic_obstacles = set()  # Only random obstacles here
    simulation_steps.append(
        {
            "step": 0,
            "agents": [{"id": 1, "x": current_pos[0], "y": current_pos[1]}],
            "obstacles": list(static_obstacles | current_dynamic_obstacles),
            "paths": {},
        }
    )
    while step_counter < max_steps and current_pos != target_pos:
        obstacles = static_obstacles | current_dynamic_obstacles
        best_val = float("-inf")
        best_move = current_pos
        for n in get_neighbors(current_pos) + [current_pos]:
            if n not in obstacles:
                val = expectimax(n, current_dynamic_obstacles, max_depth, True)
                if val > best_val:
                    best_val = val
                    best_move = n
        current_pos = best_move

        # Simulate random obstacle appearance/disappearance (only for dynamic obstacles)
        for n in get_neighbors(current_pos):
            if n in static_obstacles:
                continue  # Never change static obstacles
            if random.random() < obstacle_prob:
                if n in current_dynamic_obstacles:
                    current_dynamic_obstacles.remove(n)
                else:
                    current_dynamic_obstacles.add(n)

        simulation_steps.append(
            {
                "step": step_counter + 1,
                "agents": [{"id": 1, "x": current_pos[0], "y": current_pos[1]}],
                "obstacles": list(static_obstacles | current_dynamic_obstacles),
                "paths": {},
            }
        )
        step_counter += 1
        if current_pos == target_pos:
            break

    return simulation_steps


# ======================================================
# minima with adversary inside its
# ======================================================


def move_agents_minimax_with_adversary(
    grid,  # Pass the grid to get its shape
    agent_positions,
    target_positions,
    obstacle_positions,
    enemy_position,
    max_depth=3,
    history_length=4,  # How many past game states to remember
):
    if not agent_positions or not target_positions:
        return [
            {
                "step": 0,
                "agents": [],
                "enemy": {},
                "obstacles": list(obstacle_positions),
                "paths": {},
            }
        ]

    agent_pos = agent_positions[0]  # Assuming single agent
    target_pos = target_positions[0]  # Assuming single target for this agent
    grid_shape = grid.shape
    simulation_steps = []
    step_counter = 0
    max_steps = 100  # Prevent infinite loops in edge cases

    static_obstacles = set(obstacle_positions)

    # --- Game State History ---
    # Use a deque to store the last N (agent_pos, enemy_pos) tuples
    game_state_history = deque(maxlen=history_length)

    # --- Evaluation Function (Tuned) ---
    def evaluation(agent, enemy, steps, target):
        # Strong reward/penalty for terminal states
        if agent == enemy:
            return -10000  # Very bad if caught
        if agent == target:
            # Reward reaching target, slightly penalize taking more steps
            return 10000 - steps * 5

        # Heuristic: Distance to target is primary, distance from enemy secondary
        target_dist = manhattan_dist(agent, target)
        enemy_dist = manhattan_dist(agent, enemy)

        # Ensure enemy_dist has less impact than target_dist
        # Give a slight bonus for being further from the enemy, but prioritize target
        # Penalize steps taken
        score = -target_dist * 10 + enemy_dist * 2 - steps
        return score

    # --- Minimax Implementation ---
    # visited_states_search: Prevents cycles *within* a single minimax call for a turn
    visited_states_search = set()

    def minimax(agent, enemy, depth, maximizing, alpha, beta, steps, current_target):
        # Include depth and maximizing player in state to avoid issues in search tree
        state = (agent, enemy, depth, maximizing)
        if state in visited_states_search:
            return -9999  # Penalize cycles within the search heavily

        is_terminal = agent == current_target or agent == enemy
        if depth == 0 or is_terminal:
            return evaluation(agent, enemy, steps, current_target)

        visited_states_search.add(state)
        possible_agent_moves = get_neighbors(agent, grid_shape) + [agent]
        possible_enemy_moves = get_neighbors(enemy, grid_shape) + [enemy]

        if maximizing:  # Agent's turn within the simulation
            max_eval = float("-inf")
            for next_agent in possible_agent_moves:
                # Agent cannot move into static obstacles or the enemy's current spot
                if next_agent not in static_obstacles and next_agent != enemy:
                    eval_score = minimax(
                        next_agent,
                        enemy,
                        depth - 1,
                        False,
                        alpha,
                        beta,
                        steps + 1,
                        current_target,
                    )
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cut-off
            visited_states_search.remove(state)
            return max_eval
        else:  # Enemy's turn within the simulation
            min_eval = float("inf")
            for next_enemy in possible_enemy_moves:
                # Enemy cannot move into static obstacles or the agent's current spot
                if next_enemy not in static_obstacles and next_enemy != agent:
                    eval_score = minimax(
                        agent,
                        next_enemy,
                        depth - 1,
                        True,
                        alpha,
                        beta,
                        steps + 1,
                        current_target,
                    )
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cut-off
            visited_states_search.remove(state)
            return min_eval

    # --- Main Simulation Loop ---
    current_agent = agent_pos
    current_enemy = enemy_position
    game_state_history.append((current_agent, current_enemy))  # Initial state

    simulation_steps.append(
        {
            "step": 0,
            "agents": [{"id": 1, "x": current_agent[0], "y": current_agent[1]}],
            "enemy": {"x": current_enemy[0], "y": current_enemy[1]},
            "obstacles": list(static_obstacles),
            "paths": {},
        }
    )

    while (
        step_counter < max_steps
        and current_agent != target_pos
        and current_agent != current_enemy
    ):
        step_counter += 1

        # == Agent's Turn (Maximizer) ==
        best_agent_val = float("-inf")
        best_agent_move = current_agent  # Default to staying put if no better move
        possible_agent_moves = get_neighbors(current_agent, grid_shape) + [
            current_agent
        ]
        random.shuffle(possible_agent_moves)  # Randomize to break ties

        for next_agent_move in possible_agent_moves:
            # Basic Validity Check
            if (
                next_agent_move not in static_obstacles
                and next_agent_move != current_enemy
                and (
                    next_agent_move == target_pos  # Allow moving to target
                    or not is_adjacent(
                        next_agent_move, current_enemy
                    )  # Otherwise, must not be adjacent
                )
            ):
                # Check History: Penalize returning to recent game states
                potential_state = (next_agent_move, current_enemy)
                history_penalty = 0
                if potential_state in game_state_history:
                    history_penalty = 500  # Apply a penalty for repeating

                # Clear visited set for this specific move's search
                visited_states_search.clear()
                # Evaluate the move using minimax (Enemy will respond optimally)
                val = minimax(
                    next_agent_move,
                    current_enemy,
                    max_depth,
                    False,
                    float("-inf"),
                    float("inf"),
                    step_counter,
                    target_pos,
                )
                val -= history_penalty  # Apply history penalty AFTER evaluation

                if val > best_agent_val:
                    best_agent_val = val
                    best_agent_move = next_agent_move

        current_agent = best_agent_move  # Update agent position

        # Check if agent reached target or was caught after its move
        if current_agent == target_pos or current_agent == current_enemy:
            # Record this final state and break
            game_state_history.append((current_agent, current_enemy))
            simulation_steps.append(
                {
                    "step": step_counter,
                    "agents": [{"id": 1, "x": current_agent[0], "y": current_agent[1]}],
                    "enemy": {"x": current_enemy[0], "y": current_enemy[1]},
                    "obstacles": list(static_obstacles),
                    "paths": {},
                }
            )
            break

        # == Enemy's Turn (Minimizer) ==
        best_enemy_val = float("inf")
        best_enemy_move = current_enemy  # Default to staying put
        possible_enemy_moves = get_neighbors(current_enemy, grid_shape) + [
            current_enemy
        ]
        random.shuffle(possible_enemy_moves)  # Randomize to break ties

        for next_enemy_move in possible_enemy_moves:
            # Basic Validity Check
            if (
                next_enemy_move not in static_obstacles
                and next_enemy_move != current_agent
                and (
                    next_enemy_move != target_pos or next_enemy_move == current_agent
                )  # Only allow if catching agent
            ):
                # Check History: Penalize returning to recent game states
                potential_state = (current_agent, next_enemy_move)
                history_penalty = 0
                # Note: Enemy "wants" low scores (bad for agent). Repeating a state might be
                # tactically good for the enemy if it traps the agent, but generally bad strategy.
                # We penalize it by ADDING to the score (making it seem worse for the enemy/better for agent)
                if potential_state in game_state_history:
                    history_penalty = (
                        500  # Make repeating seem less attractive for the enemy
                    )

                # Clear visited set for this specific move's search
                visited_states_search.clear()
                # Evaluate the move using minimax (Agent will respond optimally)
                val = minimax(
                    current_agent,
                    next_enemy_move,
                    max_depth,
                    True,
                    float("-inf"),
                    float("inf"),
                    step_counter,
                    target_pos,
                )
                val += history_penalty  # Apply history penalty

                if val < best_enemy_val:
                    best_enemy_val = val
                    best_enemy_move = next_enemy_move

        current_enemy = best_enemy_move  # Update enemy position

        # --- Record State and Update History ---
        game_state_history.append((current_agent, current_enemy))
        simulation_steps.append(
            {
                "step": step_counter,
                "agents": [{"id": 1, "x": current_agent[0], "y": current_agent[1]}],
                "enemy": {"x": current_enemy[0], "y": current_enemy[1]},
                "obstacles": list(static_obstacles),
                "paths": {},
            }
        )

        # Check if enemy caught agent after its move
        if current_agent == current_enemy:
            break  # End simulation

    # Final message if loop finishes
    if step_counter >= max_steps:
        print("Minimax simulation reached max steps.")
    elif current_agent == target_pos:
        print("Minimax simulation: Agent reached target.")
    elif current_agent == current_enemy:
        print("Minimax simulation: Agent caught.")

    return simulation_steps


# ======================================================
# agents greed and stochastic
# ======================================================

def move_agents_greedy(grid, agent_positions, target_positions, obstacle_positions, shape_type=None):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    rows, cols = grid.shape
    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    n = min(n_agents, n_targets)
    agent_positions = agent_positions[:n]
    target_positions = target_positions[:n]

    def manhattan_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def prioritize_square_inner(target_positions, rows, cols):
        center = (rows // 2, cols // 2)
        return sorted(target_positions, key=lambda pos: manhattan_dist(pos, center))

    if shape_type == "square":
        target_positions = prioritize_square_inner(target_positions, rows, cols)

    # Assign agents to targets ONCE at the start
    cost_matrix = np.zeros((n, n))
    for i, agent in enumerate(agent_positions):
        for j, target in enumerate(target_positions):
            cost_matrix[i, j] = manhattan_dist(agent, target)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_targets = [target_positions[j] for i, j in sorted(zip(row_ind, col_ind))]

    move_history = {}
    simulation_steps = []
    step_counter = 0
    max_steps = 500
    agent_positions = agent_positions.copy()

    # Initial state
    simulation_steps.append({
        "step": 0,
        "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
        "paths": {},
    })

    while step_counter < max_steps:
        all_reached = True
        any_moved = False

        # Each agent moves toward its assigned target
        proposed_moves = []
        for i, agent in enumerate(agent_positions):
            target = assigned_targets[i]
            if agent == target:
                proposed_moves.append(agent)
                continue
            all_reached = False
            prev_move = move_history.get(i)
            # Only consider up/down/left/right
            possible_moves = [
                (agent[0] + 1, agent[1]),  # Down
                (agent[0] - 1, agent[1]),  # Up
                (agent[0], agent[1] + 1),  # Right
                (agent[0], agent[1] - 1),  # Left
            ]
            valid_moves = []
            for move in possible_moves:
                r, c = move
                if (
                    0 <= r < rows and 0 <= c < cols and
                    move not in agent_positions and
                    move not in obstacle_positions
                ):
                    score = manhattan_dist(move, target)
                    if prev_move and move == (agent[0] - prev_move[0], agent[1] - prev_move[1]):
                        score += 1
                    valid_moves.append((move, score))
            if valid_moves:
                best_move = min(valid_moves, key=lambda x: x[1])[0]
                proposed_moves.append(best_move)
            else:
                proposed_moves.append(agent)

        # Resolve direct collisions: if two agents want the same cell, both stay
        new_positions = agent_positions.copy()
        move_counts = {}
        for move in proposed_moves:
            move_counts[move] = move_counts.get(move, 0) + 1

        for i, move in enumerate(proposed_moves):
            if move_counts[move] == 1:
                new_positions[i] = move
                if move != agent_positions[i]:
                    move_history[i] = (move[0] - agent_positions[i][0], move[1] - agent_positions[i][1])
                    any_moved = True
            else:
                new_positions[i] = agent_positions[i]  # Stay if conflict

        agent_positions = new_positions

        # Record step
        simulation_steps.append({
            "step": step_counter + 1,
            "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
            "paths": {},
        })

        step_counter += 1
        if all_reached or not any_moved:
            break

    return simulation_steps

def move_agents_stochastic(grid, agent_positions, target_positions, obstacle_positions, shape_type=None):
    """
    Expectimax-inspired agent movement with soft lock: agents at their targets move aside (to any free cell) if needed.
    If agents are stuck for too long, reshuffle all agents to random empty cells to break deadlock.
    """
    import numpy as np
    import random
    from scipy.optimize import linear_sum_assignment
    from collections import deque

    rows, cols = grid.shape
    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    n = min(n_agents, n_targets)
    agent_positions = agent_positions[:n]
    target_positions = target_positions[:n]

    def manhattan_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid_move(position, agent_positions):
        r, c = position
        return (0 <= r < rows and 0 <= c < cols and
                position not in agent_positions and
                position not in obstacle_positions)

    def bfs_4dir(start, goal, grid, agents, obstacles):
        queue = deque([(start, [])])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in agents and
                    (nr, nc) not in obstacles and
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def bfs_to_any_free(start, grid, agents, obstacles):
        """Find the shortest path from start to any free cell in the grid."""
        queue = deque([(start, [])])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            (r, c), path = queue.popleft()
            if (grid[r, c] == 0 and (r, c) not in agents and (r, c) not in obstacles and (r, c) != start):
                return path
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in agents and
                    (nr, nc) not in obstacles and
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def assign_agents_to_targets(agent_positions, target_positions):
        n_agents = len(agent_positions)
        n_targets = len(target_positions)
        n = min(n_agents, n_targets)
        if n == 0:
            return [None] * n_agents
        cost_matrix = np.zeros((n_agents, n_targets))
        for i, agent in enumerate(agent_positions):
            for j, target in enumerate(target_positions):
                cost_matrix[i, j] = manhattan_dist(agent, target)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_targets = [None] * n_agents
        for agent_idx, target_idx in zip(row_ind[:n], col_ind[:n]):
            assigned_targets[agent_idx] = target_positions[target_idx]
        return assigned_targets

    simulation_steps = []
    step_counter = 0
    max_steps = 1000
    move_history = {}
    stuck_counter = 0
    stuck_threshold = 50  # Number of steps to wait before reshuffling

    # Initial state
    simulation_steps.append({
        "step": 0,
        "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
        "paths": {},
    })

    while step_counter < max_steps:
        all_reached = True
        any_moved = False
        agent_positions_set = set(agent_positions)
        assigned_targets = assign_agents_to_targets(agent_positions, target_positions)
        unsettled_indices = [i for i, pos in enumerate(agent_positions) if pos != assigned_targets[i]]
        settled_indices = [i for i, pos in enumerate(agent_positions) if pos == assigned_targets[i]]

        # 1. Move unsettled agents first (using BFS)
        proposed_moves = [None] * len(agent_positions)
        for i in unsettled_indices:
            agent = agent_positions[i]
            target = assigned_targets[i]
            other_agents = agent_positions_set - {agent}
            path = bfs_4dir(agent, target, grid, other_agents, set(obstacle_positions))
            if path:
                next_step = path[0]
                proposed_moves[i] = next_step
                all_reached = False
            else:
                proposed_moves[i] = agent
                all_reached = False

        # 2. Settled agents: stay unless someone else wants their spot, then try to move aside (BFS to any free cell)
        for i in settled_indices:
            agent = agent_positions[i]
            if agent not in [proposed_moves[j] for j in unsettled_indices]:
                proposed_moves[i] = agent
            else:
                other_agents = agent_positions_set - {agent}
                path = bfs_to_any_free(agent, grid, other_agents, set(obstacle_positions))
                if path:
                    proposed_moves[i] = path[0]
                else:
                    proposed_moves[i] = agent

        # 3. Priority tie-breaker for collisions
        new_positions = agent_positions.copy()
        move_counts = {}
        for move in proposed_moves:
            move_counts[move] = move_counts.get(move, 0) + 1
        conflict_cells = {move for move, count in move_counts.items() if count > 1}
        for i, move in enumerate(proposed_moves):
            if move in conflict_cells:
                first_agent = min(j for j, m in enumerate(proposed_moves) if m == move)
                if i == first_agent:
                    new_positions[i] = move
                    if move != agent_positions[i]:
                        any_moved = True
                else:
                    new_positions[i] = agent_positions[i]
            else:
                new_positions[i] = move
                if move != agent_positions[i]:
                    any_moved = True

        if agent_positions == new_positions:
            stuck_counter += 1
        else:
            stuck_counter = 0

        agent_positions = new_positions

        # If stuck for too long, reshuffle all agents randomly to break deadlock
        if stuck_counter >= stuck_threshold:
            empty_cells = [(r, c) for r in range(rows) for c in range(cols)
                           if grid[r, c] == 0 and (r, c) not in obstacle_positions]
            random.shuffle(empty_cells)
            for i in range(len(agent_positions)):
                if i < len(empty_cells):
                    agent_positions[i] = empty_cells[i]
            stuck_counter = 0  # Reset counter after reshuffle

        simulation_steps.append({
            "step": step_counter + 1,
            "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
            "paths": {},
        })

        step_counter += 1
        if all_reached or not any_moved:
            break

    return simulation_steps

def move_agents_greedy_8dir(grid, agent_positions, target_positions, obstacle_positions, shape_type=None):
    """
    At every step, assign each agent to the closest target (Hungarian), and move using 8 directions with BFS.
    Agents at their targets will move aside (using BFS to any available cell) if another agent needs their spot.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import deque

    rows, cols = grid.shape
    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    n = min(n_agents, n_targets)
    agent_positions = agent_positions[:n]
    target_positions = target_positions[:n]

    def manhattan_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs_8dir(start, goal, grid, agents, obstacles):
        queue = deque([(start, [])])
        visited = {start}
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in agents and
                    (nr, nc) not in obstacles and
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def bfs_to_any_free(start, grid, agents, obstacles):
        """Find the shortest path from start to any free cell."""
        queue = deque([(start, [])])
        visited = {start}
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        while queue:
            (r, c), path = queue.popleft()
            if (grid[r, c] == 0 and (r, c) not in agents and (r, c) not in obstacles and (r, c) != start):
                return path
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in agents and
                    (nr, nc) not in obstacles and
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    simulation_steps = []
    step_counter = 0
    max_steps = 1000
    agent_positions = agent_positions.copy()

    # Initial state
    simulation_steps.append({
        "step": 0,
        "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
        "paths": {},
    })

    while step_counter < max_steps:
        all_reached = True
        any_moved = False

        # Reassign agents to targets at every step (Hungarian)
        n = min(len(agent_positions), len(target_positions))
        if n == 0:
            break
        cost_matrix = np.zeros((n, n))
        for i, agent in enumerate(agent_positions[:n]):
            for j, target in enumerate(target_positions[:n]):
                cost_matrix[i, j] = manhattan_dist(agent, target)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_targets = [target_positions[j] for i, j in sorted(zip(row_ind, col_ind))]

        agent_positions_set = set(agent_positions)
        unsettled_indices = [i for i, pos in enumerate(agent_positions) if pos != assigned_targets[i]]
        settled_indices = [i for i, pos in enumerate(agent_positions) if pos == assigned_targets[i]]

        # 1. Move unsettled agents first (using BFS)
        proposed_moves = [None] * len(agent_positions)
        for i in unsettled_indices:
            agent = agent_positions[i]
            target = assigned_targets[i]
            other_agents = agent_positions_set - {agent}
            path = bfs_8dir(agent, target, grid, other_agents, set(obstacle_positions))
            if path:
                next_step = path[0]
                proposed_moves[i] = next_step
                all_reached = False
            else:
                proposed_moves[i] = agent
                all_reached = False

        # 2. Settled agents: stay unless someone else wants their spot, then try to move aside (BFS to any free cell)
        for i in settled_indices:
            agent = agent_positions[i]
            # If no unsettled agent wants this spot, stay
            if agent not in [proposed_moves[j] for j in unsettled_indices]:
                proposed_moves[i] = agent
            else:
                # Try to move to any available cell (BFS)
                other_agents = agent_positions_set - {agent}
                path = bfs_to_any_free(agent, grid, other_agents, set(obstacle_positions))
                if path:
                    proposed_moves[i] = path[0]
                else:
                    proposed_moves[i] = agent  # Stay if can't move aside

        # 3. Priority tie-breaker for collisions
        new_positions = agent_positions.copy()
        move_counts = {}
        for move in proposed_moves:
            move_counts[move] = move_counts.get(move, 0) + 1
        conflict_cells = {move for move, count in move_counts.items() if count > 1}
        for i, move in enumerate(proposed_moves):
            if move in conflict_cells:
                first_agent = min(j for j, m in enumerate(proposed_moves) if m == move)
                if i == first_agent:
                    new_positions[i] = move
                    if move != agent_positions[i]:
                        any_moved = True
                else:
                    new_positions[i] = agent_positions[i]
            else:
                new_positions[i] = move
                if move != agent_positions[i]:
                    any_moved = True

        agent_positions = new_positions

        # Record step
        simulation_steps.append({
            "step": step_counter + 1,
            "agents": [{"id": i + 1, "x": pos[0], "y": pos[1]} for i, pos in enumerate(agent_positions)],
            "paths": {},
        })

        step_counter += 1
        if all_reached or not any_moved:
            break

    return simulation_steps
# ======================================================
# q-learning RL implementation based
# ======================================================


def move_agents_q_learning(
    grid,
    agent_positions,
    target_positions,
    obstacle_positions,
    episodes=500,
    alpha=0.9,
    gamma=0.9,
    epsilon=0.3,
):
    """
    Q-learning for a single agent in the grid.
    Returns all episodes (learning phase), exploitation phase, and Q-values.
    """
    rows, cols = grid.shape
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    action_indices = list(range(len(actions)))
    target = target_positions[0]
    start = agent_positions[0]

    Q = np.zeros((rows, cols, len(actions)))
    rewards = np.full((rows, cols), -0.01)
    for r, c in obstacle_positions:
        rewards[r, c] = -1
    rewards[target[0], target[1]] = 1

    # --- Collect all episodes ---
    all_episodes = []
    for ep in range(episodes):
        state = start
        episode_steps = []
        for step in range(100):
            episode_steps.append(
                {
                    "step": step,
                    "agents": [{"id": 1, "x": state[0], "y": state[1]}],
                    "phase": "learning",
                    "episode": ep,
                }
            )
            if np.random.rand() < epsilon:
                a = np.random.choice(action_indices)
            else:
                a = np.argmax(Q[state[0], state[1]])
            dr, dc = actions[a]
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacle_positions:
                next_state = (nr, nc)
            else:
                next_state = state
            reward = rewards[next_state[0], next_state[1]]
            Q[state[0], state[1], a] = Q[state[0], state[1], a] + alpha * (
                reward
                + gamma * np.max(Q[next_state[0], next_state[1]])
                - Q[state[0], state[1], a]
            )
            state = next_state
            if state == target or reward == -1:
                break
        all_episodes.append(episode_steps)

    # --- After training, generate the optimal path (exploitation phase) ---
    state = start
    path = [state]
    for _ in range(100):
        a = np.argmax(Q[state[0], state[1]])
        dr, dc = actions[a]
        nr, nc = state[0] + dr, state[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacle_positions:
            next_state = (nr, nc)
        else:
            next_state = state
        path.append(next_state)
        state = next_state
        if state == target:
            break

    exploitation_steps = []
    for i, pos in enumerate(path):
        exploitation_steps.append(
            {
                "step": i,
                "agents": [{"id": 1, "x": pos[0], "y": pos[1]}],
                "phase": "exploitation",
            }
        )

    max_q = np.max(Q, axis=2)
    simulation_result = {
        "episodes": all_episodes,  # All learning episodes
        "exploitation": exploitation_steps,  # Final exploitation phase
        "q_values": max_q.tolist(),
    }
    return simulation_result


# ======================================================
# deep learning
# ======================================================


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


def move_agents_deep_qlearning(
    grid, agent_positions, target_positions, obstacle_positions, episodes=200, pretrained_model=None
):
    print("Deep Q-Learning: Starting training...")
    state_dim = 4  # (agent_row, agent_col, target_row, target_col)
    action_dim = 4  # up, down, left, right
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    device = torch.device("cpu")

    model = DQN(state_dim, action_dim).to(device)
    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded pretrained model weights.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    gamma = 0.95
    epsilon = 0.2

    target = target_positions[0]
    start = agent_positions[0]

    memory = []

    if episodes > 0:
        for ep in range(episodes):
            agent_pos = np.array(start)
            for t in range(100):
                state = np.array([agent_pos[0], agent_pos[1], target[0], target[1]])
                state_tensor = torch.FloatTensor(state).to(device)
                if random.random() < epsilon:
                    a = random.randint(0, action_dim - 1)
                else:
                    with torch.no_grad():
                        qvals = model(state_tensor)
                        a = torch.argmax(qvals).item()
                dr, dc = actions[a]
                next_agent_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
                if (
                    0 <= next_agent_pos[0] < grid.shape[0]
                    and 0 <= next_agent_pos[1] < grid.shape[1]
                    and (next_agent_pos not in obstacle_positions)
                ):
                    reward = 1 if next_agent_pos == target else -0.01
                else:
                    next_agent_pos = tuple(agent_pos)
                    reward = -1
                next_state = np.array(
                    [next_agent_pos[0], next_agent_pos[1], target[0], target[1]]
                )
                memory.append((state, a, reward, next_state))
                if len(memory) > 32:
                    batch = random.sample(memory, 32)
                    states, actions_, rewards, next_states = zip(*batch)
                    states = torch.FloatTensor(np.array(states)).to(device)
                    actions_ = torch.LongTensor(actions_).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(device)
                    qvals = model(states).gather(1, actions_.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_qvals = model(next_states).max(1)[0]
                    targets = rewards + gamma * next_qvals
                    loss = loss_fn(qvals, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                agent_pos = np.array(next_agent_pos)
                if tuple(agent_pos) == target or reward == -1:
                    break
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} completed.")

    print("Deep Q-Learning: Training complete. Starting exploitation...")

    # Exploitation: follow learned policy
    agent_pos = np.array(start)
    path = [tuple(agent_pos)]
    for _ in range(100):
        state = np.array([agent_pos[0], agent_pos[1], target[0], target[1]])
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            qvals = model(state_tensor)
            a = torch.argmax(qvals).item()
        dr, dc = actions[a]
        next_agent_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
        if (
            0 <= next_agent_pos[0] < grid.shape[0]
            and 0 <= next_agent_pos[1] < grid.shape[1]
            and (next_agent_pos not in obstacle_positions)
        ):
            agent_pos = np.array(next_agent_pos)
        else:
            break
        path.append(tuple(agent_pos))
        if tuple(agent_pos) == target:
            break

    print(f"Deep Q-Learning: Exploitation path length: {len(path)}")

    exploitation_steps = []
    for i, pos in enumerate(path):
        exploitation_steps.append(
            {
                "step": int(i),
                "agents": [{"id": 1, "x": int(pos[0]), "y": int(pos[1])}],
                "phase": "deep-exploitation",
            }
        )
    print("Deep Q-Learning: Returning exploitation steps to frontend.")
    # Save model state for future use
    return {"exploitation": exploitation_steps, "model_state": model.state_dict()}

# ======================================================
# API Endpoints
# ======================================================

# Update to the run_simulation function in app.py to modify agent placement


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    """API endpoint to receive grid data and return simulation steps."""
    try:
        import pickle
        import os

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

        # Set the target and agent start randomly ONLY for deep learning
        if algorithm == "deep":
            # Find all empty cells
            empty_cells = [
                (r, c)
                for r in range(grid.shape[0])
                for c in range(grid.shape[1])
                if grid[r, c] == 0
            ]
            if len(empty_cells) < 2:
                return jsonify({"error": "Not enough empty cells for agent and target."}), 400
            start_pos = random.choice(empty_cells)
            target_pos = random.choice([cell for cell in empty_cells if cell != start_pos])
            agent_positions = [start_pos]
            target_positions = [target_pos]

            # Model save/load logic
            model_dir = "trained_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(
                model_dir, f"dqn_{start_pos[0]}_{start_pos[1]}_{target_pos[0]}_{target_pos[1]}.pkl"
            )

            pretrained_model = None
            if os.path.exists(model_path):
                print(f"Loading pretrained model from {model_path}")
                with open(model_path, "rb") as f:
                    state_dict = pickle.load(f)
                pretrained_model = DQN(4, 4)
                pretrained_model.load_state_dict(state_dict)

            result = move_agents_deep_qlearning(
                grid, agent_positions, target_positions, obstacle_positions,
                pretrained_model=pretrained_model if pretrained_model else None
            )

            # Save model if newly trained
            if not pretrained_model:
                with open(model_path, "wb") as f:
                    pickle.dump(result["model_state"], f)

            return jsonify({
                "steps": result["exploitation"],
                "start": {"x": start_pos[0], "y": start_pos[1]},
                "target": {"x": target_pos[0], "y": target_pos[1]}
            })

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
        elif algorithm == "minimax":
            simulation_steps = move_agents_minimax(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "stochastic":
            simulation_steps = move_agents_stochastic(
        grid, agent_positions, target_positions, obstacle_positions, config_data.get("shapeType")
    )
        elif algorithm == "astar":
             simulation_steps = move_agents_astar(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "greedy-custom":
              simulation_steps = move_agents_greedy(
                grid, agent_positions, target_positions, obstacle_positions, config_data.get("shapeType")
    )
        elif algorithm == "greedy-8dir":
               simulation_steps = move_agents_greedy_8dir(
               grid, agent_positions, target_positions, obstacle_positions, config_data.get("shapeType")
    )
        elif algorithm == "cellular-automata":
            simulation_steps = move_agents_cellular_automata(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "expectimax":
            simulation_steps = move_agents_expectimax(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == "minimax-adv":
            # Example: place enemy in the center of the grid
            grid_height = grid.shape[0]
            grid_width = grid.shape[1]
            enemy_position = (grid_height // 2, grid_width // 2)
            # Make sure enemy is not on an obstacle or agent
            if (
                enemy_position in obstacle_positions
                or enemy_position in agent_positions
            ):
                # Pick first empty cell as fallback
                for r in range(grid_height):
                    for c in range(grid_width):
                        if (r, c) not in obstacle_positions and (
                            r,
                            c,
                        ) not in agent_positions:
                            enemy_position = (r, c)
                            break
                    else:
                        continue
                    break
            simulation_steps = move_agents_minimax_with_adversary(
                grid,
                agent_positions,
                target_positions,
                obstacle_positions,
                enemy_position,
            )
        elif algorithm == "qlearning":
            result = move_agents_q_learning(
                grid, agent_positions, target_positions, obstacle_positions
            )
            return jsonify(result)
        else:  # Default to inside-out
            simulation_steps = move_agents_inside_out(
                grid, agent_positions, target_positions, obstacle_positions
            )

        # Return result
        return jsonify({"steps": simulation_steps})

    except Exception as e:
        import traceback

        print(traceback.format_exc())
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
