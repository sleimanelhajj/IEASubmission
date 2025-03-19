# pathfinding_leader_follower.py
# Description:
# This file implements a leader-follower pathfinding strategy plus a partial
# parallel movement approach for multiple agents on a 2D grid.

import time
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm for optimal assignment
import streamlit as st
from grid_module import display_grid  # Custom text-based or emoji-based grid visualization

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def manhattan_dist(a, b):
    """
    Compute the Manhattan distance between two grid coordinates a and b.
    Manhattan distance = |x1 - x2| + |y1 - y2|.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_dynamic(start, goal, grid, agents):
    """
    Perform BFS to find a path from 'start' to 'goal' while treating 'agents' 
    as blocked cells (in addition to any obstacles in the grid).
    
    Arguments:
        start (tuple): The (row, col) of the agent's start position.
        goal  (tuple): The (row, col) we want to reach.
        grid  (2D array): The grid representation (used here primarily for shape).
        agents (set): Positions of other agents to treat as obstacles.

    Returns:
        A path (list of (row, col)) from start to goal, excluding start itself. 
        If no path is found, returns an empty list.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = set([start])  # Keep track of visited cells

    while queue:
        (r, c), path = queue.popleft()

        # Check if we've reached the goal
        if (r, c) == goal:
            return path

        # Explore the four possible moves (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            # Ensure it's a valid cell and not occupied by other agents
            if (
                0 <= nr < rows 
                and 0 <= nc < cols
                and (nr, nc) not in agents
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    # No path found
    return []

def hungarian_assignment(agent_positions, target_positions):
    """
    Assign each agent to a unique target using the Hungarian Algorithm for 
    minimal total cost based on Manhattan distance.

    Arguments:
        agent_positions (list): All agent positions.
        target_positions (list): All target positions.

    Returns:
        A list of (agent_pos, target_pos) pairs that optimally match 
        agents to targets with minimal total Manhattan distance.
    """
    # Sort to keep consistent ordering
    agent_positions = sorted(agent_positions)
    target_positions = sorted(target_positions)
    n_agents = len(agent_positions)
    n_targets = len(target_positions)
    size = max(n_agents, n_targets)

    # Build cost matrix; fill unmatched spots with a very large cost (999999)
    cost_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i < n_agents and j < n_targets:
                cost_matrix[i, j] = manhattan_dist(agent_positions[i], target_positions[j])
            else:
                cost_matrix[i, j] = 999999  # Large cost for placeholders

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Build the list of (agent, target) pairs
    assignments = []
    for k in range(size):
        if row_ind[k] < n_agents and col_ind[k] < n_targets:
            agent = agent_positions[row_ind[k]]
            target = target_positions[col_ind[k]]
            assignments.append((agent, target))
    return assignments

# ---------------------------------------------------------
# Leader-Follower + Partial Parallel Movement
# ---------------------------------------------------------
def move_agents_leader_follower(grid, agent_positions, target_positions, grid_placeholder, speed=0.3):
    """
    Main function that moves agents using a two-phase approach:
    1) Leader-Follower for the highest priority target.
    2) Parallel movement + Hungarian assignment for the rest of the agents/targets.

    Returns:
        Updated grid, and final agent positions after shape formation.
    """
    # Initialize performance tracking metrics
    start_time = time.time()
    metrics = {
        'agent_steps': [0] * len(agent_positions),
        'waiting_time': [0] * len(agent_positions),
        'manhattan_distances': [0] * len(agent_positions),
        'path_length': [0] * len(agent_positions),
        'phase_1_time': 0,
        'phase_2_time': 0
    }
    
    # Copy the initial agent positions so we can modify them
    current_positions = agent_positions.copy()

    # Ensure agent and target counts match by trimming if needed
    if len(current_positions) != len(target_positions):
        print("Warning: mismatch in agent/target count.")
        min_len = min(len(current_positions), len(target_positions))
        current_positions = current_positions[:min_len]
        target_positions = target_positions[:min_len]

    # ---------------------------------------------------
    # Step 1: Leader-Follower approach for the first target
    # ---------------------------------------------------
    phase_1_start = time.time()
    
    # Calculate a centroid of all current agent positions (used to decide priority)
    agent_centroid = (
        sum(pos[0] for pos in current_positions) / len(current_positions),
        sum(pos[1] for pos in current_positions) / len(current_positions)
    )

    # Count how many neighbors (up, down, left, right) each target has
    target_set = set(target_positions)
    def neighbor_count(pos):
        r, c = pos
        nb = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        return sum(1 for n in nb if n in target_set)

    neighbor_scores = {t: neighbor_count(t) for t in target_positions}

    # Create a priority function that ranks targets by distance from centroid 
    # plus a multiplier for neighbor_score.
    def priority(t):
        dist_component = manhattan_dist(t, agent_centroid)
        neighbor_component = neighbor_scores[t]
        return dist_component + neighbor_component * 3

    # Sort the targets from highest to lowest priority
    sorted_targets = sorted(target_positions, key=priority, reverse=True)

    # Leader is the first agent, leader target is highest priority target
    leader_pos = current_positions[0]
    leader_target = sorted_targets[0]
    
    # Calculate and store initial Manhattan distances
    for i, (agent, target) in enumerate(zip(current_positions, sorted_targets)):
        metrics['manhattan_distances'][i] = manhattan_dist(agent, target)

    # Initial visualization: mark the agents on the grid
    init_grid = np.zeros_like(grid)
    for (r, c) in current_positions:
        init_grid[r, c] = 1
    display_grid(init_grid, grid_placeholder)
    time.sleep(1)

    # BFS path for the leader to its target, ignoring other agents as obstacles
    leader_path = bfs_dynamic(leader_pos, leader_target, grid, set(current_positions[1:]))
    if not leader_path:
        print("No leader path found!")
        return grid, current_positions

    # Store expected path length for leader
    metrics['path_length'][0] = len(leader_path)
    
    # Combine start + BFS path so we can iterate from index 1 onwards
    full_leader_path = [leader_pos] + leader_path

    # Keep track of each agent's position over time
    position_history = [[] for _ in current_positions]
    for i, pos in enumerate(current_positions):
        position_history[i].append(pos)

    # Move the leader step by step
    for si in range(1, len(full_leader_path)):
        # Leader moves - count a step
        current_positions[0] = full_leader_path[si]
        position_history[0].append(current_positions[0])
        metrics['agent_steps'][0] += 1

        # Occupied set for this step (leader's new position first)
        occupied = {current_positions[0]}

        # Followers try to move one step to the position that the next agent was in previously
        for i in range(1, len(current_positions)):
            prev_ahead = position_history[i-1][-2]  # The prior agent's old position
            curr_pos = current_positions[i]

            # If the current follower isn't already at the old position of the agent ahead:
            if curr_pos != prev_ahead:
                mini_path = bfs_dynamic(curr_pos, prev_ahead, grid, occupied)
                if mini_path and mini_path[0] not in occupied:
                    # Move one step - count a step
                    current_positions[i] = mini_path[0]
                    occupied.add(mini_path[0])
                    metrics['agent_steps'][i] += 1
                else:
                    # If no path or blocked, stay put - count waiting
                    occupied.add(curr_pos)
                    metrics['waiting_time'][i] += 1
            else:
                # Already at the old position of the agent ahead
                occupied.add(curr_pos)

            # Update follower's history
            position_history[i].append(current_positions[i])

        # Visualize the step
        step_grid = np.zeros_like(grid)
        for (r, c) in current_positions:
            step_grid[r, c] = 1
        display_grid(step_grid, grid_placeholder)
        time.sleep(speed)

    metrics['phase_1_time'] = time.time() - phase_1_start
        
    # ---------------------------------------------------
    # Step 2: Parallel movement for remaining agents/targets
    # ---------------------------------------------------
    phase_2_start = time.time()
    
    filled_targets = {leader_target}   # The leader's target is already settled
    current_agent_positions = set(current_positions)
    followers = current_positions[1:]  # All except the leader
    remaining_targets = [t for t in sorted_targets[1:] if t not in filled_targets]

    # Build cost matrix for the Hungarian-like assignment
    # But we incorporate neighbor_scores to encourage agents to fill more connected targets first.
    cost_matrix = np.zeros((len(followers), len(remaining_targets)))
    for i, fpos in enumerate(followers):
        for j, tpos in enumerate(remaining_targets):
            base_dist = manhattan_dist(fpos, tpos)
            nb_score = neighbor_scores[tpos]
            # Subtract neighbor_scores so higher neighbor count is lower cost => more priority
            cost_matrix[i, j] = base_dist - (nb_score * 2)

    # Solve the assignment using the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build a list to store each follower's assigned target
    agents = []
    # Track assignments for metrics
    target_assignments = {}
    
    for x in range(len(row_ind)):
        if x < len(followers) and col_ind[x] < len(remaining_targets):
            follower_idx = row_ind[x] + 1  # +1 because we skipped the leader (idx 0)
            agent_pos = followers[row_ind[x]]
            target_pos = remaining_targets[col_ind[x]]
            
            agents.append({
                'id': follower_idx,
                'pos': agent_pos,
                'target': target_pos,
                'neighbors': neighbor_scores[target_pos]
            })
            
            # Store for metrics
            target_assignments[follower_idx] = target_pos
            # Calculate expected path length
            path_len = len(bfs_dynamic(agent_pos, target_pos, grid, set()))
            metrics['path_length'][follower_idx] = path_len

    # Sort the assigned agents so those with higher neighbor targets move first
    agents.sort(key=lambda a: a['neighbors'], reverse=True)

    # Perform up to max_iterations of partial parallel movement
    max_iterations = 200
    for iteration in range(max_iterations):
        # Check if all assigned agents have reached their targets
        if all(a['pos'] == a['target'] for a in agents):
            break

        # Calculate a proposed move for each agent in parallel
        moves = {}
        for ag in agents:
            if ag['pos'] == ag['target']:
                # Already at target
                moves[ag['pos']] = ag['pos']
                continue
                
            # BFS toward assigned target, ignoring current occupant positions
            others = current_agent_positions - {ag['pos']}
            path = bfs_dynamic(ag['pos'], ag['target'], grid, others)
            if path and path[0] not in moves.values():
                # Move one step
                moves[ag['pos']] = path[0]
            else:
                # Stay in place if no path or conflict
                moves[ag['pos']] = ag['pos']
                # Count as waiting
                metrics['waiting_time'][ag['id']] += 1

        # Resolve any collisions using neighbor priority
        next_positions = {}
        for src, dst in moves.items():
            if dst not in next_positions:
                # Destination unoccupied, proceed
                next_positions[dst] = src
            else:
                # Collision => compare neighbor priorities
                other_src = next_positions[dst]    # agent that already claimed 'dst'
                a1 = next(a for a in agents if a['pos'] == src)
                a2 = next(a for a in agents if a['pos'] == other_src)
                
                # The one with higher neighbor priority gets the spot
                if a1['neighbors'] > a2['neighbors']:
                    moves[other_src] = other_src  # Force the other agent to stay
                    next_positions[dst] = src
                    # Count waiting for agent that was forced to stay
                    metrics['waiting_time'][a2['id']] += 1
                else:
                    moves[src] = src
                    # Count waiting for agent that was forced to stay
                    metrics['waiting_time'][a1['id']] += 1

        # Apply moves
        new_positions = {current_positions[0]}  # Leader's position doesn't change now
        for ag in agents:
            old = ag['pos']
            new = moves[old]
            
            # Count a step if agent moved
            if old != new:
                metrics['agent_steps'][ag['id']] += 1
                
            ag['pos'] = new
            new_positions.add(new)
            # If we have arrived at the target
            if new == ag['target']:
                filled_targets.add(new)
        current_agent_positions = new_positions

        # Visualization of the new positions
        step_grid = np.zeros_like(grid)
        for p in current_agent_positions:
            r, c = p
            step_grid[r, c] = 1
        display_grid(step_grid, grid_placeholder)
        time.sleep(speed)

    metrics['phase_2_time'] = time.time() - phase_2_start
    
    # Calculate final metrics
    total_time = time.time() - start_time
    total_steps = sum(metrics['agent_steps'])
    avg_steps = total_steps / len(agent_positions) if agent_positions else 0
    total_waiting = sum(metrics['waiting_time'])
    
    # Calculate path efficiency
    path_efficiency = []
    for i in range(len(agent_positions)):
        if metrics['agent_steps'][i] > 0:
            # For leader-follower, efficiency is harder to define, since followers follow the leader
            # rather than going straight to targets. Use direct distance as rough comparison.
            efficiency = (metrics['manhattan_distances'][i] / metrics['agent_steps'][i]) * 100
            path_efficiency.append(efficiency)
        else:
            path_efficiency.append(0)
            
    avg_efficiency = sum(path_efficiency) / len(path_efficiency) if path_efficiency else 0
    
    # Display metrics in sidebar
    with st.sidebar:
        st.markdown("## 📊 Performance Metrics")
        
        # Overall statistics
        st.markdown("### Overall Performance")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Total Time", f"{total_time:.2f}s")
            st.metric("Phase 1 Time", f"{metrics['phase_1_time']:.2f}s")
        with cols[1]:
            st.metric("Total Steps", total_steps)
            st.metric("Phase 2 Time", f"{metrics['phase_2_time']:.2f}s")
        
        st.metric("Average Steps", f"{avg_steps:.1f}")
        st.metric("Total Wait Count", total_waiting)
        
        # Display efficiency warning
        st.info("""
        Note: For leader-follower algorithms, path efficiency can be over 100% for followers.
        This is normal as followers don't always take the shortest path to their targets.
        """)
        
        # Phase breakdown
        st.markdown("### Leader Performance")
        leader_stats = st.expander("Leader (Agent 1)")
        with leader_stats:
            st.markdown(f"**Steps**: {metrics['agent_steps'][0]}")
            st.markdown(f"**Wait count**: {metrics['waiting_time'][0]}")
            st.markdown(f"**Path length**: {metrics['path_length'][0]}")
            st.markdown(f"**Efficiency**: {path_efficiency[0]:.1f}%")
        
        # Follower performance
        st.markdown("### Follower Performance")
        for i in range(1, len(agent_positions)):
            agent_stats = st.expander(f"Follower {i+1}")
            with agent_stats:
                st.markdown(f"**Steps**: {metrics['agent_steps'][i]}")
                st.markdown(f"**Wait count**: {metrics['waiting_time'][i]}")
                if i in target_assignments:
                    st.markdown(f"**Target**: {target_assignments[i]}")
                st.markdown(f"**Efficiency**: {path_efficiency[i]:.1f}%")
    
    # Store in session state
    if 'performance_metrics' not in st.session_state:
        st.session_state['performance_metrics'] = {}
        
    st.session_state['performance_metrics']['leader_follower'] = {
        'total_time': total_time,
        'total_steps': total_steps,
        'agent_steps': metrics['agent_steps'],
        'waiting_time': metrics['waiting_time'],
        'efficiency': path_efficiency,
        'phase_1_time': metrics['phase_1_time'],
        'phase_2_time': metrics['phase_2_time'],
        'filled_targets': len(filled_targets),
        'total_targets': len(target_positions)
    }

    print(f"Done. Filled {len(filled_targets)} of {len(target_positions)} targets.")

    # Final grid to return
    final_grid = np.zeros_like(grid)
    for p in current_agent_positions:
        r, c = p
        final_grid[r, c] = 1

    return final_grid, list(current_agent_positions)