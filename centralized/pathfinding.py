import time
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm

from grid_module import display_grid  # Custom visualization

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_dynamic(start, goal, grid, agents):
    """
    BFS for pathfinding while avoiding other agents.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = set([start])

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in agents and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return []

def hungarian_assignment(agent_positions, target_positions):
    """
    Assign each agent to a unique target using the Hungarian Algorithm.
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
                cost_matrix[i, j] = manhattan_dist(agent_positions[i], target_positions[j])
            else:
                cost_matrix[i, j] = 999999

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = [(agent_positions[row_ind[i]], target_positions[col_ind[i]]) for i in range(size) if row_ind[i] < n_agents and col_ind[i] < n_targets]
    return assignments


def move_agents_leader_follower(grid, agent_positions, target_positions, grid_placeholder, speed=0.3):
    """
    Leader-Follower (Chu-Chu Train) movement with smooth transitions.
    - The first agent (leader) moves towards its target.
    - Each follower moves step-by-step to the position previously occupied by the agent in front.
    - After leader reaches target, followers navigate independently to their targets.
    """
    # Copy agent positions to avoid modifying the original list
    current_positions = agent_positions.copy()
    
    # Assign permanent IDs to each agent (starting from 1)
    agent_ids = list(range(1, len(current_positions) + 1))
    
    leader_pos = current_positions[0]
    leader_target = target_positions[0]
    follower_targets = target_positions[1:]
    
    # Find leader's path
    leader_path = bfs_dynamic(leader_pos, leader_target, grid, set(current_positions[1:]))
    if not leader_path:
        print("No path found for leader!")
        return grid, current_positions
    
    # Insert initial position at beginning of path for smooth movement
    full_leader_path = [leader_pos] + leader_path
    
    # Track position history for each agent (initialize with current positions)
    position_history = [[] for _ in range(len(current_positions))]
    for i, pos in enumerate(current_positions):
        position_history[i].append(pos)
    
    # Leader movement phase
    for step_idx in range(1, len(full_leader_path)):
        # Move leader to next position in path
        current_positions[0] = full_leader_path[step_idx]
        position_history[0].append(current_positions[0])
        
        # For each follower, calculate path to the previous position of the agent ahead
        for i in range(1, len(current_positions)):
            # Get position of agent ahead from previous step
            ahead_agent_previous_pos = position_history[i-1][-2]  # Position before its current one
            current_pos = current_positions[i]
            
            # Calculate one step toward destination (no teleporting)
            if current_pos != ahead_agent_previous_pos:
                # Find path for this follower (avoiding other agents)
                others = set([pos for j, pos in enumerate(current_positions) if j != i])
                mini_path = bfs_dynamic(current_pos, ahead_agent_previous_pos, grid, others)
                
                if mini_path and len(mini_path) > 0:
                    # Move just one step along the path
                    current_positions[i] = mini_path[0]
            
            # Record this position in history
            position_history[i].append(current_positions[i])
        
        # Update visualization with NUMBERED agents (using permanent IDs)
        new_grid = np.zeros_like(grid)
        for i, (r, c) in enumerate(current_positions):
            new_grid[r, c] = agent_ids[i]  # Use permanent ID
            
        display_grid(new_grid, grid_placeholder)
        time.sleep(speed)
        
    # Continue leader phase until followers are reasonably close to their targets
    max_iterations = 20  # Prevent infinite looping
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if all followers are already at the positions in the leader's path
        all_aligned = True
        for i in range(1, len(current_positions)):
            target_pos = position_history[i-1][-2]
            if current_positions[i] != target_pos:
                all_aligned = False
                break
                
        if all_aligned:
            break
            
        # For each follower, calculate path to the previous position of the agent ahead
        any_moved = False
        for i in range(1, len(current_positions)):
            # Get position of agent ahead from previous step
            ahead_agent_previous_pos = position_history[i-1][-2]
            current_pos = current_positions[i]
            
            if current_pos != ahead_agent_previous_pos:
                # Find path for this follower (avoiding other agents)
                others = set([pos for j, pos in enumerate(current_positions) if j != i])
                mini_path = bfs_dynamic(current_pos, ahead_agent_previous_pos, grid, others)
                
                if mini_path and len(mini_path) > 0:
                    # Move just one step along the path
                    current_positions[i] = mini_path[0]
                    any_moved = True
            
            # Record this position in history
            position_history[i].append(current_positions[i])
        
        if not any_moved:
            break
            
        # Update visualization with NUMBERED agents
        new_grid = np.zeros_like(grid)
        for i, (r, c) in enumerate(current_positions):
            new_grid[r, c] = agent_ids[i]  # Use permanent ID
            
        display_grid(new_grid, grid_placeholder)
        time.sleep(speed)
        
    print("Leader reached target, followers now moving independently")
    
    # Independent movement phase for followers
    follower_positions = current_positions[1:]
    new_assignments = hungarian_assignment(follower_positions, follower_targets)
    
    # Create agent objects with positions and targets - maintain original IDs
    agents = []
    for i, follower_pos in enumerate(follower_positions):
        target_pos = new_assignments[i][1]
        agents.append({
            'pos': follower_pos,
            'target': target_pos,
            'id': agent_ids[i+1]  # Use original permanent ID
        })
    
    # Independent movement until all agents reach targets
    max_iterations = 100  # Safety to prevent infinite loops
    current_iteration = 0
    
    while current_iteration < max_iterations:
        current_iteration += 1
        
        # Check if all agents reached targets
        all_reached = True
        for agent in agents:
            if agent['pos'] != agent['target']:
                all_reached = False
                break
                
        if all_reached:
            break
            
        # Calculate moves for each agent (avoiding collisions)
        moves = {}
        occupied = set([current_positions[0]])  # Leader position is occupied
        
        for agent in agents:
            current = agent['pos']
            target = agent['target']
            
            if current == target:
                moves[current] = current
                occupied.add(current)
                continue
                
            # Find path avoiding other agents
            others = occupied.copy()
            path = bfs_dynamic(current, target, grid, others)
            
            if path and len(path) > 0:
                next_pos = path[0]
                if next_pos not in occupied:
                    moves[current] = next_pos
                    occupied.add(next_pos)
                else:
                    moves[current] = current  # Wait if position is occupied
            else:
                moves[current] = current  # No path, stay in place
        
        # Apply moves
        for agent in agents:
            agent['pos'] = moves[agent['pos']]
            
        # Update visualization with NUMBERED agents
        new_grid = np.zeros_like(grid)
        new_grid[current_positions[0][0], current_positions[0][1]] = agent_ids[0]  # Leader keeps its ID
        for agent in agents:
            new_grid[agent['pos'][0], agent['pos'][1]] = agent['id']  # Each follower has its permanent ID
            
        display_grid(new_grid, grid_placeholder)
        time.sleep(speed)
    
    final_positions = [current_positions[0]] + [agent['pos'] for agent in agents]
    return new_grid, final_positions

def move_agents_centralized(grid, agent_positions, target_positions, grid_placeholder, speed=0.3):
    """
    Centralized movement approach:
    - The first agent (leader) visits all target positions one by one
    - As the leader visits each target, the closest agent is assigned to that position
    - After the leader's tour, other agents move one by one to their assigned targets
    """
    # Copy agent positions to avoid modifying the original list
    current_positions = agent_positions.copy()
    
    # Assign permanent IDs to each agent (starting from 1)
    agent_ids = list(range(1, len(current_positions) + 1))
    
    # Store assignments (target_position: agent_index)
    target_assignments = {}
    available_agents = list(range(1, len(current_positions)))  # All agents except leader
    
    # Leader starts at its current position
    leader_pos = current_positions[0]
    leader_id = agent_ids[0]
    
    # Initialize visualization grid
    new_grid = np.zeros_like(grid)
    for i, (r, c) in enumerate(current_positions):
        new_grid[r, c] = agent_ids[i]
    display_grid(new_grid, grid_placeholder)
    time.sleep(speed)
    
    print("Leader is starting its tour...")
    
    # Create a set to track visited targets
    visited_targets = set()
    
    # Leader visits each target position exactly once
    for target_idx, target in enumerate(target_positions):
        # Skip if target has already been visited
        if target in visited_targets:
            print(f"Target {target} has already been visited, skipping...")
            continue
            
        # Find path for leader to current target - all other agents are obstacles
        obstacles = set(current_positions[1:])  # All agents except leader are obstacles
        leader_path = bfs_dynamic(leader_pos, target, grid, obstacles)
        
        if not leader_path:
            print(f"No path found for leader to target {target_idx+1}!")
            continue
        
        # Mark all positions in the path as potentially visited targets
        for pos in leader_path:
            if pos in target_positions:
                visited_targets.add(pos)
                
                # Find the closest available agent to this target
                min_dist = float('inf')
                closest_agent_idx = -1
                
                for agent_idx in available_agents:
                    dist = manhattan_dist(current_positions[agent_idx], pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_agent_idx = agent_idx
                
                # Assign target to closest agent
                if closest_agent_idx != -1:
                    target_assignments[pos] = closest_agent_idx
                    available_agents.remove(closest_agent_idx)
                    print(f"Target at {pos} assigned to Agent {agent_ids[closest_agent_idx]}")
        
        # Move leader along path
        for step in leader_path:
            current_positions[0] = step
            
            # Update visualization
            new_grid = np.zeros_like(grid)
            for i, (r, c) in enumerate(current_positions):
                new_grid[r, c] = agent_ids[i]
            display_grid(new_grid, grid_placeholder)
            time.sleep(speed)
        
        # Mark the target as visited if not already
        visited_targets.add(target)
        
        # Find the closest available agent to this target if not assigned yet
        if target not in target_assignments and available_agents:
            min_dist = float('inf')
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
                print(f"Target at {target} assigned to Agent {agent_ids[closest_agent_idx]}")
        
        leader_pos = target  # Update leader position
    
    print("Leader has completed its tour. Now moving other agents to their targets...")
    
    # Move other agents to their assigned targets one by one
    for target, agent_idx in target_assignments.items():
        agent_pos = current_positions[agent_idx]
        
        # Skip if agent is already at target
        if agent_pos == target:
            continue
        
        # Fix: Create proper set of obstacles (all agents except the current one)
        obstacles = set()
        for i, pos in enumerate(current_positions):
            if i != agent_idx:
                obstacles.add(pos)
                
        agent_path = bfs_dynamic(agent_pos, target, grid, obstacles)
        
        if not agent_path:
            print(f"No path found for Agent {agent_ids[agent_idx]} to target {target}!")
            continue
        
        # Move agent along path
        for step in agent_path:
            current_positions[agent_idx] = step
            
            # Update visualization
            new_grid = np.zeros_like(grid)
            for i, (r, c) in enumerate(current_positions):
                new_grid[r, c] = agent_ids[i]
            display_grid(new_grid, grid_placeholder)
            time.sleep(speed)
    
    print("All agents have reached their assigned positions")
    return new_grid, current_positions