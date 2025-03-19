import time
import numpy as np
import streamlit as st
from collections import deque
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
from grid_module import display_grid  # Custom visualization

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_dynamic(start, goal, grid, agents, obstacles):
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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Treat agent cells and obstacles as impassable
                if (nr, nc) not in agents and (nr, nc) not in visited and (nr, nc) not in obstacles:
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
                cost_matrix[i, j] = manhattan_dist(agent_positions[i], target_positions[j])
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

def move_agents_no_collision(grid, agent_positions, target_positions, grid_placeholder, speed_or_obstacles=0.3, speed=None):
    """
    Strictly enforced inside-out algorithm:
    1) First, fill ALL inner targets with closest agents
    2) Then, fill outer targets with remaining agents
    """
    # Initialize performance tracking metrics with similar structure to leader-follower
    start_time = time.time()
    metrics = {
        'agent_steps': [0] * len(agent_positions),
        'waiting_time': [0] * len(agent_positions),
        'manhattan_distances': [0] * len(agent_positions),
        'path_length': [0] * len(agent_positions),
        'reached_target': [False] * len(agent_positions),
        'is_inner': [False] * len(agent_positions),
        'assigned_target': [None] * len(agent_positions),
        'inner_targets_count': 0,
        'outer_targets_count': 0,
        'obstacles_count': 0,
        'phase_1_time': 0,  # Inner targets phase
        'phase_2_time': 0,  # Outer targets phase
        'conflicts': 0
    }
    
    # Handle parameters
    if isinstance(speed_or_obstacles, list) or isinstance(speed_or_obstacles, set):
        user_obstacles = speed_or_obstacles
        animation_speed = 0.3 if speed is None else speed
    else:
        user_obstacles = None
        animation_speed = speed_or_obstacles
    
    # Identify obstacles
    if user_obstacles is None:
        obstacle_positions = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == -1:
                    obstacle_positions.append((r, c))
    else:
        obstacle_positions = user_obstacles
    
    # Update obstacle count in metrics
    metrics['obstacles_count'] = len(obstacle_positions)
    
    # -------- Identify Inner and Outer Targets --------
    target_set = set(target_positions)
    
    inner_targets = []
    outer_targets = []
    
    # Directions for neighboring cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
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
            sorted_targets = sorted(target_positions, 
                                   key=lambda pos: (pos[0] - center_r) ** 2 + (pos[1] - center_c) ** 2)
            
            inner_count = max(1, len(target_positions) // 3)
            inner_targets = sorted_targets[:inner_count]
            outer_targets = sorted_targets[inner_count:]
    
    # Update target counts in metrics
    metrics['inner_targets_count'] = len(inner_targets)
    metrics['outer_targets_count'] = len(outer_targets)
    
    # Initialize grid
    new_grid = np.zeros_like(grid)
    for r, c in obstacle_positions:
        new_grid[r, c] = -1
    for a in agent_positions:
        new_grid[a[0], a[1]] = 1
    grid = new_grid
    
    # -------- PHASE 1: INNER TARGETS ASSIGNMENT --------
    phase_1_start = time.time()
    
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
            agent_costs.sort()  # Sort by distance
            inner_cost.append(agent_costs)
        
        # Flatten and sort all agent-target pairs by distance
        all_costs = []
        for agent_costs in inner_cost:
            all_costs.extend(agent_costs)
        all_costs.sort()  # Closest agent-target pairs first
        
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
                metrics['is_inner'][agent_idx] = True
                metrics['assigned_target'][agent_idx] = target
                metrics['manhattan_distances'][agent_idx] = dist
                
                # Calculate initial path length for efficiency metrics
                initial_path = bfs_dynamic(agent, target, grid, set(), set(obstacle_positions))
                if initial_path:
                    metrics['path_length'][agent_idx] = len(initial_path)
                else:
                    metrics['path_length'][agent_idx] = dist  # Use Manhattan distance if no path found
                
                # Stop when all inner targets are assigned
                if len(assigned_targets) == len(inner_targets):
                    break
        
        # Remove assigned agents from remaining_agents
        remaining_agents = [a for a in remaining_agents if a not in assigned_agents]
    
    # -------- PHASE 2: OUTER TARGETS ASSIGNMENT --------
    phase_2_start = time.time()
    metrics['phase_1_time'] = phase_2_start - phase_1_start
    
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
            agent_costs.sort()  # Sort by distance
            outer_cost.append(agent_costs)
        
        # Flatten and sort all agent-target pairs by distance
        all_costs = []
        for agent_costs in outer_cost:
            all_costs.extend(agent_costs)
        all_costs.sort()  # Closest agent-target pairs first
        
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
                metrics['is_inner'][agent_idx] = False
                metrics['assigned_target'][agent_idx] = target
                metrics['manhattan_distances'][agent_idx] = dist
                
                # Calculate initial path length for efficiency metrics
                initial_path = bfs_dynamic(agent, target, grid, set(), set(obstacle_positions))
                if initial_path:
                    metrics['path_length'][agent_idx] = len(initial_path)
                else:
                    metrics['path_length'][agent_idx] = dist  # Use Manhattan distance if no path found
                
                # Stop when all agents or targets are assigned
                if len(assigned_targets) == len(outer_targets) or len(assigned_agents) == len(remaining_agents):
                    break
    
    # Combine assignments (inner first, then outer)
    all_assignments = inner_assignments + outer_assignments
    
    # -------- Store agent state with target info --------
    agents = []
    for agent, target in all_assignments:
        is_inner = target in inner_targets
        agent_idx = agent_positions.index(agent)
        agents.append({
            'id': agent_idx,
            'pos': agent, 
            'target': target, 
            'is_inner': is_inner
        })
    
    # Track any agents without targets (happens when more agents than targets)
    unassigned_agents = [a for a in agent_positions if a not in [ag['pos'] for ag in agents]]
    for agent in unassigned_agents:
        agent_idx = agent_positions.index(agent)
        agents.append({
            'id': agent_idx,
            'pos': agent,
            'target': None,
            'is_inner': False
        })
    
    # -------- PHASE 3: MOVEMENT ALGORITHM --------
    phase_2_end = time.time()
    metrics['phase_2_time'] = phase_2_end - phase_2_start
    movement_start = time.time()
    
    # Display initial metrics
    display_live_metrics(metrics, False, agent_positions, start_time)
    
    # Main movement loop
    while True:
        new_grid = np.zeros_like(grid)
        # Add obstacles back
        for r, c in obstacle_positions:
            new_grid[r, c] = -1
            
        move_dict = {}
        conflict_positions = set()
        all_reached = True

        # First sort by inner/outer, then by distance to target
        # This ensures inner targets get priority in movement
        agents_with_targets = [ag for ag in agents if ag['target'] is not None]
        agents_with_targets.sort(key=lambda ag: (not ag['is_inner'], 
                                              manhattan_dist(ag['pos'], ag['target'])))
        
        # Get positions of all agents for collision avoidance
        agent_positions_set = set(ag['pos'] for ag in agents)

        # First process agents with targets
        for ag in agents_with_targets:
            current = ag['pos']
            target = ag['target']
            agent_id = ag['id']
            
            if current == target:
                move_dict[current] = current  # Already at target
                
                # If this is the first time reaching the target, update metrics
                if not metrics['reached_target'][agent_id]:
                    metrics['reached_target'][agent_id] = True
                continue
                
            all_reached = False  # At least one agent still moving

            # Calculate path avoiding other agents and obstacles
            agent_set_except_self = agent_positions_set - {current}
            path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)
            
            if path:
                next_step = path[0]
                if next_step not in conflict_positions:
                    move_dict[current] = next_step
                    conflict_positions.add(next_step)
                else:
                    move_dict[current] = current  # Stay put due to conflict
                    metrics['waiting_time'][agent_id] += 1
                    metrics['conflicts'] += 1
            else:
                move_dict[current] = current  # No path, stay put
                metrics['waiting_time'][agent_id] += 1
        
        # Then process agents without targets (just stay in place)
        for ag in agents:
            if ag['target'] is None:
                move_dict[ag['pos']] = ag['pos']
                
        # Make sure ALL agents are accounted for
        for ag in agents:
            if ag['pos'] not in move_dict:
                move_dict[ag['pos']] = ag['pos']

        # Resolve direct swaps
        final_moves = {}
        for old_p, new_p in move_dict.items():
            if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
                # No swapping - both agents stay put
                final_moves[old_p] = old_p
                
                # Find the agents involved in swap and increment their conflict counters
                for ag in agents:
                    if ag['pos'] == old_p or ag['pos'] == new_p:
                        metrics['waiting_time'][ag['id']] += 1
                
                metrics['conflicts'] += 2  # Count as two conflicts
            else:
                final_moves[old_p] = new_p

        # Update agent positions
        updated_agents = []
        for ag in agents:
            old_pos = ag['pos']
            agent_id = ag['id']
            
            if old_pos in final_moves:
                new_pos = final_moves[old_pos]
                if new_pos in obstacle_positions:
                    new_pos = old_pos  # Safety check
                    metrics['waiting_time'][agent_id] += 1
            else:
                new_pos = old_pos  # Default to staying put if not in moves dict
                metrics['waiting_time'][agent_id] += 1
            
            # Count a step if the agent actually moved
            if old_pos != new_pos:
                metrics['agent_steps'][agent_id] += 1
                
            # Add agent to grid
            new_grid[new_pos[0], new_pos[1]] = 1
            
            # Update agent position
            updated_agents.append({
                'id': agent_id,
                'pos': new_pos,
                'target': ag['target'],
                'is_inner': ag['is_inner']
            })

        agents = updated_agents
        grid = new_grid

        # Visualization
        display_grid(grid, grid_placeholder)
        
        # Display live metrics every few iterations
        if int(time.time() - start_time) % 5 == 0:
            display_live_metrics(metrics, all_reached, agent_positions, start_time)

        # Exit condition - all agents have reached their targets
        if all_reached:
            # Final metrics calculation
            metrics['total_time'] = time.time() - start_time
            metrics['movement_time'] = time.time() - movement_start
            display_final_metrics(metrics, agent_positions)
            break

        time.sleep(animation_speed)

    return grid, [a['pos'] for a in agents]

def display_live_metrics(metrics, all_reached, agent_positions, start_time):
    """Display metrics during the movement process in the leader-follower style"""
    elapsed_time = time.time() - start_time
    
    with st.sidebar:
        st.markdown("### 🔄 Live Metrics")
        
        # Status and progress
        if all_reached:
            st.success("All agents reached targets! ✅")
        else:
            st.info(f"Moving agents... ({elapsed_time:.2f}s elapsed)")
        
        # Summary stats in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Obstacles", metrics['obstacles_count'])
            st.metric("Inner Targets", metrics['inner_targets_count'])
        with col2:
            st.metric("Outer Targets", metrics['outer_targets_count'])
            st.metric("Total Agents", len(agent_positions))
        
        # Show steps and waits
        total_steps = sum(metrics['agent_steps'])
        total_waits = sum(metrics['waiting_time'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Steps", total_steps)
        with col2:
            st.metric("Total Waits", total_waits)

def display_final_metrics(metrics, agent_positions):
    """Display final metrics in the leader-follower style"""
    # Calculate derived metrics
    total_steps = sum(metrics['agent_steps'])
    avg_steps = total_steps / len(agent_positions) if agent_positions else 0
    total_waiting = sum(metrics['waiting_time'])
    
    # Calculate path efficiencies for agents that reached targets
    path_efficiency = []
    for i in range(len(agent_positions)):
        if metrics['reached_target'][i] and metrics['agent_steps'][i] > 0:
            # Simple efficiency calculation: optimal path vs actual steps
            efficiency = (metrics['path_length'][i] / metrics['agent_steps'][i]) * 100
            path_efficiency.append(efficiency)
        else:
            path_efficiency.append(0)
    
    avg_efficiency = sum(path_efficiency) / sum(1 for e in path_efficiency if e > 0) if any(path_efficiency) else 0
    
    with st.sidebar:
        st.markdown("## 📊 Performance Summary")
        
        # Overall timing
        st.markdown("### Timing")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Time", f"{metrics['total_time']:.2f}s")
            st.metric("Assignment Time", f"{metrics['phase_1_time'] + metrics['phase_2_time']:.2f}s")
        with col2:
            st.metric("Movement Time", f"{metrics['movement_time']:.2f}s")
            st.metric("Conflicts", metrics['conflicts'])
        
        # Target statistics
        st.markdown("### Target Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Inner Targets", metrics['inner_targets_count'])
        with col2:
            st.metric("Outer Targets", metrics['outer_targets_count'])
        
        # Movement statistics
        st.markdown("### Movement Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Steps", total_steps)
            st.metric("Avg Steps/Agent", f"{avg_steps:.1f}")
        with col2:
            st.metric("Total Waits", total_waiting)
            reached = sum(1 for r in metrics['reached_target'] if r)
            st.metric("Success Rate", f"{(reached/len(agent_positions))*100:.1f}%")
        
        st.metric("Path Efficiency", f"{avg_efficiency:.1f}%", 
                 help="100% means agents took optimal paths. >100% means diagonal shortcuts were effective.")
        
        # Display efficiency explanation
        st.info("""
        Note: Path efficiency is calculated as optimal path length divided by actual steps taken.
        With diagonal movement, efficiencies >100% are possible when diagonal shortcuts are used.
        """)
        
        # Inner target agents
        st.markdown("### Inner Target Agents")
        for i in range(len(agent_positions)):
            if metrics['is_inner'][i]:
                agent_stats = st.expander(f"Agent {i+1}")
                with agent_stats:
                    st.markdown(f"**Steps**: {metrics['agent_steps'][i]}")
                    st.markdown(f"**Wait count**: {metrics['waiting_time'][i]}")
                    st.markdown(f"**Path length**: {metrics['path_length'][i]}")
                    if metrics['agent_steps'][i] > 0:
                        efficiency = f"{(metrics['path_length'][i] / metrics['agent_steps'][i]) * 100:.1f}%"
                    else:
                        efficiency = "N/A"
                    st.markdown(f"**Efficiency**: {efficiency}")
                    st.markdown(f"**Status**: {'✅ Reached' if metrics['reached_target'][i] else '⏳ Not Reached'}")
        
        # Outer target agents
        st.markdown("### Outer Target Agents")
        for i in range(len(agent_positions)):
            if metrics['assigned_target'][i] is not None and not metrics['is_inner'][i]:
                agent_stats = st.expander(f"Agent {i+1}")
                with agent_stats:
                    st.markdown(f"**Steps**: {metrics['agent_steps'][i]}")
                    st.markdown(f"**Wait count**: {metrics['waiting_time'][i]}")
                    st.markdown(f"**Path length**: {metrics['path_length'][i]}")
                    if metrics['agent_steps'][i] > 0:
                        efficiency = f"{(metrics['path_length'][i] / metrics['agent_steps'][i]) * 100:.1f}%"
                    else:
                        efficiency = "N/A"
                    st.markdown(f"**Efficiency**: {efficiency}")
                    st.markdown(f"**Status**: {'✅ Reached' if metrics['reached_target'][i] else '⏳ Not Reached'}")
        
        # Unassigned agents
        unassigned = [i for i, target in enumerate(metrics['assigned_target']) if target is None]
        if unassigned:
            st.markdown("### Unassigned Agents")
            for i in unassigned:
                st.markdown(f"**Agent {i+1}**: No target assigned")
    
    # Store metrics in session state for later use
    st.session_state['performance_metrics'] = {
        'total_time': metrics['total_time'],
        'assignment_time': metrics['phase_1_time'] + metrics['phase_2_time'],
        'movement_time': metrics['movement_time'],
        'agent_steps': metrics['agent_steps'],
        'waiting_time': metrics['waiting_time'],
        'path_efficiency': path_efficiency,
        'inner_targets_count': metrics['inner_targets_count'],
        'outer_targets_count': metrics['outer_targets_count'],
        'obstacles_count': metrics['obstacles_count'],
        'conflicts': metrics['conflicts'],
        'success_rate': sum(1 for r in metrics['reached_target'] if r) / len(agent_positions) if agent_positions else 0
    }