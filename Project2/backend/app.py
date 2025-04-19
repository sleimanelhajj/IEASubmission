# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import time
# from collections import deque
# from scipy.optimize import linear_sum_assignment
# import random

# app = Flask(__name__)
# CORS(app)  # Enable cross-origin requests

# # Shape definitions from your original code
# SHAPES = {
#     "square": [
#       [0, 0], [0, 1], [0, 2], [0, 3],
#       [1, 0], [1, 3],
#       [2, 0], [2, 3],
#       [3, 0], [3, 1], [3, 2], [3, 3]
#     ],
#     "triangle": [
#       [0, 2],
#       [1, 1], [1, 3],
#       [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]
#     ],
#     "circle": [
#       [0, 1], [0, 2], [0, 3],
#       [1, 0], [1, 4],
#       [2, 0], [2, 4],
#       [3, 0], [3, 4],
#       [4, 1], [4, 2], [4, 3]
#     ]
# }

# # Your original functions - adjusted to work with the API
# def manhattan_dist(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def bfs_dynamic(start, goal, grid, agents, obstacles):
#     """
#     BFS that treats other agents and obstacles as impassable cells.
#     agents is a set of all current agent positions (except this agent's own position).
#     obstacles is a set of impassable cells on the grid.
#     Now considers 8 directions (including diagonals).
#     Returns a list of steps from start->goal (excluding start, including goal).
#     If no path, returns [].
#     """
#     rows, cols = grid.shape
#     queue = deque([(start, [])])
#     visited = {start}

#     # 8-direction movement: up, down, left, right, plus 4 diagonals
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
#                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

#     while queue:
#         (r, c), path = queue.popleft()
#         if (r, c) == goal:
#             return path

#         for dr, dc in directions:
#             nr, nc = r + dr, c + dc
#             if 0 <= nr < rows and 0 <= nc < cols:
#                 # Treat agent cells and obstacles as impassable
#                 if (nr, nc) not in agents and (nr, nc) not in visited and (nr, nc) not in obstacles:
#                     visited.add((nr, nc))
#                     queue.append(((nr, nc), path + [(nr, nc)]))
#     return []

# def hungarian_assignment(agent_positions, target_positions):
#     """
#     Assign each agent to a unique target using the Hungarian Algorithm
#     (via scipy's linear_sum_assignment).
#     Returns list of (agent, target) pairs.
#     """
#     agent_positions = sorted(agent_positions)
#     target_positions = sorted(target_positions)
#     n_agents = len(agent_positions)
#     n_targets = len(target_positions)
#     size = max(n_agents, n_targets)

#     cost_matrix = np.zeros((size, size), dtype=int)
#     for i in range(size):
#         for j in range(size):
#             if i < n_agents and j < n_targets:
#                 cost_matrix[i, j] = manhattan_dist(agent_positions[i], target_positions[j])
#             else:
#                 # If there's a mismatch in counts, put a large cost so it's never chosen
#                 cost_matrix[i, j] = 999999

#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     assignments = []
#     for i in range(size):
#         if row_ind[i] < n_agents and col_ind[i] < n_targets:
#             agent = agent_positions[row_ind[i]]
#             target = target_positions[col_ind[i]]
#             assignments.append((agent, target))
#     return assignments

# def move_agents_no_collision_api(grid, agent_positions, target_positions, obstacle_positions=None):
#     """
#     Modified version of your move_agents_no_collision function to:
#     1. Remove Streamlit dependencies
#     2. Return step-by-step data for the React frontend
#     3. Slightly simplified for API usage
#     """
#     # Initialize metrics
#     metrics = {
#         'agent_steps': [0] * len(agent_positions),
#         'waiting_time': [0] * len(agent_positions),
#         'manhattan_distances': [0] * len(agent_positions),
#         'path_length': [0] * len(agent_positions),
#         'reached_target': [False] * len(agent_positions),
#         'is_inner': [False] * len(agent_positions),
#         'assigned_target': [None] * len(agent_positions),
#         'inner_targets_count': 0,
#         'outer_targets_count': 0,
#         'obstacles_count': 0,
#         'conflicts': 0
#     }
    
#     # Use provided obstacle positions or extract from grid
#     if obstacle_positions is None:
#         obstacle_positions = []
#         for r in range(grid.shape[0]):
#             for c in range(grid.shape[1]):
#                 if grid[r, c] == -1:
#                     obstacle_positions.append((r, c))
    
#     metrics['obstacles_count'] = len(obstacle_positions)
    
#     # -------- Identify Inner and Outer Targets --------
#     target_set = set(target_positions)
    
#     inner_targets = []
#     outer_targets = []
    
#     # Directions for neighboring cells
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
#                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
#     for target in target_positions:
#         r, c = target
#         is_boundary = False
        
#         for dr, dc in directions:
#             nr, nc = r + dr, c + dc
#             # If the neighbor is not in targets, this is a boundary target
#             if (nr, nc) not in target_set:
#                 is_boundary = True
#                 break
        
#         if is_boundary:
#             outer_targets.append(target)
#         else:
#             inner_targets.append(target)
    
#     # If no inner targets were found, use centrality as an alternative
#     if not inner_targets:
#         if target_positions:
#             center_r = sum(r for r, _ in target_positions) / len(target_positions)
#             center_c = sum(c for _, c in target_positions) / len(target_positions)
            
#             # Sort by distance to center
#             sorted_targets = sorted(target_positions, 
#                                    key=lambda pos: (pos[0] - center_r) ** 2 + (pos[1] - center_c) ** 2)
            
#             inner_count = max(1, len(target_positions) // 3)
#             inner_targets = sorted_targets[:inner_count]
#             outer_targets = sorted_targets[inner_count:]
    
#     metrics['inner_targets_count'] = len(inner_targets)
#     metrics['outer_targets_count'] = len(outer_targets)
    
#     # Initialize grid
#     new_grid = np.zeros_like(grid)
#     for r, c in obstacle_positions:
#         new_grid[r, c] = -1
#     for a in agent_positions:
#         new_grid[a[0], a[1]] = 1
#     grid = new_grid
    
#     # -------- PHASE 1: INNER TARGETS ASSIGNMENT --------
#     # Phase 1: Assign agents to inner targets only
#     remaining_agents = agent_positions.copy()
#     inner_assignments = []
    
#     # If we have inner targets, assign closest agents to them
#     if inner_targets:
#         # Create cost matrix for inner targets only
#         inner_cost = []
#         for agent in remaining_agents:
#             agent_costs = []
#             for target in inner_targets:
#                 dist = manhattan_dist(agent, target)
#                 agent_costs.append((dist, agent, target))
#             agent_costs.sort()
#             inner_cost.append(agent_costs)
        
#         # Flatten and sort all agent-target pairs by distance
#         all_costs = []
#         for agent_costs in inner_cost:
#             all_costs.extend(agent_costs)
#         all_costs.sort()
        
#         # Greedy assignment - take closest pairs first
#         assigned_agents = set()
#         assigned_targets = set()
        
#         for dist, agent, target in all_costs:
#             if agent not in assigned_agents and target not in assigned_targets:
#                 inner_assignments.append((agent, target))
#                 assigned_agents.add(agent)
#                 assigned_targets.add(target)
                
#                 # Find the agent index and update metrics
#                 agent_idx = agent_positions.index(agent)
#                 metrics['is_inner'][agent_idx] = True
#                 metrics['assigned_target'][agent_idx] = target
#                 metrics['manhattan_distances'][agent_idx] = dist
                
#                 # Calculate initial path length for efficiency metrics
#                 initial_path = bfs_dynamic(agent, target, grid, set(), set(obstacle_positions))
#                 if initial_path:
#                     metrics['path_length'][agent_idx] = len(initial_path)
#                 else:
#                     metrics['path_length'][agent_idx] = dist  # Use Manhattan distance if no path found
                
#                 # Stop when all inner targets are assigned
#                 if len(assigned_targets) == len(inner_targets):
#                     break
        
#         # Remove assigned agents from remaining_agents
#         remaining_agents = [a for a in remaining_agents if a not in assigned_agents]
    
#     # -------- PHASE 2: OUTER TARGETS ASSIGNMENT --------
#     # Phase 2: Assign remaining agents to outer targets
#     outer_assignments = []
    
#     if outer_targets and remaining_agents:
#         # Create cost matrix for outer targets only
#         outer_cost = []
#         for agent in remaining_agents:
#             agent_costs = []
#             for target in outer_targets:
#                 dist = manhattan_dist(agent, target)
#                 agent_costs.append((dist, agent, target))
#             agent_costs.sort()
#             outer_cost.append(agent_costs)
        
#         # Flatten and sort all agent-target pairs by distance
#         all_costs = []
#         for agent_costs in outer_cost:
#             all_costs.extend(agent_costs)
#         all_costs.sort()
        
#         # Greedy assignment - take closest pairs first
#         assigned_agents = set()
#         assigned_targets = set()
        
#         for dist, agent, target in all_costs:
#             if agent not in assigned_agents and target not in assigned_targets:
#                 outer_assignments.append((agent, target))
#                 assigned_agents.add(agent)
#                 assigned_targets.add(target)
                
#                 # Find the agent index and update metrics
#                 agent_idx = agent_positions.index(agent)
#                 metrics['is_inner'][agent_idx] = False
#                 metrics['assigned_target'][agent_idx] = target
#                 metrics['manhattan_distances'][agent_idx] = dist
                
#                 # Calculate initial path length for efficiency metrics
#                 initial_path = bfs_dynamic(agent, target, grid, set(), set(obstacle_positions))
#                 if initial_path:
#                     metrics['path_length'][agent_idx] = len(initial_path)
#                 else:
#                     metrics['path_length'][agent_idx] = dist  # Use Manhattan distance if no path found
                
#                 # Stop when all agents or targets are assigned
#                 if len(assigned_targets) == len(outer_targets) or len(assigned_agents) == len(remaining_agents):
#                     break
    
#     # Combine assignments (inner first, then outer)
#     all_assignments = inner_assignments + outer_assignments
    
#     # -------- Store agent state with target info --------
#     agents = []
#     for agent, target in all_assignments:
#         is_inner = target in inner_targets
#         agent_idx = agent_positions.index(agent)
#         agents.append({
#             'id': agent_idx,
#             'pos': agent, 
#             'target': target, 
#             'is_inner': is_inner
#         })
    
#     # Track any agents without targets (happens when more agents than targets)
#     unassigned_agents = [a for a in agent_positions if a not in [ag['pos'] for ag in agents]]
#     for agent in unassigned_agents:
#         agent_idx = agent_positions.index(agent)
#         agents.append({
#             'id': agent_idx,
#             'pos': agent,
#             'target': None,
#             'is_inner': False
#         })
    
#     # -------- PHASE 3: MOVEMENT ALGORITHM WITH STEP TRACKING --------
#     # For React frontend - collect all steps
#     simulation_steps = []
    
#     # Add initial state
#     initial_agents = []
#     for ag in agents:
#         initial_agents.append({
#             'id': ag['id'] + 1,  # 1-indexed for display
#             'x': ag['pos'][0],
#             'y': ag['pos'][1]
#         })
    
#     # Create path visualization matrix for the frontend
#     path_visualization = {}
#     for agent in agents:
#         if agent['target'] is not None:
#             current = agent['pos']
#             target = agent['target']
#             path = bfs_dynamic(current, target, grid, set(), set(obstacle_positions))
#             # Mark path cells for visualization
#             for cell in path:
#                 r, c = cell
#                 if (r, c) != current and (r, c) != target:  # Skip start and end
#                     if r not in path_visualization:
#                         path_visualization[r] = {}
#                     path_visualization[r][c] = True
    
#     simulation_steps.append({
#         'step': 0,
#         'agents': initial_agents,
#         'paths': path_visualization
#     })
    
#     # Main movement loop
#     step_counter = 1
#     max_steps = 100  # Limit steps for API response size

#     while True and step_counter < max_steps:
#         new_grid = np.zeros_like(grid)
#         # Add obstacles back
#         for r, c in obstacle_positions:
#             new_grid[r, c] = -1
            
#         move_dict = {}
#         conflict_positions = set()
#         all_reached = True

#         # First sort by inner/outer, then by distance to target
#         # This ensures inner targets get priority in movement
#         agents_with_targets = [ag for ag in agents if ag['target'] is not None]
#         agents_with_targets.sort(key=lambda ag: (not ag['is_inner'], 
#                                               manhattan_dist(ag['pos'], ag['target'])))
        
#         # Get positions of all agents for collision avoidance
#         agent_positions_set = set(ag['pos'] for ag in agents)

#         # First process agents with targets
#         for ag in agents_with_targets:
#             current = ag['pos']
#             target = ag['target']
#             agent_id = ag['id']
            
#             if current == target:
#                 move_dict[current] = current  # Already at target
                
#                 # If this is the first time reaching the target, update metrics
#                 if not metrics['reached_target'][agent_id]:
#                     metrics['reached_target'][agent_id] = True
#                 continue
                
#             all_reached = False  # At least one agent still moving

#             # Calculate path avoiding other agents and obstacles
#             agent_set_except_self = agent_positions_set - {current}
#             path = bfs_dynamic(current, target, grid, agent_set_except_self, obstacle_positions)
            
#             if path:
#                 next_step = path[0]
#                 if next_step not in conflict_positions:
#                     move_dict[current] = next_step
#                     conflict_positions.add(next_step)
#                 else:
#                     move_dict[current] = current  # Stay put due to conflict
#                     metrics['waiting_time'][agent_id] += 1
#                     metrics['conflicts'] += 1
#             else:
#                 move_dict[current] = current  # No path, stay put
#                 metrics['waiting_time'][agent_id] += 1
        
#         # Then process agents without targets (just stay in place)
#         for ag in agents:
#             if ag['target'] is None:
#                 move_dict[ag['pos']] = ag['pos']
                
#         # Make sure ALL agents are accounted for
#         for ag in agents:
#             if ag['pos'] not in move_dict:
#                 move_dict[ag['pos']] = ag['pos']

#         # Resolve direct swaps
#         final_moves = {}
#         for old_p, new_p in move_dict.items():
#             if new_p in move_dict and move_dict[new_p] == old_p and new_p != old_p:
#                 # No swapping - both agents stay put
#                 final_moves[old_p] = old_p
                
#                 # Find the agents involved in swap and increment their conflict counters
#                 for ag in agents:
#                     if ag['pos'] == old_p or ag['pos'] == new_p:
#                         metrics['waiting_time'][ag['id']] += 1
                
#                 metrics['conflicts'] += 2  # Count as two conflicts
#             else:
#                 final_moves[old_p] = new_p

#         # Update agent positions
#         updated_agents = []
#         for ag in agents:
#             old_pos = ag['pos']
#             agent_id = ag['id']
            
#             if old_pos in final_moves:
#                 new_pos = final_moves[old_pos]
#                 if new_pos in obstacle_positions:
#                     new_pos = old_pos  # Safety check
#                     metrics['waiting_time'][agent_id] += 1
#             else:
#                 new_pos = old_pos  # Default to staying put if not in moves dict
#                 metrics['waiting_time'][agent_id] += 1
            
#             # Count a step if the agent actually moved
#             if old_pos != new_pos:
#                 metrics['agent_steps'][agent_id] += 1
                
#             # Add agent to grid
#             new_grid[new_pos[0], new_pos[1]] = 1
            
#             # Update agent position
#             updated_agents.append({
#                 'id': agent_id,
#                 'pos': new_pos,
#                 'target': ag['target'],
#                 'is_inner': ag['is_inner']
#             })

#         agents = updated_agents
#         grid = new_grid

#         # Create step data for React frontend
#         step_agents = []
#         for ag in agents:
#             step_agents.append({
#                 'id': ag['id'] + 1,  # 1-indexed for display
#                 'x': ag['pos'][0],
#                 'y': ag['pos'][1]
#             })
        
#         simulation_steps.append({
#             'step': step_counter,
#             'agents': step_agents,
#             'paths': path_visualization
#         })
        
#         step_counter += 1

#         # Exit condition - all agents have reached their targets
#         if all_reached:
#             break

#     return simulation_steps

# def shift_shape_coords(shape, grid_dims):
#     """Center the shape in the grid"""
#     rows, cols = grid_dims
#     shape_rows = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
#     shape_cols = max(c for _, c in shape) - min(c for _, c in shape) + 1
    
#     row_offset = (rows - shape_rows) // 2
#     col_offset = (cols - shape_cols) // 2
    
#     return [(r + row_offset, c + col_offset) for r, c in shape]

# @app.route('/run_simulation', methods=['POST'])
# def run_simulation():
#     """API endpoint to receive grid data and return simulation steps."""
#     try:
#         # Get data from request
#         data = request.json
#         grid_data = data['gridData']
#         config_data = data['configData']
        
#         # Convert grid data to numpy array
#         grid = np.array(grid_data)
        
#         # Extract shape positions (where value is 1)
#         target_positions = []
#         for r in range(len(grid_data)):
#             for c in range(len(grid_data[r])):
#                 if grid_data[r][c] == 1:  # Shape cells
#                     target_positions.append((r, c))
        
#         # Extract obstacle positions (where value is 2)
#         obstacle_positions = []
#         for r in range(len(grid_data)):
#             for c in range(len(grid_data[r])):
#                 if grid_data[r][c] == 2:  # Obstacle cells
#                     obstacle_positions.append((r, c))
        
#         # Determine number of agents based on config
#         agent_count = config_data.get('agentCount', 5)
        
#         # Place agents randomly on empty cells
#         empty_cells = []
#         for r in range(len(grid_data)):
#             for c in range(len(grid_data[r])):
#                 if grid_data[r][c] == 0:  # Empty cells
#                     empty_cells.append((r, c))
        
#         random.shuffle(empty_cells)
#         agent_positions = empty_cells[:agent_count]
        
#         # Run the simulation
#         simulation_steps = move_agents_no_collision_api(
#             grid,
#             agent_positions,
#             target_positions,
#             obstacle_positions
#         )
        
#         # Return result
#         return jsonify({"steps": simulation_steps})
    
#     except Exception as e:
#         import traceback
#         return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# @app.route('/get_predefined_shape', methods=['GET'])
# def get_predefined_shape():
#     """Return a predefined shape centered in the grid."""
#     try:
#         shape_name = request.args.get('shape', 'square')
#         grid_size = int(request.args.get('gridSize', 15))
        
#         if shape_name not in SHAPES:
#             return jsonify({"error": f"Shape {shape_name} not found"}), 404
        
#         # Get the raw shape
#         shape = SHAPES[shape_name]
        
#         # Center it in the grid
#         centered_shape = shift_shape_coords(shape, (grid_size, grid_size))
        
#         # Validate shape is within bounds
#         valid_shape = [(r, c) for r, c in centered_shape 
#                       if 0 <= r < grid_size and 0 <= c < grid_size]
        
#         return jsonify({"shape": valid_shape})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)


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
      [0, 0], [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 3],
      [2, 0], [2, 3],
      [3, 0], [3, 1], [3, 2], [3, 3]
    ],
    "triangle": [
      [0, 2],
      [1, 1], [1, 3],
      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]
    ],
    "circle": [
      [0, 1], [0, 2], [0, 3],
      [1, 0], [1, 4],
      [2, 0], [2, 4],
      [3, 0], [3, 4],
      [4, 1], [4, 2], [4, 3]
    ]
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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
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

# ======================================================
# 1. Inside-Out Algorithm (from your original code)
# ======================================================

def move_agents_inside_out(grid, agent_positions, target_positions, obstacle_positions):
    """
    Modified version of your move_agents_no_collision function to:
    1. Remove Streamlit dependencies
    2. Return step-by-step data for the React frontend
    3. Slightly simplified for API usage
    """
    # Initialize metrics
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
        'conflicts': 0
    }
    
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
    
    # -------- PHASE 3: MOVEMENT ALGORITHM WITH STEP TRACKING --------
    # For React frontend - collect all steps
    simulation_steps = []
    
    # Add initial state
    initial_agents = []
    for ag in agents:
        initial_agents.append({
            'id': ag['id'] + 1,  # 1-indexed for display
            'x': ag['pos'][0],
            'y': ag['pos'][1]
        })
    
    # Create path visualization matrix for the frontend
    path_visualization = {}
    for agent in agents:
        if agent['target'] is not None:
            current = agent['pos']
            target = agent['target']
            path = bfs_dynamic(current, target, grid, set(), set(obstacle_positions))
            # Mark path cells for visualization
            for cell in path:
                r, c = cell
                if (r, c) != current and (r, c) != target:  # Skip start and end
                    if r not in path_visualization:
                        path_visualization[r] = {}
                    path_visualization[r][c] = True
    
    simulation_steps.append({
        'step': 0,
        'agents': initial_agents,
        'paths': path_visualization
    })
    
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

        # Create step data for React frontend
        step_agents = []
        for ag in agents:
            step_agents.append({
                'id': ag['id'] + 1,  # 1-indexed for display
                'x': ag['pos'][0],
                'y': ag['pos'][1]
            })
        
        simulation_steps.append({
            'step': step_counter,
            'agents': step_agents,
            'paths': path_visualization
        })
        
        step_counter += 1

        # Exit condition - all agents have reached their targets
        if all_reached:
            break

    return simulation_steps

# ======================================================
# 2. Leader-Follower Algorithm
# ======================================================

def move_agents_leader_follower(grid, agent_positions, target_positions, obstacle_positions):
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
    sorted_targets = sorted(target_positions, key=lambda t: manhattan_dist(leader_pos, t))
    leader_target = sorted_targets[0]
    
    # Assign remaining targets using Hungarian algorithm
    if len(current_positions) > 1 and len(target_positions) > 1:
        follower_targets = hungarian_assignment(
            current_positions[1:], 
            [t for t in sorted_targets if t != leader_target]
        )
        follower_assignments = {agent: target for agent, target in follower_targets}
    else:
        follower_assignments = {}
    
    # Find leader's path
    leader_path = bfs_dynamic(leader_pos, leader_target, grid, set(), obstacle_positions)
    if not leader_path:
        # If no path found, return at least the initial state
        initial_agents = []
        for i, pos in enumerate(current_positions):
            initial_agents.append({
                'id': i + 1,
                'x': pos[0],
                'y': pos[1]
            })
        return [{
            'step': 0,
            'agents': initial_agents,
            'paths': {}
        }]
    
    # Track position history for each agent
    position_history = [[] for _ in range(len(current_positions))]
    for i, pos in enumerate(current_positions):
        position_history[i].append(pos)
    
    # For React frontend - collect all steps
    simulation_steps = []
    
    # Add initial state
    initial_agents = []
    for i, pos in enumerate(current_positions):
        initial_agents.append({
            'id': i + 1,
            'x': pos[0],
            'y': pos[1]
        })
    
    # Create path visualization for the React frontend
    path_visualization = {}
    
    # Highlight leader's path
    for cell in leader_path:
        r, c = cell
        if (r, c) != leader_pos and (r, c) != leader_target:  # Skip start and end
            if r not in path_visualization:
                path_visualization[r] = {}
            path_visualization[r][c] = True
    
    simulation_steps.append({
        'step': 0,
        'agents': initial_agents,
        'paths': path_visualization
    })
    
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
            ahead_agent_previous_pos = position_history[i-1][-2]  # Position before its current one
            current_pos = current_positions[i]
            
            # If not already at that position, try to move one step toward it
            if current_pos != ahead_agent_previous_pos:
                # Build set of positions occupied by other agents
                others = set()
                for j, pos in enumerate(current_positions):
                    if j != i:  # Skip self
                        others.add(pos)
                
                # Find path avoiding other agents and obstacles
                mini_path = bfs_dynamic(current_pos, ahead_agent_previous_pos, grid, others, obstacle_positions)
                
                if mini_path and len(mini_path) > 0:
                    # Move just one step along the path
                    current_positions[i] = mini_path[0]
            
            # Record this position in history
            position_history[i].append(current_positions[i])
        
        # Create step data for React frontend
        step_agents = []
        for i, pos in enumerate(current_positions):
            step_agents.append({
                'id': i + 1,
                'x': pos[0],
                'y': pos[1]
            })
        
        simulation_steps.append({
            'step': step_idx,
            'agents': step_agents,
            'paths': path_visualization
        })
    
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
            followers.append({
                'id': i,
                'pos': pos,
                'target': target
            })
    
    # Continue until all followers reach targets or max steps
    while step_counter < max_steps:
        # Check if all followers reached targets
        all_reached = True
        for follower in followers:
            if follower['pos'] != follower['target']:
                all_reached = False
                break
        
        if all_reached:
            break
        
        # Move followers independently
        agent_positions_set = {current_positions[0]}  # Leader position is fixed
        moves = {}
        
        for follower in followers:
            current = follower['pos']
            target = follower['target']
            
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
            old_pos = follower['pos']
            follower['pos'] = moves.get(old_pos, old_pos)
        
        # Update current_positions
        for i, follower in enumerate(followers, 1):
            current_positions[i] = follower['pos']
        
        # Create step data for React frontend
        step_agents = []
        for i, pos in enumerate(current_positions):
            step_agents.append({
                'id': i + 1,
                'x': pos[0],
                'y': pos[1]
            })
        
        simulation_steps.append({
            'step': step_counter,
            'agents': step_agents,
            'paths': path_visualization
        })
        
        step_counter += 1
    
    return simulation_steps

# ======================================================
# 3. Centralized Algorithm
# ======================================================

def move_agents_centralized(grid, agent_positions, target_positions, obstacle_positions):
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
    available_agents = list(range(1, len(current_positions)))  # All agents except leader
    
    # Leader starts at its current position
    leader_pos = current_positions[0]
    
    # For React frontend - collect all steps
    simulation_steps = []
    
    # Add initial state
    initial_agents = []
    for i, pos in enumerate(current_positions):
        initial_agents.append({
            'id': i + 1,
            'x': pos[0],
            'y': pos[1]
        })
    
    # Create path visualization for the React frontend
    path_visualization = {}
    
    simulation_steps.append({
        'step': 0,
        'agents': initial_agents,
        'paths': path_visualization
    })
    
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
        leader_path = bfs_dynamic(leader_pos, target, grid, obstacles, obstacle_positions)
        
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
                step_agents.append({
                    'id': i + 1,
                    'x': pos[0],
                    'y': pos[1]
                })
            
            simulation_steps.append({
                'step': step_counter,
                'agents': step_agents,
                'paths': path_visualization
            })
            
            step_counter += 1
        
        # Mark the target as visited
        visited_targets.add(target)
        
        # Find the closest available agent to this target
        if available_agents:
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
                step_agents.append({
                    'id': i + 1,
                    'x': pos[0],
                    'y': pos[1]
                })
            
            simulation_steps.append({
                'step': step_counter,
                'agents': step_agents,
                'paths': path_visualization
            })
            
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
# API Endpoints
# ======================================================

# Update to the run_simulation function in app.py to modify agent placement

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """API endpoint to receive grid data and return simulation steps."""
    try:
        # Get data from request
        data = request.json
        grid_data = data['gridData']
        config_data = data['configData']
        
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
        agent_count = config_data.get('agentCount', 5)
        
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
                    if grid_data[r][c] == 0 and (r, c) not in agent_positions:  # Empty cells
                        empty_cells.append((r, c))
            
            # Sort empty cells by bottom-to-top, left-to-right
            empty_cells.sort(key=lambda pos: (pos[0], -pos[1]), reverse=True)
            
            # Add needed cells
            needed = agent_count - len(agent_positions)
            agent_positions.extend(empty_cells[:needed])
        
        # Determine which algorithm to use
        algorithm = config_data.get('algorithm', 'inside-out')
        
        # Generate simulation steps using the selected algorithm
        if algorithm == 'leader-follower':
            simulation_steps = move_agents_leader_follower(
                grid, agent_positions, target_positions, obstacle_positions
            )
        elif algorithm == 'centralized':
            simulation_steps = move_agents_centralized(
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
@app.route('/get_predefined_shape', methods=['GET'])
def get_predefined_shape():
    """Return a predefined shape centered in the grid."""
    try:
        shape_name = request.args.get('shape', 'square')
        grid_size = int(request.args.get('gridSize', 15))
        
        if shape_name not in SHAPES:
            return jsonify({"error": f"Shape {shape_name} not found"}), 404
        
        # Get the raw shape
        shape = SHAPES[shape_name]
        
        # Center it in the grid
        centered_shape = shift_shape_coords(shape, (grid_size, grid_size))
        
        # Validate shape is within bounds
        valid_shape = [(r, c) for r, c in centered_shape 
                      if 0 <= r < grid_size and 0 <= c < grid_size]
        
        return jsonify({"shape": valid_shape})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)