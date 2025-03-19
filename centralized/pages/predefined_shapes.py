import streamlit as st
from grid_module import initialize_grid, display_grid
from pathfinding import move_agents_centralized
from shapes_library import SHAPES, shift_shape_coords


rows, cols = st.session_state.get("grid_size", (10, 10))
obstacles = st.session_state.get("obstacles", [])

st.title("🎭 Predefined Shapes Mode - Pick from Dropdown")
grid_placeholder = st.empty()

# Select shape first to determine number of agents
shape_choice = st.selectbox("Choose a shape:", list(SHAPES.keys()), index=0)

# Calculate the target positions for the selected shape
target_positions = shift_shape_coords(SHAPES[shape_choice], (rows, cols))

# Validate targets (remove any that would be invalid)
def validate_targets(target_positions, rows, cols):
    """Ensure target positions are within bounds and not obstacles."""
    return [(r, c) for (r, c) in target_positions if 0 <= r < rows and 0 <= c < rows and (r, c) not in obstacles]

target_positions = validate_targets(target_positions, rows, cols)

# Initialize grid with exactly the number of agents needed for the shape
grid, agent_positions, _ = initialize_grid(rows, cols, obstacles, num_agents=len(target_positions))
display_grid(grid, grid_placeholder)

# Show how many agents are being placed
st.info(f"Placing {len(agent_positions)} agents to form the {shape_choice} shape with {len(target_positions)} positions.")

movement_type = st.radio(
    "Movement Approach:", 
    ["Centralized"],
    index=0,
    help="Leader-Follower: Agents move in a train formation. Centralized: Leader visits all positions, then agents move to assigned targets."
)

if st.button("Start Movement"):
    if movement_type == "Centralized":
        grid, agent_positions = move_agents_centralized(grid, agent_positions, target_positions, grid_placeholder)
    else:
        from pathfinding import move_agents_leader_follower
        grid, agent_positions = move_agents_leader_follower(grid, agent_positions, target_positions, grid_placeholder)

if st.button("🔙 Back to Home"):
    st.page_link("app.py", label="Return to Home", icon="🏠")
