import streamlit as st
from grid_module import initialize_grid, display_grid
from pathfinding import move_agents_no_collision
from shapes_library import SHAPES, shift_shape_coords

rows, cols = st.session_state.get("grid_size", (10, 10))
obstacles = st.session_state.get("obstacles", [])

def validate_targets(target_positions, rows, cols, obstacles):
    """Ensure target positions are within bounds and not obstacles."""
    return [(r, c) for (r, c) in target_positions if 0 <= r < rows and 0 <= c < cols and (r, c) not in obstacles]

st.title("🟦 Interactive Mode - Select Shape from UI")
grid_placeholder = st.empty()
grid, agent_positions, _ = initialize_grid(rows, cols, obstacles)
display_grid(grid, grid_placeholder)

def start_movement(shape_name, grid, agent_positions):
    target_positions = shift_shape_coords(SHAPES[shape_name], (rows, cols))
    target_positions = validate_targets(target_positions, rows, cols, obstacles)
    
    if len(agent_positions) > len(target_positions):
        st.warning(f"⚠ Not enough valid target positions! {len(agent_positions)} agents but only {len(target_positions)} targets.")
    else:
        grid, agent_positions = move_agents_no_collision(grid, agent_positions, target_positions, grid_placeholder)

display_grid(grid, grid_placeholder)

st.subheader("Select a shape by clicking on the buttons below:")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🟥 Square"):
        start_movement("Square", grid, agent_positions)
with col2:
    if st.button("🔵 Circle"):
        start_movement("Circle", grid, agent_positions)
with col3:
    if st.button("🔺 Triangle"):
        start_movement("Triangle", grid, agent_positions)
with col4:
    if st.button("❤️ Heart"):
        start_movement("Heart", grid, agent_positions)
with col5:
    if st.button("⭐ Star"):
        start_movement("Star", grid, agent_positions)

if st.button("🔙 Back to Home"):
    st.page_link("app.py", label="Return to Home", icon="🏠")
