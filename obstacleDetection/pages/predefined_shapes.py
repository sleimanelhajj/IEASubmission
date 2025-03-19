


import streamlit as st
from grid_module import initialize_grid, display_grid
from pathfinding import move_agents_no_collision
from shapes_library import SHAPES, shift_shape_coords


rows, cols = st.session_state.get("grid_size", (10, 10))
obstacles = st.session_state.get("obstacles", [])

st.title("🎭 Predefined Shapes Mode - Pick from Dropdown")
grid_placeholder = st.empty()
grid, agent_positions, _ = initialize_grid(rows, cols, obstacles)
display_grid(grid, grid_placeholder)

shape_choice = st.selectbox("Choose a shape:", list(SHAPES.keys()), index=0)

def validate_targets(target_positions, rows, cols):
    """Ensure target positions are within bounds and not obstacles."""
    return [(r, c) for (r, c) in target_positions if 0 <= r < rows and 0 <= c < cols and (r, c) not in obstacles]

def mark_reserved_cells(grid, shape, rows, cols):
    """Mark reserved shape cells as red."""
    for (r, c) in shape:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = 2  # Mark as reserved (red)
    return grid

# Get target positions for the selected shape
def get_shape_positions(shape_choice, rows, cols):
    target_positions = shift_shape_coords(SHAPES[shape_choice], (rows, cols))
    return validate_targets(target_positions, rows, cols)

# After calling move_agents_no_collision in the button "Start Movement":
if st.button("Start Movement"):
    target_positions = get_shape_positions(shape_choice, rows, cols)
    grid = mark_reserved_cells(grid, target_positions, rows, cols)
    
    if len(agent_positions) > len(target_positions):
        st.warning(f"⚠ Not enough valid target positions! {len(agent_positions)} agents but only {len(target_positions)} targets.")
    else:
        grid, agent_positions = move_agents_no_collision(grid, agent_positions, target_positions, grid_placeholder, st.session_state["obstacles"])
    
    # Ensure the obstacles are rendered correctly after the movement
    display_grid(grid, grid_placeholder)



place_obstacles = st.button(" Place obstacles?")

if(place_obstacles):
    st.session_state["use_obstacles"] = True
    st.switch_page("pages/obstacle_setup.py")
    

if st.button("🔙 Back to Home"):
    st.page_link("app.py", label="Return to Home", icon="🏠")   