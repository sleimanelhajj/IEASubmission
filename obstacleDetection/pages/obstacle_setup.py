

import streamlit as st
from shapes_library import SHAPES, shift_shape_coords
from grid_module import display_grid

# Initialize grid size
if "grid_size" not in st.session_state:
    st.session_state["grid_size"] = (10, 10)
rows, cols = st.session_state["grid_size"]

# Initialize reserved positions and obstacles
if "reserved_positions" not in st.session_state:
    st.session_state["reserved_positions"] = []
if "obstacles" not in st.session_state:
    st.session_state["obstacles"] = []

# Title and instructions
st.title("Obstacle Setup with Predefined Shapes")
st.write("Select a shape, mark reserved spots, and place obstacles.")

# Shape selection
shape_choice = st.selectbox("Choose a shape:", list(SHAPES.keys()), index=0)

# Button to mark reserved cells
grid_placeholder = st.empty()
if st.button("Mark Reserved Places for Shape"):
    # Get base shape from library
    base_shape = SHAPES.get(shape_choice, [])
    
    # Shift shape to center of grid
    shifted_positions = shift_shape_coords(base_shape, (rows, cols))
    
    # Store in session state
    st.session_state["reserved_positions"] = shifted_positions
    
    # Update grid display
    grid = [[0 for _ in range(cols)] for _ in range(rows)]  # Initialize empty grid
    for r, c in shifted_positions:
        grid[r][c] = 2  # Mark as reserved
    display_grid(grid, grid_placeholder)

# Function to display the grid
def display_grid(grid, placeholder):
    """
    Render the grid with colored cells.
    - Green: Reserved cells
    - Red: Obstacles
    - White: Available cells
    """
    grid_html = ""
    for i in range(rows):
        for j in range(cols):
            if (i, j) in st.session_state.get("reserved_positions", []):
                color = "green"
            elif (i, j) in st.session_state["obstacles"]:
                color = "red"
            else:
                color = "white"
            grid_html += f"""
            <div style="display: inline-block; 
                        width: 30px; 
                        height: 30px; 
                        border: 1px solid black; 
                        background-color: {color}; 
                        margin: 1px;">
            </div>
            """
        grid_html += "<br/>"
    placeholder.markdown(grid_html, unsafe_allow_html=True)

# Initialize grid
grid = [[0 for _ in range(cols)] for _ in range(rows)]
display_grid(grid, grid_placeholder)

# Obstacle placement interface
st.subheader("Place obstacles (avoid green reserved cells):")
obstacle_states = []
for i in range(rows):
    cols_obstacle = st.columns(cols)
    row_state = []
    for j in range(cols):
        # Disable checkbox if cell is reserved
        disabled = (i, j) in st.session_state.get("reserved_positions", [])
        obstacle = cols_obstacle[j].checkbox(
            "", 
            key=f"obstacle_{i}_{j}", 
            value=(i, j) in st.session_state["obstacles"], 
            disabled=disabled
        )
        row_state.append(obstacle)
    obstacle_states.append(row_state)

# Confirm obstacles
if st.button("Confirm Obstacles"):
    new_obstacles = [
        (i, j) 
        for i in range(rows) 
        for j in range(cols) 
        if obstacle_states[i][j]
    ]
    
    # Remove any obstacles in reserved areas
    valid_obstacles = [
        pos for pos in new_obstacles 
        if pos not in st.session_state["reserved_positions"]
    ]
    
    st.session_state["obstacles"] = valid_obstacles
    st.success("Obstacles confirmed!")
    st.switch_page("pages/predefined_shapes.py")