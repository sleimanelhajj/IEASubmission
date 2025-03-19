# mode_shape_formation.py
# Description:
# This Streamlit page/file orchestrates a shape formation scenario where a user selects
# from predefined shapes, and agents in a grid must move to form that shape.

import streamlit as st
from grid_module import initialize_grid, display_grid
from pathfinding import move_agents_leader_follower
from shapes_library import SHAPES, shift_shape_coords
import numpy as np

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
# Sets up the page title, icon, and layout (wide, in this case).
st.set_page_config(
    page_title="Shape Formation Simulator",
    page_icon="🎭",
    layout="wide"
)

# ---------------------------------------------------------
# Custom CSS for styling
# ---------------------------------------------------------
# We define styles for titles, sections, buttons, etc.
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #4682B4; /* SteelBlue color */
            text-align: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f0f0f0;
        }
        .section-header {
            font-size: 1.5rem;
            color: #333;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3rem;
            font-weight: 600;
        }
        .stButton.start-button button {
            background-color: #4CAF50;
            color: white;
        }
        .stButton.back-button button {
            background-color: #f0f0f0;
            color: #333;
        }
        .info-box {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #4682B4;
            margin-bottom: 1rem;
        }
        .grid-container {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            background-color: #f9f9f9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Title (HTML-styled)
# ---------------------------------------------------------
# Renders a large title at the top of the page.
st.markdown("<h1 class='main-title'>🎭 Shape Formation Simulator</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Create a two-column layout for the page
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

# =========================================================
# Left Column: Configuration Panel
# =========================================================
with col1:
    # Section header
    st.markdown("<h2 class='section-header'>Configuration</h2>", unsafe_allow_html=True)
    
    # Shape selection
    # We retrieve the keys from SHAPES and let the user select one. 
    shape_options = list(SHAPES.keys())
    shape_choice = st.selectbox(
        "Choose a shape to form:",
        shape_options,
        index=1,  # Default index for the select box (second item in list)
        format_func=lambda x: f"{x} ({len(SHAPES[x])} agents)" # basically appends the number of target position thus corresponding to the number of agents 
        # The format_func appends the number of agents needed for that shape.
    )
    
    # Determine how many agents the chosen shape needs
    shape_positions = SHAPES[shape_choice]
    num_agents_needed = len(shape_positions)
    
    # Animation settings in an expandable panel
    # The user can toggle step-by-step movement and specify speed.
    with st.expander("Animation Settings", expanded=True):
        speed = st.slider(
            "Animation Speed",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            format="%.1f"
        )
    
    # Display an info box about the selected shape
    st.markdown(f"""
    <div class='info-box'>
        <strong>Shape Details:</strong><br>
        • {shape_choice}<br>
        • Requires {num_agents_needed} agents<br>
        • Animation speed: {speed}s per step
    </div>
    """, unsafe_allow_html=True)
    
    # Start and back buttons
    st.markdown("<div class='stButton start-button'>", unsafe_allow_html=True)
    start_button = st.button("▶️ Start Formation")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='stButton back-button'>", unsafe_allow_html=True)
    back_button = st.button("🏠 Back to Home")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If back button is pressed, go to the main app page.
    if back_button:
        st.page_link("app.py", label="Return to Home", icon="🏠")


# =========================================================
# Right Column: Simulation Display
# =========================================================
rows, cols = st.session_state.get("grid_size", (30, 30))  
obstacles = st.session_state.get("obstacles", [])        # Retrieve obstacle list if any

with col2:
    # Section header for the simulation area
    st.markdown("<h2 class='section-header'>Simulation</h2>", unsafe_allow_html=True)
    
    # Place the text-based grid output in a styled container
    st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
    grid_placeholder = st.empty()  # We'll use this placeholder to display the grid
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Initialize a grid that has exactly the number of agents we need
    # -----------------------------------------------------
    def initialize_grid_with_exact_agents(num_needed, rows, cols, obstacles):
        """
        Creates a grid of size rows×cols, placing exactly 'num_needed' agents
        at the bottom. We also embed any obstacle positions.
        """
        # Create a blank grid (all zeros)
        grid = np.zeros((rows, cols), dtype=int)
        
        # Mark obstacles in the grid with -1
        if obstacles is None:
            obstacles = []
        for x, y in obstacles:
            grid[x, y] = -1
        
        # Place agents from the bottom row upwards until we have 'num_needed' agents
        agent_positions = []
        for i in range(rows):
            for j in range(cols):
                if len(agent_positions) < num_needed:
                    pos = (rows - i - 1, j)  # (x, y) from bottom
                    if pos not in obstacles:
                        agent_positions.append(pos)
                        grid[pos] = 1  # Mark that cell as an agent
        return grid, agent_positions

    # Build the grid using our custom method:
    grid, agent_positions = initialize_grid_with_exact_agents(num_agents_needed, rows, cols, obstacles)
    # Display the initial layout
    display_grid(grid, grid_placeholder)


# ---------------------------------------------------------
# Helper functions for target position validation
# ---------------------------------------------------------
def validate_targets(target_positions, rows, cols):
    """
    Return only valid (row, col) targets that are within the grid bounds
    and are not on obstacles.
    """
    return [
        (r, c) for (r, c) in target_positions
        if 0 <= r < rows and 0 <= c < cols and (r, c) not in obstacles
    ]

def position_targets_far(shape_positions, grid_size):
    """
    Offsets the chosen shape positions so that they appear
    near the top of the grid, instead of centered or bottom.
    """
    rows, cols = grid_size
    offset_y = 3  # Bring them down from the top by 3 rows

    # Calculate the shape's width by using min and max of x-coordinates
    min_x = min(x for _, x in shape_positions) if shape_positions else 0
    max_x = max(x for _, x in shape_positions) if shape_positions else 0
    shape_width = max_x - min_x + 1
    
    # Attempt to center the shape horizontally in the grid
    offset_x = (cols - shape_width) // 2
    
    # Return a list of adjusted coordinates
    return [(y + offset_y, x + offset_x) for (y, x) in shape_positions]


# ---------------------------------------------------------
# If the user presses the Start button, do the formation
# ---------------------------------------------------------
if start_button:
    status = st.empty()  # status placeholder for messages
    status.info("Calculating optimal paths...")

    # Prepare the shape's target positions near the top
    target_positions = position_targets_far(shape_positions, (rows, cols))
    # Filter out invalid or obstacle-laden positions
    target_positions = validate_targets(target_positions, rows, cols)
    
    # Ensure agent count matches target count
    if len(agent_positions) != len(target_positions):
        st.warning(
            f"Agent/target mismatch: {len(agent_positions)} agents "
            f"vs. {len(target_positions)} targets"
        )
        # If there are more agents than targets, cut off extra agents
        if len(agent_positions) > len(target_positions):
            agent_positions = agent_positions[:len(target_positions)]
    
    # Run leader-follower pathfinding to move the agents
    status.info("Moving agents to form the shape...")
    grid, agent_positions = move_agents_leader_follower(
        grid,
        agent_positions,
        target_positions,
        grid_placeholder,
        speed=speed
    )
    
    # Once done, display success
    status.success("✅ Formation complete! The agents have successfully formed the shape.")
