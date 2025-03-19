import streamlit as st
from grid_module import initialize_grid, display_grid

st.title("🚧 Place Obstacles on the Grid")

# If the user never set grid_size, default to (10, 10)
if "grid_size" not in st.session_state:
    st.session_state["grid_size"] = (10, 10)

rows, cols = st.session_state["grid_size"]

# Ensure obstacles list exists
if "obstacles" not in st.session_state:
    st.session_state["obstacles"] = []

grid_placeholder = st.empty()
grid, agent_positions, _ = initialize_grid(rows, cols, st.session_state["obstacles"])
display_grid(grid, grid_placeholder)

st.subheader("Click on cells to place obstacles:")
obstacle_states = []
for i in range(rows):
    row_cols = st.columns(cols)
    row_state = [row_cols[j].checkbox("", key=f"obstacle_{i}_{j}", value=False) for j in range(cols)]
    obstacle_states.append(row_state)

if st.button("Confirm Obstacles"):
    st.session_state["obstacles"] = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if obstacle_states[i][j]
    ]
    st.switch_page("pages/mode_selection.py")
