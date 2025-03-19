import streamlit as st
import numpy as np
from grid_module import initialize_grid, display_grid
from pathfind_customshapes import move_agents_layered_fill  # <-- Our new function!

# 1️⃣ Initialize Session States
if "grid_size" not in st.session_state:
    st.session_state["grid_size"] = (10, 10)
if "obstacles" not in st.session_state:
    st.session_state["obstacles"] = []

rows, cols = st.session_state["grid_size"]
obstacles = st.session_state["obstacles"]

st.title("🖊 Custom Shape Mode - Draw Your Own Shape")
grid_placeholder = st.empty()

# 2️⃣ Initialize Grid & Agents
grid, agent_positions, _ = initialize_grid(rows, cols, obstacles)
display_grid(grid, grid_placeholder)

st.subheader("Draw your shape on the grid:")
cell_states = []
for i in range(rows):
    row_cols = st.columns(cols)
    row_state = [
        row_cols[j].checkbox(
            "",
            key=f"cell_{i}_{j}",
            value=False,
            disabled=(i, j) in obstacles  # Disable obstacles on shape cells
        )
        for j in range(cols)
    ]
    cell_states.append(row_state)

if st.button("Start Movement"):
    # 3️⃣ Collect target cells from checkboxes
    custom_targets = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if cell_states[i][j] and (i, j) not in obstacles
    ]

    if not custom_targets:
        st.warning("⚠ Please select at least one valid target.")
    else:
        # If you have 16 agents, confirm 16 boxes
        if len(custom_targets) != len(agent_positions):
            st.warning(
                f"Please select exactly {len(agent_positions)} target cells. "
                f"(You selected {len(custom_targets)})"
            )
        else:
            st.write(f"Selected Targets: {custom_targets}")

            # 4️⃣ Prepare a new grid
            new_grid = np.zeros((rows, cols), dtype=int)

            # Mark obstacles if needed
            for (r, c) in obstacles:
                new_grid[r, c] = -1

            # Mark the agent positions
            for (r, c) in agent_positions:
                new_grid[r, c] = 1

            # 5️⃣ Mark custom shape cells as reserved (red)
            for (r, c) in custom_targets:
                new_grid[r, c] = 2  # Mark as reserved for shape

            # 6️⃣ Call dynamic BFS collision-free movement
            new_grid, new_positions = move_agents_layered_fill(
                new_grid,
                agent_positions,
                custom_targets,
                grid_placeholder,
                speed=0.3,
            )
            st.success("Movement complete!")

if st.button("🔙 Back to Mode Selection"):
    st.switch_page("pages/mode_selection.py")