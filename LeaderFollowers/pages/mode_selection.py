# mode_selection.py
# Description:
# This file allows the user to select which mode they want to use after deciding
# whether or not to place obstacles. It provides a few buttons that navigate to
# different pages, each dedicated to a certain shape formation mode or scenario.

import streamlit as st

# ---------------------------------------------------------
# Inline CSS for additional styling:
# - We set a different background color (light grayish)
# - We keep the same heading color (#2C3E50)
# - We style the button with a rounded corner, dark background, etc.
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #F9FBFD; /* Light background */
    }
    h1, h2, h3 {
        color: #2C3E50; /* Darker slate color */
    }
    div.stButton > button:first-child {
        background-color: #2C3E50 !important; /* Button color */
        color: #FFFFFF !important;            /* Text color */
        border-radius: 8px;                   /* Rounded corners */
        margin: 0.5em 0;                      /* Vertical spacing */
        width: 90%;                           /* Button width */
        font-size: 1.05em;                    /* Slightly larger font */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Page Title
# ---------------------------------------------------------
st.title("🎭 Select Your Mode")

# ---------------------------------------------------------
# Brief description of available shape or mode options.
# ---------------------------------------------------------
st.write("""
Now that obstacles are set (or not), choose how to form your shapes:

- **Custom Shape**: Draw or define your own arrangement of target points.
- **Predefined Shapes**: Let the robots form common shapes (square, circle, etc.).
- **Interactive Shapes**: Pick from a small palette of shapes directly on the grid.
""")

# ---------------------------------------------------------
# Layout the buttons in three columns so they appear side by side.
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    # If the user clicks on "🖊 Custom Shape", switch to the corresponding page.
    if st.button("🖊 Custom Shape"):
        st.switch_page("pages/custom_shape.py")

with col2:
    # If the user clicks on "📐 Predefined Shapes", switch to that page.
    if st.button("📐 Predefined Shapes"):
        st.switch_page("pages/predefined_shapes.py")

with col3:
    # If the user clicks on "🔵 Interactive Shapes", switch to that page.
    if st.button("🔵 Interactive Shapes"):
        st.switch_page("pages/interactive_shapes.py")

# ---------------------------------------------------------
# Optional back button to return to obstacle setup page
# ---------------------------------------------------------
if st.button("🔙 Back to Obstacle Setup"):
    st.switch_page("pages/obstacle_setup.py")
