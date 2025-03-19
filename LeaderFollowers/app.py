# app.py
# Description: The main application file that sets up the Streamlit app and switches between different pages.

import streamlit as st

# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
# This sets up some basic configuration for the Streamlit page:
# - page_title: The title displayed in the browser tab.
# - page_icon: An icon (emoji or image) to display in the browser tab.
# - layout: 'centered' or 'wide' determines how the page content is laid out.
st.set_page_config(
    page_title="Modular Robots",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------------
# Custom CSS styling for a consistent look and feel
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background color + text styling */
    .main {
        background-color: #F5F8FA; /* Light gray-blue background */
    }
    /* Default headings get a darker color for better contrast */
    h1, h2, h3, h4 {
        color: #2C3E50; /* Darker slate-gray text */
    }
    /* Style the buttons */
    div.stButton > button:first-child {
        background-color: #2C3E50; /* Dark slate color for button background */
        color: white;             /* White text for contrast */
        border-radius: 8px;       /* Rounded corners */
        margin: 0.5em 0;          /* Spacing around the button */
        width: 80%;               /* Button width at 80% of container */
        font-size: 1.1em;         /* Slightly larger font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Title and introductory text
# ---------------------------------------------------------
# The app's main title and a short description or overview
st.title("🤖 Modular Robot Shape Formation")
st.markdown(
    """
**Dr. Joe Tekli** - *Intelligent Engineering Algorithms, Spring 2025*  

Welcome to the **Modular Robot** project! This application demonstrates how multiple agents (like tiny robots)
coordinate to form various shapes on a grid. They can navigate around obstacles (if enabled) and work together to
reach target points. Feel free to explore and switch between different modes to see how the robots handle shape
formation.
"""
)

# ---------------------------------------------------------
# Let user decide if they want obstacles or not
# ---------------------------------------------------------
# We use a subheader to prompt the user about obstacle placement.
st.subheader("Do you want to place obstacles before selecting a mode?")

# If the user chooses to place obstacles, update session state to reflect that
# and navigate (switch_page) to the obstacle setup page.
if st.button("Yes, I want obstacles"):
    st.session_state["use_obstacles"] = True
    st.switch_page("pages/obstacle_setup.py")

# If the user chooses not to place obstacles, reset obstacles to an empty list
# and go directly to the mode selection page.
if st.button("No, proceed without obstacles"):
    st.session_state["use_obstacles"] = False
    st.session_state["obstacles"] = []
    st.switch_page("pages/mode_selection.py")
