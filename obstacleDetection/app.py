import streamlit as st

st.title("🏁 Welcome to the State-Space Search Agent Program")
st.subheader("Do you want to place obstacles before selecting a mode?")


##if st.button("Yes, I want obstacles"):
  ##  st.session_state["use_obstacles"] = True
    ##st.switch_page("pages/obstacle_setup.py")

if st.button("No, proceed without obstacles"):
    st.session_state["use_obstacles"] = False
    st.session_state["obstacles"] = []  # No obstacles
    st.switch_page("pages/mode_selection.py")
