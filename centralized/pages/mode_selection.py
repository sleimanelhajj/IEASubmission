import streamlit as st

st.title("🎭 Select Your Mode")

st.write("Now that obstacles are set, choose a mode:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🖊 Custom Shape"):
        st.switch_page("pages/custom_shape.py")

with col2:
    if st.button("📐 Predefined Shapes"):
        st.switch_page("pages/predefined_shapes.py")

with col3:
    if st.button("🔵 Interactive Shapes"):
        st.switch_page("pages/interactive_shapes.py")

if st.button("🔙 Back to Obstacle Setup"):
    st.switch_page("pages/obstacle_setup.py")
