import streamlit as st
from utils.loader import upload_files
from components.hero import show_hero
from components.sidebar import show_sidebar
def load_css():
    with open("assets/styles/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )        
st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()
show_sidebar()
show_hero()
upload_files()