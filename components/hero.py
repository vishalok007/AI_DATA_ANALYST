import streamlit as st

def show_hero():
    hero_html = """
<div class="hero-container">
    <div class="hero-title">
        AI Data Analyst Platform
    </div>
    <div class="hero-subtitle">
        Analyze CSV and Excel datasets using Artificial Intelligence,
        Machine Learning and Interactive Analytics.
    </div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)