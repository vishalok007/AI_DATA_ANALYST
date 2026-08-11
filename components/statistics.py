import streamlit as st
def show_statistics(statistics):
    st.header("Statistical Summary")
    if statistics.empty:
        st.info("No numeric columns available.")
        return
    st.dataframe(statistics)