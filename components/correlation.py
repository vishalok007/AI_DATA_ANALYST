import streamlit as st
import plotly.express as px
def show_correlation(correlation_matrix):
    if correlation_matrix is None:
        return
    st.header("Correlation Analysis")
    st.caption("Explore relationships between numeric columns.")

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )