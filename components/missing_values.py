import streamlit as st
import plotly.express as px
def show_missing_values(result):
    st.header("Missing Values Analysis")

    if result is None:
        #st.success("No missing values found.")
        st.markdown("""
        <div class="custom-success-card">
           <strong>No missing values found.</strong>
        </div>
        """, unsafe_allow_html=True)
        return
    st.dataframe(
        result,
        use_container_width=True
    )
    fig = px.bar(
        result,
        x="Column",
        y="Missing Values",
        color="Missing Values",
        title="Missing Values by Column"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )