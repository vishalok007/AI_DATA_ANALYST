import streamlit as st


def apply_filters(df):
    """
    Apply interactive filters to the dataset.
    """

    st.divider()
    
    st.sidebar.header("Filters")

    filtered_df = df.copy()

    # ----------------------------
    # Year Filter
    # ----------------------------

    years = sorted(filtered_df["Date"].dt.year.unique())

    selected_year = st.sidebar.selectbox(
        "Select Year",
        options=["All"] + years
    )

    if selected_year != "All":
        filtered_df = filtered_df[
            filtered_df["Date"].dt.year == selected_year
        ]

    # ----------------------------
    # Store Filter
    # ----------------------------

    stores = sorted(filtered_df["Store"].unique())

    selected_store = st.selectbox(
        "Select Store",
        options=["All"] + stores
    )

    if selected_store != "All":
        filtered_df = filtered_df[
            filtered_df["Store"] == selected_store
        ]

    return filtered_df