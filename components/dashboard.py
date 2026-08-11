import streamlit as st


def show_dashboard(df):
    """Display business KPI cards."""
    st.divider()
    st.header("Business Dashboard")

    # Calculate KPIs
    total_sales = df["Sales"].sum()
    total_customers = df["Customers"].sum()
    total_stores = df["Store"].nunique()
    average_daily_sales = df["Sales"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        " Total Sales",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        " Total Customers",
        f"{total_customers:,.0f}"
    )

    col3.metric(
        " Total Stores",
        total_stores
    )

    col4.metric(
        " Avg Daily Sales",
        f"{average_daily_sales:,.2f}"
    )