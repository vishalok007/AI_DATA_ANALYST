import streamlit as st
import plotly.express as px


def show_charts(df):
    """
    Display interactive business charts.
    """

    st.divider()
    st.header("Sales Analytics")

    #Monthly Sales Trend
    monthly_sales = (
        df.groupby(df["Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )
    monthly_sales["Date"] = monthly_sales["Date"].astype(str)

    # Figure
    fig1 = px.line(
        monthly_sales,
        x="Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    #Top 10 Stores
    top_store = (
        df.groupby("Store")["Sales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig2 = px.bar(
        top_store,
        x="Store",
        y="Sales",
        title="Top 10 Stores by Sales"
    )

    st.plotly_chart(fig2, use_container_width=True)