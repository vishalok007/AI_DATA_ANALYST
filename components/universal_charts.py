import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from utils.chart_recommender import get_auto_recommendations
from utils.chart_style import apply_chart_style
from utils.datatype import is_id_column


def shorten_label(text, max_len=22):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def get_clean_numeric_columns(df):
    """Return numeric columns excluding row IDs and primary keys."""
    if df is None or df.empty:
        return []
    return [c for c in df.select_dtypes(include="number").columns if not is_id_column(c, df)]


def get_clean_categorical_columns(df):
    """Return categorical columns excluding row IDs."""
    if df is None or df.empty:
        return []
    valid = []
    for c in df.select_dtypes(exclude="number").columns:
        if not is_id_column(c, df):
            nunique = df[c].nunique(dropna=True)
            if 1 <= nunique <= 60:
                valid.append(c)
    return valid


def get_clean_datetime_columns(df):
    """Return valid date columns."""
    if df is None or df.empty:
        return []
    valid = []
    for c in df.columns:
        if "date" in c.lower() or c.lower().endswith("_dt") or "year" in c.lower() or "month" in c.lower():
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() >= 0.4:
                    valid.append(c)
            except Exception:
                pass
    return valid


def determine_most_appropriate_chart(df):
    """
    Determine the single most appropriate chart type and columns
    for whatever CSV dataset the user uploads.
    """
    numeric_cols = get_clean_numeric_columns(df)
    categorical_cols = get_clean_categorical_columns(df)
    datetime_cols = get_clean_datetime_columns(df)

    # 1. If Date and Numeric exist -> Time Series Line Trend is Most Appropriate
    if datetime_cols and numeric_cols:
        target_num = "Sales" if "Sales" in numeric_cols else ("Weekly_Sales" if "Weekly_Sales" in numeric_cols else numeric_cols[0])
        return {
            "type": "Line Chart (Trend over Time)",
            "x_col": datetime_cols[0],
            "y_col": target_num,
            "reason": f"Time-series date column '{datetime_cols[0]}' detected. A Line Chart is the most appropriate representation to track momentum of '{target_num}' over time."
        }

    # 2. If Category and Numeric exist -> Bar Chart Comparison is Most Appropriate
    if categorical_cols and numeric_cols:
        target_cat = categorical_cols[0]
        target_num = "Sales" if "Sales" in numeric_cols else ("Weekly_Sales" if "Weekly_Sales" in numeric_cols else numeric_cols[0])
        return {
            "type": "Bar Chart (Category Ranking)",
            "x_col": target_cat,
            "y_col": target_num,
            "reason": f"Categorical variable '{target_cat}' and metric '{target_num}' detected. A Bar Chart is the most appropriate graph to compare category performance."
        }

    # 3. If Multiple Numerics exist -> Scatter Plot Relationship is Most Appropriate
    if len(numeric_cols) >= 2:
        return {
            "type": "Scatter Plot (Relationship)",
            "x_col": numeric_cols[0],
            "y_col": numeric_cols[1],
            "reason": f"Multiple continuous metrics detected. A Scatter Plot is the most appropriate graph to analyze correlation between '{numeric_cols[1]}' and '{numeric_cols[0]}'."
        }

    # 4. Fallback -> Histogram
    if numeric_cols:
        return {
            "type": "Histogram (Value Distribution)",
            "x_col": numeric_cols[0],
            "y_col": None,
            "reason": f"Single numerical variable '{numeric_cols[0]}' detected. Displaying value distribution histogram."
        }

    return None


def show_charts(df, profile, file_id):
    st.header("Smart Chart Studio")
    st.caption("Automatic best graph detection upon CSV upload with full manual chart type selection.")

    numeric_cols = get_clean_numeric_columns(df)
    categorical_cols = get_clean_categorical_columns(df)
    datetime_cols = get_clean_datetime_columns(df)
    all_cols = list(df.columns)

    if not all_cols:
        st.info("No data columns available to visualize.")
        return

    # Determine most appropriate chart for this dataset
    auto_best = determine_most_appropriate_chart(df)

    # -------------------------------------------------------------------------
    # SECTION 1: MOST APPROPRIATE GRAPH (AUTO-DETECTED ON UPLOAD)
    # -------------------------------------------------------------------------
    if auto_best:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #0284C7; border-left: 5px solid #0284C7; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #38BDF8; font-size: 1.1rem;">Most Appropriate Graph for Uploaded Dataset</span>
                <span style="background: #0284C7; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">
                    {auto_best['type']}
                </span>
            </div>
            <div style="color: #CBD5E1; font-size: 0.88rem; margin-top: 6px;">
                {auto_best['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: MANUAL GRAPH TYPE & AXES SELECTOR (FULL USER CONTROL)
    # -------------------------------------------------------------------------
    st.markdown("### Manually Choose Which Graph to Show")
    st.caption("Select your choice of graph type, X-axis, Y-axis, and grouping to customize your visualization.")

    chart_types = [
        "Line Chart (Trend over Time)",
        "Bar Chart (Category Ranking)",
        "Scatter Plot (Relationship)",
        "Pie / Donut Chart (Proportional Share)",
        "Histogram (Value Distribution)",
        "Box Plot (Spread & Outliers)",
        "Violin Plot (Density & Outliers)",
        "Area Chart (Filled Trend)",
        "Treemap (Hierarchical Share)"
    ]

    # Pre-select the most appropriate graph type as default index
    default_chart_type = auto_best["type"] if auto_best else "Bar Chart (Category Ranking)"
    default_type_idx = chart_types.index(default_chart_type) if default_chart_type in chart_types else 0

    c1, c2, c3, c4 = st.columns([1.3, 1.1, 1.1, 1])

    with c1:
        selected_chart_type = st.selectbox(
            "Choose Graph Type",
            chart_types,
            index=default_type_idx,
            key=f"{file_id}_selected_chart_type"
        )

    # Axis Options depending on chosen chart type
    default_x = auto_best["x_col"] if (auto_best and auto_best["x_col"] in all_cols) else all_cols[0]
    default_y = auto_best["y_col"] if (auto_best and auto_best["y_col"] and auto_best["y_col"] in numeric_cols) else (numeric_cols[0] if numeric_cols else all_cols[0])

    with c2:
        if "Line" in selected_chart_type or "Area" in selected_chart_type:
            x_axis_cols = datetime_cols + categorical_cols + numeric_cols
        elif "Bar" in selected_chart_type or "Pie" in selected_chart_type or "Treemap" in selected_chart_type:
            x_axis_cols = categorical_cols + numeric_cols
        else:
            x_axis_cols = numeric_cols + categorical_cols

        if not x_axis_cols:
            x_axis_cols = all_cols

        x_default_idx = x_axis_cols.index(default_x) if default_x in x_axis_cols else 0
        x_col = st.selectbox("X-Axis / Category Column", x_axis_cols, index=x_default_idx, key=f"{file_id}_x_col")

    with c3:
        if "Histogram" in selected_chart_type or "Box" in selected_chart_type or "Violin" in selected_chart_type:
            y_col = None
            st.selectbox("Y-Axis Metric", ["N/A (Single Variable Plot)"], disabled=True, key=f"{file_id}_y_col_disabled")
        else:
            y_axis_cols = numeric_cols if numeric_cols else all_cols
            y_default_idx = y_axis_cols.index(default_y) if default_y in y_axis_cols else 0
            y_col = st.selectbox("Y-Axis Metric Column", y_axis_cols, index=y_default_idx, key=f"{file_id}_y_col")

    with c4:
        color_options = ["None"] + categorical_cols
        color_col = st.selectbox("Color / Group By", color_options, key=f"{file_id}_color_col")
        group_color = None if color_col == "None" else color_col

    # Secondary Controls Row
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])

    with ctrl1:
        if y_col and ("Bar" in selected_chart_type or "Pie" in selected_chart_type or "Treemap" in selected_chart_type or "Line" in selected_chart_type):
            agg_method = st.selectbox("Aggregation Method", ["Sum", "Mean", "Median", "Count", "Max", "Min"], key=f"{file_id}_agg_method")
        else:
            agg_method = "Mean"

    with ctrl2:
        if "Bar" in selected_chart_type or "Pie" in selected_chart_type or "Treemap" in selected_chart_type:
            top_n = st.slider("Top N Categories", 5, 30, 12, 1, key=f"{file_id}_top_n")
        elif "Histogram" in selected_chart_type:
            bins_count = st.slider("Histogram Bins", 10, 80, 25, 5, key=f"{file_id}_bins_count")
        else:
            top_n = 15

    with ctrl3:
        color_scale = st.selectbox("Color Theme", ["Blues", "Viridis", "Plasma", "Turbo", "Reds", "Greens"], key=f"{file_id}_color_scale")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 3: LIVE GRAPH RENDERING ENGINE
    # -------------------------------------------------------------------------
    try:
        if "Line Chart" in selected_chart_type:
            temp = df.copy()
            if x_col in datetime_cols or "date" in x_col.lower():
                temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
                temp = temp.dropna(subset=[x_col, y_col]).sort_values(by=x_col)

            if group_color:
                fig = px.line(temp, x=x_col, y=y_col, color=group_color, title=f"{y_col} Trend over {x_col} (Grouped by {group_color})", markers=True)
            else:
                grouped = temp.groupby(x_col, dropna=False)[y_col].agg(agg_method.lower()).reset_index()
                fig = px.line(grouped, x=x_col, y=y_col, title=f"{agg_method} {y_col} Trend over {x_col}", markers=True)

        elif "Bar Chart" in selected_chart_type:
            grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg_method.lower()).reset_index()
            grouped = grouped.sort_values(by=y_col, ascending=False).head(top_n)
            grouped[x_col] = grouped[x_col].fillna("Missing").astype(str).apply(lambda x: shorten_label(x, 18))

            fig = px.bar(grouped, x=x_col, y=y_col, color=y_col, color_continuous_scale=color_scale, title=f"Top {len(grouped)} {x_col} by {agg_method} {y_col}")

        elif "Scatter Plot" in selected_chart_type:
            fig = px.scatter(df, x=x_col, y=y_col, color=group_color, opacity=0.8, title=f"{y_col} vs {x_col}")

        elif "Pie / Donut" in selected_chart_type:
            grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg_method.lower()).reset_index()
            grouped = grouped.sort_values(by=y_col, ascending=False).head(min(top_n, 10))
            grouped[x_col] = grouped[x_col].fillna("Missing").astype(str).apply(lambda x: shorten_label(x, 18))

            fig = px.pie(grouped, names=x_col, values=y_col, hole=0.4, title=f"{x_col} Share by {agg_method} {y_col}")
            fig.update_traces(textinfo="percent+label")

        elif "Histogram" in selected_chart_type:
            fig = px.histogram(df, x=x_col, color=group_color, nbins=bins_count, title=f"Distribution Histogram of {x_col}")

        elif "Box Plot" in selected_chart_type:
            fig = px.box(df, y=x_col, x=group_color, color=group_color, title=f"Box Plot & Outliers of {x_col}")

        elif "Violin Plot" in selected_chart_type:
            fig = px.violin(df, y=x_col, x=group_color, color=group_color, box=True, points="outliers", title=f"Violin Density & Outliers of {x_col}")

        elif "Area Chart" in selected_chart_type:
            temp = df.copy()
            if x_col in datetime_cols or "date" in x_col.lower():
                temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
                temp = temp.dropna(subset=[x_col, y_col]).sort_values(by=x_col)

            grouped = temp.groupby(x_col, dropna=False)[y_col].agg(agg_method.lower()).reset_index()
            fig = px.area(grouped, x=x_col, y=y_col, title=f"Filled Area Trend: {agg_method} {y_col} over {x_col}")

        elif "Treemap" in selected_chart_type:
            grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg_method.lower()).reset_index()
            grouped = grouped.sort_values(by=y_col, ascending=False).head(top_n)
            grouped[x_col] = grouped[x_col].fillna("Missing").astype(str)

            fig = px.treemap(grouped, path=[x_col], values=y_col, color=y_col, color_continuous_scale=color_scale, title=f"Treemap Share: {agg_method} {y_col} by {x_col}")

        fig = apply_chart_style(fig, tickangle=-35)
        st.plotly_chart(fig, use_container_width=True, key=f"{file_id}_main_chart_render")

    except Exception as e:
        st.error(f"Error generating {selected_chart_type}: {str(e)}")


