import streamlit as st
import plotly.express as px

from utils.column_analysis import analyze_column
from utils.chart_style import apply_chart_style


def render_summary_note(items):
    if not items:
        return

    bullet_html = "".join(
        [f"<li>{item}</li>" for item in items]
    )

    st.markdown(f"""
    <div class="column-summary-card">
        <div class="column-summary-title">Column Insights</div>
        <ul class="column-summary-list">
            {bullet_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)


from utils.datatype import is_id_column


def show_column_analyzer(df, file_id):
    st.header("Column Explorer")
    st.caption("Analyze an individual column with metadata, statistical patterns, and outlier insights.")

    # Find the first non-ID column for smart default selection
    non_id_cols = [c for c in df.columns if not is_id_column(c, df)]
    default_idx = 0
    if non_id_cols and non_id_cols[0] in df.columns:
        default_idx = list(df.columns).index(non_id_cols[0])

    selected_column = st.selectbox(
        "Select Column",
        df.columns,
        index=default_idx,
        key=f"{file_id}_column_analyzer"
    )

    if is_id_column(selected_column, df):
        st.warning(
            f"Note: '{selected_column}' is detected as a Unique Row Identifier / Primary Key (values 1 to {len(df)}). "
            "Sequential ID columns produce uniform solid distributions rather than meaningful feature variation."
        )


    analysis = analyze_column(df, selected_column)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Selected Column</div>
            <div class="insight-banner-text">
                Exploring <strong>{analysis['name']}</strong> with detected type
                <strong>{analysis['dtype']}</strong> and {analysis['unique']} unique values.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Basic metadata
    # --------------------------------------------------
    st.markdown("### Column Metadata")
    st.caption("Basic profile information for the selected column.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Column Name", analysis["name"])
    col2.metric("Data Type", analysis["dtype"])
    col3.metric("Missing Values", analysis["missing"])

    col4, col5 = st.columns(2)
    col4.metric("Unique Values", analysis["unique"])
    col5.metric("Memory (KB)", analysis["memory"])

    # --------------------------------------------------
    # Numeric analysis
    # --------------------------------------------------
    if "mean" in analysis:
        st.markdown("### Statistical Summary")
        st.caption("Central tendency, spread, and shape metrics for the selected numeric column.")

        stats_html = f"""
        <div class="mini-card-grid">
            <div class="mini-card">
                <div class="mini-card-label">Mean</div>
                <div class="mini-card-value">{analysis['mean']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Median</div>
                <div class="mini-card-value">{analysis['median']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Mode</div>
                <div class="mini-card-value">{analysis['mode']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Minimum</div>
                <div class="mini-card-value">{analysis['minimum']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Maximum</div>
                <div class="mini-card-value">{analysis['maximum']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Std Dev</div>
                <div class="mini-card-value">{analysis['std']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Variance</div>
                <div class="mini-card-value">{analysis['variance']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Skewness</div>
                <div class="mini-card-value">{analysis['skewness']}</div>
            </div>
            <div class="mini-card">
                <div class="mini-card-label">Kurtosis</div>
                <div class="mini-card-value">{analysis['kurtosis']}</div>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

        # Charts
        st.markdown("### Distribution Analysis")
        st.caption("Visual inspection of value spread and extreme observations.")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_hist = px.histogram(
                df,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )
            fig_hist = apply_chart_style(fig_hist)
            fig_hist.update_layout(title_x=0.02)
            st.plotly_chart(fig_hist, use_container_width=True, key=f"{file_id}_column_histogram")

        with chart_col2:
            fig_box = px.box(
                df,
                y=selected_column,
                title=f"Box Plot of {selected_column}"
            )
            fig_box = apply_chart_style(fig_box)
            fig_box.update_layout(title_x=0.02)
            st.plotly_chart(fig_box, use_container_width=True, key=f"{file_id}_column_boxplot")

        # Outlier section
        st.markdown("### Outlier Analysis")
        st.caption("Outliers are detected using the IQR-based method.")

        outlier_status = "No outliers detected." if analysis["outlier_count"] == 0 else f"{analysis['outlier_count']} outliers detected."

        st.markdown(f"""
        <div class="outlier-card">
            <div class="outlier-card-title">Outlier Status</div>
            <div class="outlier-card-text">{outlier_status}</div>
        </div>
        """, unsafe_allow_html=True)

        out1, out2, out3 = st.columns(3)
        out1.metric("Outliers", analysis["outlier_count"])
        out2.metric("Lower Bound", analysis["lower_bound"])
        out3.metric("Upper Bound", analysis["upper_bound"])

        if analysis["outlier_count"] > 0:
            with st.expander("View Outlier Values"):
                st.write(analysis["outliers"])

        # Summary insights
        insights = []

        if analysis["missing"] == 0:
            insights.append("No missing values detected in this column.")
        else:
            insights.append(f"{analysis['missing']} missing values are present.")

        if analysis["outlier_count"] == 0:
            insights.append("No statistically significant outliers were detected.")
        else:
            insights.append(f"{analysis['outlier_count']} outliers were detected and may require review.")

        if analysis["skewness"] > 1:
            insights.append("The distribution is strongly right-skewed.")
        elif analysis["skewness"] < -1:
            insights.append("The distribution is strongly left-skewed.")
        else:
            insights.append("The distribution is approximately symmetric.")

        if analysis["variance"] == 0:
            insights.append("This column has no variability and may not be analytically useful.")

        render_summary_note(insights)

    # --------------------------------------------------
    # Non-numeric analysis
    # --------------------------------------------------
    else:
        st.markdown("### Categorical / Text Summary")
        st.caption("Summary statistics for non-numeric columns.")

        insights = []

        if analysis["missing"] == 0:
            insights.append("No missing values detected in this column.")
        else:
            insights.append(f"{analysis['missing']} missing values are present.")

        if analysis["unique"] == 1:
            insights.append("This column contains only one unique value.")
        elif analysis["unique"] == len(df):
            insights.append("This column behaves like an identifier with highly unique values.")
        else:
            insights.append(f"This column contains {analysis['unique']} unique values.")

        render_summary_note(insights)
