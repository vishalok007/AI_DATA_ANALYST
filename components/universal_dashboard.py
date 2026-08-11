import streamlit as st


def show_dashboard(df):
    st.header("Executive Dashboard")
    st.caption("A high-level snapshot of dataset readiness and analytical signals.")

    rows = len(df)
    cols = len(df.columns)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    total_cells = rows * cols if rows and cols else 1
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    completeness = round(((total_cells - missing_values) / total_cells) * 100, 1)
    null_density = round((missing_values / total_cells) * 100, 2)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Dashboard Snapshot</div>
            <div class="insight-banner-text">
                {rows:,} records loaded with {completeness}% completeness,
                {len(numeric_cols)} numeric columns, and {duplicate_rows} duplicate rows detected.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols:,}")
    c3.metric("Completeness", f"{completeness}%")
    c4.metric("Duplicates", f"{duplicate_rows:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Numeric Columns", len(numeric_cols))
    c6.metric("Categorical Columns", len(categorical_cols))
    c7.metric("Missing Cells", f"{missing_values:,}")
    c8.metric("Null Density", f"{null_density}%")

    # Automated signals
    widest_range_col = "N/A"
    highest_variance_col = "N/A"
    most_diverse_category = "N/A"

    if numeric_cols:
        valid_numeric = df[numeric_cols].select_dtypes(include="number")
        if not valid_numeric.empty:
            ranges = (valid_numeric.max(numeric_only=True) - valid_numeric.min(numeric_only=True)).dropna()
            if not ranges.empty:
                widest_range_col = str(ranges.idxmax())

            variances = valid_numeric.var(numeric_only=True).dropna()
            if not variances.empty:
                highest_variance_col = str(variances.idxmax())

    if categorical_cols:
        unique_counts = {
            col: df[col].nunique(dropna=True)
            for col in categorical_cols
        }
        if unique_counts:
            most_diverse_category = max(unique_counts, key=unique_counts.get)

    dataset_status = (
        "Ready"
        if completeness >= 95 and duplicate_rows == 0
        else "Needs Review"
    )

    st.markdown("### Analytical Signals")
    st.caption("Automatically detected highlights from the dataset structure.")

    st.markdown(f"""
    <div class="mini-card-grid">
        <div class="mini-card">
            <div class="mini-card-label">Widest Range</div>
            <div class="mini-card-value small-value">{widest_range_col}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Highest Variance</div>
            <div class="mini-card-value small-value">{highest_variance_col}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Most Diverse Category</div>
            <div class="mini-card-value small-value">{most_diverse_category}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Dataset Status</div>
            <div class="mini-card-value small-value">{dataset_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optional quick interpretation
    if dataset_status == "Ready":
        summary_text = "The dataset appears well-structured and ready for advanced visualization and modeling."
    else:
        summary_text = "The dataset can be analyzed, but a quality review is recommended before modeling."

    st.markdown(f"""
    <div class="quality-note">
        <div class="quality-note-title">Dashboard Interpretation</div>
        <div class="quality-note-text">{summary_text}</div>
    </div>
    """, unsafe_allow_html=True)
