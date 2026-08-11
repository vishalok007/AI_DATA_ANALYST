import streamlit as st


def render_chip_list(items, empty_text="No issues found"):
    if not items:
        st.markdown(
            f'<div class="empty-note">{empty_text}</div>',
            unsafe_allow_html=True
        )
        return

    chips = "".join(
        [f'<span class="overview-chip">{item}</span>' for item in items]
    )

    st.markdown(
        f'<div class="overview-chip-wrap">{chips}</div>',
        unsafe_allow_html=True
    )


def get_quality_status(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs Attention"


def show_quality(report):
    st.header("Data Quality Report")
    st.caption("Review missing values, duplicates, constant columns, and quality indicators.")

    status = get_quality_status(report["quality_score"])

    st.markdown(f"""
    <div class="quality-hero">
        <div>
            <div class="quality-hero-title">Quality Score</div>
            <div class="quality-hero-text">
                Current dataset health is rated as <strong>{status}</strong>.
            </div>
        </div>
        <div class="quality-score-pill">{report["quality_score"]}%</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{report['rows']:,}")
    col2.metric("Columns", f"{report['columns']:,}")
    col3.metric("Memory Usage", report["memory_usage"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Missing Values", f"{report['missing_values']:,}")
    col5.metric("Duplicate Rows", f"{report['duplicate_rows']:,}")
    col6.metric("Quality Status", status)

    st.markdown("### Issue Summary")
    st.caption("Quick count of the most common quality concerns.")

    st.markdown(f"""
    <div class="mini-card-grid">
        <div class="mini-card">
            <div class="mini-card-label">Missing Columns</div>
            <div class="mini-card-value">{len(report['missing_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Constant Columns</div>
            <div class="mini-card-value">{len(report['constant_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">High Cardinality</div>
            <div class="mini-card-value">{len(report['high_cardinality_columns'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recommendation = []

    if report["missing_values"] > 0:
        recommendation.append("Handle missing values before machine learning.")
    if report["duplicate_rows"] > 0:
        recommendation.append("Review duplicate rows to avoid biased analysis.")
    if len(report["constant_columns"]) > 0:
        recommendation.append("Remove constant columns because they add no predictive value.")
    if len(report["high_cardinality_columns"]) > 0:
        recommendation.append("Inspect high-cardinality fields before encoding or grouping.")
    if not recommendation:
        recommendation.append("Dataset looks clean and ready for deeper analysis.")

    st.markdown(f"""
    <div class="quality-note">
        <div class="quality-note-title">Recommendation</div>
        <div class="quality-note-text">{' '.join(recommendation)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Quality Details")
    st.caption("Expand each section to inspect affected columns.")

    with st.expander(f"Columns with Missing Values ({len(report['missing_columns'])})", expanded=True):
        render_chip_list(report["missing_columns"], "No columns with missing values.")

    with st.expander(f"Constant Columns ({len(report['constant_columns'])})"):
        render_chip_list(report["constant_columns"], "No constant columns found.")

    with st.expander(f"High Cardinality Columns ({len(report['high_cardinality_columns'])})"):
        render_chip_list(report["high_cardinality_columns"], "No high-cardinality columns found.")
