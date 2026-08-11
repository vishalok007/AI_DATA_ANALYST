import streamlit as st
def render_chip_list(items, empty_text="No columns found"):
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


def show_profile(profile):
    st.header("Dataset Overview")
    st.caption(
        "Quick profile of the uploaded dataset including size, structure, memory usage, and column composition."
    )

    health_text = (
        f"{profile['rows']:,} rows, {profile['columns']:,} columns, "
        f"{profile['missing_values']:,} missing values, and "
        f"{profile['duplicate_rows']:,} duplicate rows detected."
    )

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Overview Snapshot</div>
            <div class="insight-banner-text">{health_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{profile['rows']:,}")
    col2.metric("Columns", f"{profile['columns']:,}")
    col3.metric("Memory Usage", profile["memory_usage"])

    col4, col5 = st.columns(2)
    col4.metric("Missing Values", f"{profile['missing_values']:,}")
    col5.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")

    st.markdown("### Column Composition")
    st.caption("High-level breakdown of detected column types.")

    composition_html = f"""
    <div class="mini-card-grid">
        <div class="mini-card">
            <div class="mini-card-label">Numeric</div>
            <div class="mini-card-value">{len(profile['numeric_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Categorical</div>
            <div class="mini-card-value">{len(profile['categorical_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Date</div>
            <div class="mini-card-value">{len(profile['datetime_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Boolean</div>
            <div class="mini-card-value">{len(profile['boolean_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Identifier</div>
            <div class="mini-card-value">{len(profile['identifier_columns'])}</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-label">Text</div>
            <div class="mini-card-value">{len(profile['text_columns'])}</div>
        </div>
    </div>
    """
    st.markdown(composition_html, unsafe_allow_html=True)

    st.markdown("### Column Groups")
    st.caption("Expand each section to inspect detected columns.")

    with st.expander(f"Numeric Columns ({len(profile['numeric_columns'])})", expanded=True):
        render_chip_list(profile["numeric_columns"], "No numeric columns detected.")

    with st.expander(f"Categorical Columns ({len(profile['categorical_columns'])})", expanded=True):
        render_chip_list(profile["categorical_columns"], "No categorical columns detected.")

    with st.expander(f"Date Columns ({len(profile['datetime_columns'])})"):
        render_chip_list(profile["datetime_columns"], "No date columns detected.")

    with st.expander(f"Boolean Columns ({len(profile['boolean_columns'])})"):
        render_chip_list(profile["boolean_columns"], "No boolean columns detected.")

    with st.expander(f"Identifier Columns ({len(profile['identifier_columns'])})"):
        render_chip_list(profile["identifier_columns"], "No identifier columns detected.")

    with st.expander(f"Text Columns ({len(profile['text_columns'])})"):
        render_chip_list(profile["text_columns"], "No text columns detected.")
