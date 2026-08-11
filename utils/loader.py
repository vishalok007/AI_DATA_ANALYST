import pandas as pd
import streamlit as st

from components.ai_chat import show_ai_chat
from components.column_analyzer import show_column_analyzer
from components.correlation import show_correlation
from components.data_cleaner import show_data_cleaner
from components.machine_learning import show_machine_learning
from components.missing_values import show_missing_values
from components.profile import show_profile
from components.quality import show_quality
from components.statistics import show_statistics
from components.universal_charts import show_charts
from components.universal_dashboard import show_dashboard
from utils.correlation import calculate_correlation
from utils.missing_values import analyze_missing_values
from utils.profiler import profile_dataset
from utils.quality import analyze_quality
from utils.statistics import calculate_statistics


def read_uploaded_file(file):
    """Read CSV or Excel file safely."""
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")
    else:
        df = pd.read_excel(file)
    return df


def upload_files():
    upload_container = st.container()

    with upload_container:
        st.markdown(
            """
            <div class="upload-panel">
                <div class="upload-panel-left">
                    <div class="upload-panel-title">Upload Dataset</div>
                    <div class="upload-panel-text">
                        Choose a CSV or Excel dataset to begin analysis.
                    </div>
                    <div class="upload-panel-meta">Supported formats: CSV (.csv), Excel (.xlsx)</div>
                    <div class="upload-panel-meta">Maximum file size: 200 MB</div>
                    <div class="upload-panel-tip">
                        Tip: Upload a clean dataset for better analysis and ML results.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploader_col1, uploader_col2 = st.columns([1.15, 1.35], gap="large")

        with uploader_col1:
            st.empty()

        with uploader_col2:
            uploaded_files = st.file_uploader(
                label="Upload one or more datasets",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
            )

    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = []

    if not uploaded_files:
        return

    current_files = [file.name for file in uploaded_files]

    if current_files != st.session_state.last_uploaded_files:
        st.toast(f"{len(uploaded_files)} file(s) uploaded successfully.")
        st.session_state.last_uploaded_files = current_files

    for file in uploaded_files:
        st.divider()

        data_key = f"df_{file.name}_{file.size}"
        raw_key = f"raw_df_{file.name}_{file.size}"

        if data_key not in st.session_state or raw_key not in st.session_state:
            initial_df = read_uploaded_file(file)
            st.session_state[data_key] = initial_df
            st.session_state[raw_key] = initial_df.copy()

        df = st.session_state[data_key]
        raw_df = st.session_state[raw_key]

        is_modified = not df.equals(raw_df)

        col_left, col_right = st.columns([3, 1])
        with col_left:
            status_tag = " (Cleaned / Modified)" if is_modified else " (Original)"
            st.caption(f"Analysis dashboard for: **{file.name}** {status_tag} | {len(df):,} rows x {len(df.columns):,} columns")
        with col_right:
            if is_modified:
                if st.button("Reset to Original", key=f"reset_{data_key}", use_container_width=True):
                    st.session_state[data_key] = st.session_state[raw_key].copy()
                    st.toast("Dataset reset to original state.")
                    st.rerun()

        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        is_wide_dataset = cols > 200
        is_large_numeric_dataset = len(numeric_cols) > 100
        is_large_dataset = rows * cols > 500000

        if is_wide_dataset or is_large_numeric_dataset or is_large_dataset:
            st.markdown(
                """
                <div class="theme-warning-card">
                    <div class="theme-warning-title">Large Dataset Mode</div>
                    <div class="theme-warning-text">
                        This dataset is very large or wide. Heavy analyses are limited automatically
                        to keep the app stable.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        profile = profile_dataset(df)
        quality = analyze_quality(df)
        missing_result = analyze_missing_values(df)

        statistics = None
        correlation_matrix = None

        if not is_wide_dataset and not is_large_numeric_dataset:
            statistics = calculate_statistics(df, profile)

        if 2 <= len(numeric_cols) <= 40:
            correlation_matrix = calculate_correlation(df[numeric_cols])

        tabs = st.tabs(
            [
                "Overview",
                "Quality",
                "Data Cleaner",
                "Dashboard",
                "Charts",
                "Statistics",
                "Column Explorer",
                "Correlation",
                "Machine Learning",
                "AI Chat",
            ]
        )

        with tabs[0]:
            show_profile(profile)
            st.markdown("### Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True, height=320)

        with tabs[1]:
            show_quality(quality)
            show_missing_values(missing_result)

        with tabs[2]:
            show_data_cleaner(df, file_key=data_key)

        with tabs[3]:
            show_dashboard(df)

        with tabs[4]:
            show_charts(df, profile, file.name)

        with tabs[5]:
            if statistics is not None:
                show_statistics(statistics)
            else:
                st.info("Statistics are limited for very wide datasets.")

        with tabs[6]:
            show_column_analyzer(df, file.name)

        with tabs[7]:
            if correlation_matrix is not None:
                show_correlation(correlation_matrix)
            else:
                st.info("Correlation heatmap is disabled for datasets with too many numeric columns.")

        with tabs[8]:
            show_machine_learning(df)

        with tabs[9]:
            show_ai_chat(df, file.name)
