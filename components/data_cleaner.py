import streamlit as st
from utils.cleaning import (
    convert_dtype,
    drop_columns,
    drop_missing_values,
    fill_constant,
    fill_mean,
    fill_median,
    fill_mode,
    remove_duplicates,
)


def show_data_cleaner(df, file_key=None):
    st.header("Interactive Data Cleaner")
    st.caption("Perform interactive dataset cleaning. All changes immediately update analysis tabs.")

    # Remove Duplicate Rows
    st.subheader("Duplicate Rows")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"Detected {dup_count:,} duplicate rows.")
        if st.button("Remove Duplicate Rows", key=f"btn_dup_{file_key}"):
            cleaned_df, removed = remove_duplicates(df)
            if file_key and file_key in st.session_state:
                st.session_state[file_key] = cleaned_df
            st.toast(f"{removed} duplicate rows removed successfully.")
            st.rerun()
    else:
        st.markdown(
            """
            <div class="custom-success-card">
                <strong>No duplicate rows found.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Missing Values
    st.subheader("Missing Values Imputation")
    missing_columns = df.columns[df.isnull().any()].tolist()
    if not missing_columns:
        st.markdown(
            """
            <div class="custom-success-card">
                <strong>No missing values found across dataset columns.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        column = st.selectbox("Select Column to Clean", missing_columns, key=f"cleaning_col_{file_key}")
        missing = df[column].isnull().sum()
        percentage = (missing / len(df)) * 100
        st.info(f"Column **{column}** has **{missing:,}** missing values (**{percentage:.2f}%**).")

        method = st.selectbox(
            "Select Cleaning Method",
            ["Drop Missing Rows", "Fill Mean", "Fill Median", "Fill Mode", "Fill Constant"],
            key=f"cleaning_method_{file_key}",
        )
        constant = None

        if method == "Fill Constant":
            constant = st.text_input("Enter Constant Value", key=f"constant_val_{file_key}")

        if st.button("Apply Cleaning Operation", type="primary", key=f"btn_apply_clean_{file_key}"):
            if method == "Drop Missing Rows":
                cleaned_df, removed = drop_missing_values(df)
                st.toast(f"{removed} rows containing missing values dropped.")
            elif method == "Fill Mean":
                cleaned_df = fill_mean(df, column)
                st.toast(f"Missing values in {column} filled with Mean.")
            elif method == "Fill Median":
                cleaned_df = fill_median(df, column)
                st.toast(f"Missing values in {column} filled with Median.")
            elif method == "Fill Mode":
                cleaned_df = fill_mode(df, column)
                st.toast(f"Missing values in {column} filled with Mode.")
            else:
                cleaned_df = fill_constant(df, column, constant)
                st.toast(f"Missing values in {column} filled with '{constant}'.")

            if file_key and file_key in st.session_state:
                st.session_state[file_key] = cleaned_df
            st.rerun()

    st.divider()

    # Drop Columns
    st.subheader("Drop Unnecessary Columns")
    selected_columns = st.multiselect("Select Columns to Remove", df.columns, key=f"drop_cols_{file_key}")
    if selected_columns:
        if st.button(f"Remove Selected Columns ({len(selected_columns)})", key=f"btn_drop_cols_{file_key}"):
            cleaned_df = drop_columns(df, selected_columns)
            if file_key and file_key in st.session_state:
                st.session_state[file_key] = cleaned_df
            st.toast(f"Removed {len(selected_columns)} column(s).")
            st.rerun()

    st.divider()

    # Convert Data Type
    st.subheader("Data Type Conversion")
    dtype_column = st.selectbox("Select Column for Conversion", df.columns, key=f"dtype_col_{file_key}")
    dtype = st.selectbox(
        "Convert Target Type To",
        ["Numeric", "String", "Datetime", "Boolean"],
        key=f"dtype_type_{file_key}",
    )
    if st.button("Convert Column Type", key=f"btn_convert_dtype_{file_key}"):
        cleaned_df, error = convert_dtype(df, dtype_column, dtype)
        if error:
            st.error(f"Type conversion failed: {error}")
        else:
            if file_key and file_key in st.session_state:
                st.session_state[file_key] = cleaned_df
            st.toast(f"Column '{dtype_column}' converted to {dtype}.")
            st.rerun()

    st.divider()
    st.markdown("### Cleaned Dataset Export")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Current Dataset CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key=f"btn_download_cleaned_{file_key}",
    )