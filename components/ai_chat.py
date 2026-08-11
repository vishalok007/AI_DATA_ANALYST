import os
import re
import tempfile
import streamlit as st

from utils.ai_analyst import (
    ask_ai,
    explain_chart_recommendations,
    generate_business_insights,
    generate_cleaning_advice,
    generate_feature_engineering_advice,
)
from utils.pdf_report import generate_pdf_report


SUGGESTED_QUESTIONS = [
    "Summarize this dataset for an executive audience.",
    "What are the strongest analytical signals in this data?",
    "Which columns need cleaning before modeling?",
    "What charts should I review first and why?",
]


QUICK_ACTIONS = [
    {
        "key": "pdf_report",
        "title": "Generate PDF Report",
        "description": "Create a polished downloadable summary of the current dataset.",
        "button_label": "Generate PDF Report",
    },
    {
        "key": "business_insights",
        "title": "Business Insights",
        "description": "Surface high-level findings, patterns, and decision-oriented observations.",
        "button_label": "Generate Business Insights",
    },
    {
        "key": "data_cleaning",
        "title": "Data Cleaning Advisor",
        "description": "Review missing values, formatting issues, and practical cleanup recommendations.",
        "button_label": "AI Data Cleaning Advisor",
    },
    {
        "key": "explain_charts",
        "title": "Explain Charts",
        "description": "Understand why certain visualizations were recommended for this dataset.",
        "button_label": "AI Explain Recommended Charts",
    },
    {
        "key": "feature_engineering",
        "title": "Feature Engineering",
        "description": "Identify potential derived fields and transformations for modeling readiness.",
        "button_label": "AI Feature Engineering Advisor",
    },
]


def _slugify(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_")
    return slug.lower() or "dataset"


def _build_dataset_key(df, dataset_name):
    column_signature = "_".join(str(col) for col in df.columns[:6])
    return _slugify(f"{dataset_name}_{df.shape[0]}_{df.shape[1]}_{column_signature}")


def _default_state():
    return {
        "result_title": "",
        "result_body": "",
        "pdf_bytes": None,
        "pdf_name": "",
    }


def _set_result(state, title, body):
    state["result_title"] = title
    state["result_body"] = body


def _render_action_card(action):
    st.markdown(
        f"""
        <div class="ai-action-card">
            <div class="ai-action-title">{action['title']}</div>
            <div class="ai-action-text">{action['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_panel(state, dataset_name):
    if not state.get("result_title") and not state.get("pdf_bytes"):
        return

    st.markdown(
        f"""
        <div class="ai-response-shell">
            <div class="ai-response-title">Latest AI Output</div>
            <div class="ai-response-subtitle">Results generated for {dataset_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if state.get("result_title"):
        st.markdown(
            f"""
            <div class="quality-note">
                <div class="quality-note-title">{state['result_title']}</div>
                <div class="quality-note-text">Review the generated analysis below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(state["result_body"])

    if state.get("pdf_bytes"):
        st.download_button(
            label="Download PDF Report",
            data=state["pdf_bytes"],
            file_name=state["pdf_name"],
            mime="application/pdf",
            key=f"download_{_slugify(dataset_name)}",
            use_container_width=True,
        )


def show_ai_chat(df, dataset_name="dataset"):
    dataset_key = _build_dataset_key(df, dataset_name)
    state_key = f"ai_chat_state_{dataset_key}"

    if state_key not in st.session_state:
        st.session_state[state_key] = _default_state()

    state = st.session_state[state_key]

    rows, columns = df.shape
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_columns = len(df.select_dtypes(include="number").columns)
    completion = round((((rows * columns) - missing_values) / max(rows * columns, 1)) * 100, 1)

    st.header("AI Data Analyst")
    st.caption("Ask questions about your dataset using AI, with the same polished look and structure as the rest of the app.")

    st.markdown(
        f"""
        <div class="insight-banner">
            <div>
                <div class="insight-banner-title">AI Workspace Snapshot</div>
                <div class="insight-banner-text">
                    {rows:,} records loaded from <strong>{dataset_name}</strong> with {completion}% completeness,
                    {numeric_columns} numeric columns, and {duplicate_rows} duplicate rows detected.
                </div>
            </div>
            <div class="quality-score-pill">{completion}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ai-stat-grid">
            <div class="ai-stat-card">
                <div class="ai-stat-label">Rows</div>
                <div class="ai-stat-value">{rows:,}</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-label">Columns</div>
                <div class="ai-stat-value">{columns:,}</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-label">Missing Cells</div>
                <div class="ai-stat-value">{missing_values:,}</div>
            </div>
            <div class="ai-stat-card">
                <div class="ai-stat-label">Numeric Columns</div>
                <div class="ai-stat-value">{numeric_columns}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick Actions")
    st.caption("Run pre-built AI workflows from a cleaner and more professional action area.")

    primary_action_columns = st.columns(3, gap="large")
    primary_actions = QUICK_ACTIONS[:3]
    quick_action_clicks = {}

    for column, action in zip(primary_action_columns, primary_actions):
        with column:
            _render_action_card(action)
            quick_action_clicks[action["key"]] = st.button(
                action["button_label"],
                key=f"{dataset_key}_{action['key']}",
                use_container_width=True,
            )

    secondary_layout = st.columns([0.15, 1, 1, 0.15], gap="large")
    secondary_actions = QUICK_ACTIONS[3:]

    for column, action in zip(secondary_layout[1:3], secondary_actions):
        with column:
            _render_action_card(action)
            quick_action_clicks[action["key"]] = st.button(
                action["button_label"],
                key=f"{dataset_key}_{action['key']}",
                use_container_width=True,
            )

    st.markdown("### Ask your question")
    st.caption("Type a custom question about the active dataset and get an AI-generated answer.")

    suggestion_chips = "".join(
        [f'<span class="ai-suggestion-chip">{question}</span>' for question in SUGGESTED_QUESTIONS]
    )
    st.markdown(
        f"""
        <div class="ai-composer-card">
            <div class="ai-composer-title">Suggested prompts</div>
            <div class="ai-composer-text">Use these examples as inspiration for your own question.</div>
            <div class="ai-suggestion-wrap">{suggestion_chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Ask your question",
        key=f"{dataset_key}_question",
        height=160,
        placeholder="Example: Summarize this dataset. What are the top correlations? Which columns have outliers?",
        label_visibility="collapsed",
    )

    composer_columns = st.columns([1.2, 1.2, 4.6], gap="medium")
    ask_ai_clicked = composer_columns[0].button(
        "Ask AI",
        key=f"{dataset_key}_ask_ai",
        use_container_width=True,
    )
    clear_result_clicked = composer_columns[1].button(
        "Clear Result",
        key=f"{dataset_key}_clear_result",
        use_container_width=True,
    )

    if clear_result_clicked:
        st.session_state[state_key] = _default_state()
        state = st.session_state[state_key]

    try:
        if quick_action_clicks.get("pdf_report"):
            with st.spinner("Generating PDF report..."):
                safe_name = _slugify(dataset_name)
                output_path = os.path.join(
                    tempfile.gettempdir(),
                    f"{safe_name}_ai_report.pdf",
                )
                generate_pdf_report(df, output_path, dataset_name=dataset_name)
                with open(output_path, "rb") as file:
                    pdf_bytes = file.read()

            state["pdf_bytes"] = pdf_bytes
            state["pdf_name"] = f"{safe_name}_AI_Report.pdf"
            _set_result(
                state,
                "PDF report ready",
                "Your polished PDF summary has been generated successfully. Use the download button below to save it.",
            )

        if quick_action_clicks.get("business_insights"):
            with st.spinner("Generating business insights..."):
                insights = generate_business_insights(df)
            _set_result(state, "Business insights", insights)

        if quick_action_clicks.get("data_cleaning"):
            with st.spinner("Reviewing data quality..."):
                cleaning_report = generate_cleaning_advice(df)
            _set_result(state, "Data cleaning advisor", cleaning_report)

        if quick_action_clicks.get("explain_charts"):
            with st.spinner("Explaining recommended charts..."):
                chart_explanation = explain_chart_recommendations(df)
            _set_result(state, "Recommended chart explanation", chart_explanation)

        if quick_action_clicks.get("feature_engineering"):
            with st.spinner("Reviewing feature engineering opportunities..."):
                feature_report = generate_feature_engineering_advice(df)
            _set_result(state, "Feature engineering advisor", feature_report)

        if ask_ai_clicked:
            if question.strip() == "":
                st.warning("Please enter a question before asking AI.")
            else:
                with st.spinner("Thinking..."):
                    answer = ask_ai(df, question, dataset_name=dataset_name)
                _set_result(state, "Custom AI answer", answer)

    except Exception as error:
        st.error(f"Something went wrong while generating the AI output: {error}")

    _render_result_panel(state, dataset_name)
