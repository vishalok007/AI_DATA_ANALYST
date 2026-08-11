import streamlit as st
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand-pro">
            <div class="sidebar-badge">Academic Project</div>
            <div class="sidebar-title">AI Data Analyst</div>
            <div class="sidebar-subtitle">Business Intelligence Platform</div>
            <div class="sidebar-brand-text">
                Intelligent dashboard for dataset profiling, quality assessment,
                visualization, machine learning, and AI-assisted interpretation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-status-row">
            <span class="sidebar-status-pill">CSV / XLSX</span>
            <span class="sidebar-status-pill">ML Ready</span>
            <span class="sidebar-status-pill">AI Insights</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Workflow</div>
            <div class="sidebar-step">1. Upload dataset</div>
            <div class="sidebar-step">2. Review quality</div>
            <div class="sidebar-step">3. Explore charts</div>
            <div class="sidebar-step">4. Train model</div>
            <div class="sidebar-step">5. Generate insights</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Core Features</div>
            <div class="sidebar-chip-wrap">
                <span class="sidebar-chip">Data Profiling</span>
                <span class="sidebar-chip">Quality Check</span>
                <span class="sidebar-chip">Visualization</span>
                <span class="sidebar-chip">Correlation</span>
                <span class="sidebar-chip">AutoML</span>
                <span class="sidebar-chip">AI Chat</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Platform Summary</div>
            <div class="sidebar-card-text">
                A unified analytics workspace built for academic presentation,
                combining exploratory data analysis, statistical reporting,
                predictive modeling, and interactive business intelligence
                in one streamlined interface.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-mini-card">
            <div class="sidebar-mini-label">Use Case</div>
            <div class="sidebar-mini-value">Dataset Exploration & Smart Analytics</div>
        </div>
        """, unsafe_allow_html=True)
