import streamlit as st
import pandas as pd
import plotly.express as px
import time

from utils.automl import (
    automl_regression,
    automl_classification,
    interpret_r2
)
from utils.ai_analyst import explain_automl_results
from utils.machine_learning import (
    prepare_data,
    train_regression,
    train_logistic,
    train_random_forest,
    train_decision_tree_classifier,
    train_decision_tree_regression
)
from utils.chart_style import apply_chart_style
def get_regression_targets(df):
    return df.select_dtypes(include="number").columns.tolist()
def get_classification_targets(df, max_unique=20):
    valid_targets = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        if 2 <= unique_count <= max_unique:
            valid_targets.append(col)

    return valid_targets
def render_prediction_download(prediction_df, file_name="predictions.csv"):
    csv = prediction_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions",
        csv,
        file_name,
        "text/csv"
    )
def render_results_download(results_df, file_name):
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download AutoML Report",
        csv,
        file_name,
        "text/csv"
    )
def add_rank_column(results_df, score_column):
    results = results_df.sort_values(score_column, ascending=False).reset_index(drop=True)

    rank = []
    for i in range(len(results)):
        rank.append(f"{i+1}")

    if "Rank" in results.columns:
        results = results.drop(columns=["Rank"])

    results.insert(0, "Rank", rank)
    return results
def render_regression_results(model_name, y_test, predictions, r2, mae, rmse):
    #st.success(f"{model_name} trained successfully.")
    st.markdown(f"""
    <div class="insight-banner">
       <div>
           <div class="insight-banner-title">Success</div>
           <div class="insight-banner-text">
               {model_name} trained successfully.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Regression Summary</div>
            <div class="insight-banner-text">
                Model <strong>{model_name}</strong> completed successfully with
                R² score <strong>{r2:.4f}</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")

    prediction_df = pd.DataFrame({
        "Actual": pd.Series(y_test).reset_index(drop=True),
        "Predicted": pd.Series(predictions).reset_index(drop=True)
    })

    st.markdown("### Prediction Results")
    st.caption("Preview of actual vs predicted values.")
    st.dataframe(prediction_df.head(200), use_container_width=True, height=320)

    fig = px.scatter(
        prediction_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted"
    )
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    render_prediction_download(prediction_df, "regression_predictions.csv")

    title, message, level = interpret_r2(r2)
    st.markdown("### Model Interpretation")

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)
def render_classification_results(model_name, y_test, predictions, accuracy, report, matrix, importance=None):
    st.success(f"{model_name} trained successfully.")

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Classification Summary</div>
            <div class="insight-banner-text">
                Model <strong>{model_name}</strong> completed successfully with
                accuracy <strong>{accuracy:.4f}</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("### Confusion Matrix")
    fig = px.imshow(matrix, text_auto=True, title="Confusion Matrix")
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True, height=320)

    if importance is not None and not importance.empty:
        st.markdown("### Feature Importance")
        st.caption("Top features contributing to the model output.")

        st.dataframe(importance.head(20), use_container_width=True, height=320)

        fig = px.bar(
            importance.head(15).sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Important Features"
        )
        fig = apply_chart_style(fig)
        fig.update_layout(title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    prediction_df = pd.DataFrame({
        "Actual": pd.Series(y_test).reset_index(drop=True),
        "Predicted": pd.Series(predictions).reset_index(drop=True)
    })

    st.markdown("### Prediction Results")
    st.caption("Preview of actual vs predicted labels.")
    st.dataframe(prediction_df.head(200), use_container_width=True, height=320)

    render_prediction_download(prediction_df, "classification_predictions.csv")
def render_automl_summary(results, best, problem, execution_time):
    st.markdown("### AutoML Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models Tested", len(results))
    c2.metric("Best Model", best["Model"])

    if problem == "Regression":
        c3.metric("Best R²", f"{best['R² Score']:.4f}")
    else:
        c3.metric("Best Accuracy", f"{best['Accuracy']:.4f}")

    c4.metric("Execution Time", f"{execution_time:.2f}s")
def render_automl_ranking(results, problem):
    score_column = "R² Score" if problem == "Regression" else "Accuracy"
    ranking_title = "Regression Model Comparison" if problem == "Regression" else "Classification Model Comparison"

    ranked_results = add_rank_column(results, score_column)

    st.markdown("### Model Ranking")
    st.dataframe(ranked_results, use_container_width=True, hide_index=True)

    fig = px.bar(
        ranked_results,
        x="Model",
        y=score_column,
        title=ranking_title,
        text=score_column
    )
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    return ranked_results

def show_machine_learning(df):
    st.header("Machine Learning Lab")
    st.caption("Train predictive models, compare algorithms, and review automated model recommendations.")
    numeric_targets = get_regression_targets(df)
    classification_targets = get_classification_targets(df)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">ML Workspace</div>
            <div class="insight-banner-text">
                {len(df):,} rows, {len(df.columns):,} columns,
                {len(numeric_targets)} regression-ready targets, and
                {len(classification_targets)} classification-ready targets detected.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(df.columns) > 150:
        st.markdown("""
        <div class="theme-warning-card">
            <div class="theme-warning-title">High-Dimensional Dataset</div>
            <div class="theme-warning-text">
                This dataset contains many columns. Model training and AutoML may take longer than usual.
            </div>
        </div>
        """, unsafe_allow_html=True)
    problem = st.radio(
        "Problem Type",
        ["Classification", "Regression"],
        horizontal=True
    )
    if problem == "Regression":
        if not numeric_targets:
            st.info("No numeric target columns are available for regression.")
            return

        left, right = st.columns(2)
        with left:
            target = st.selectbox("Target Column", numeric_targets)
        with right:
            algorithm = st.selectbox(
                "Regression Algorithm",
                ["Linear Regression", "Decision Tree"]
            )
    else:
        if not classification_targets:
            st.info("No suitable classification target columns were found. Classification targets should usually have a limited number of unique classes.")
            return
        left, right = st.columns(2)
        with left:
            target = st.selectbox("Target Column", classification_targets)
        with right:
            algorithm = st.selectbox(
                "Classification Algorithm",
                ["Logistic Regression", "Random Forest", "Decision Tree"]
            )
    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )
    b1, b2 = st.columns(2)
    train_clicked = b1.button("Train Model", type="primary", use_container_width=True)
    automl_clicked = b2.button("Run AutoML", use_container_width=True)
    if train_clicked:
        try:
            X_train, X_test, y_train, y_test = prepare_data(
                df,
                target,
                test_size=test_size / 100
            )
            if problem == "Regression":
                if algorithm == "Linear Regression":
                    model_name = "Linear Regression"
                    _, predictions, r2, mae, rmse = train_regression(
                        X_train, X_test, y_train, y_test
                    )
                else:
                    model_name = "Decision Tree Regressor"
                    _, predictions, r2, mae, rmse = train_decision_tree_regression(
                        X_train, X_test, y_train, y_test
                    )
                render_regression_results(model_name, y_test, predictions, r2, mae, rmse)
            else:
                if algorithm == "Logistic Regression":
                    model_name = "Logistic Regression"
                    _, predictions, accuracy, report, matrix = train_logistic(
                        X_train, X_test, y_train, y_test
                    )
                    importance = None
                elif algorithm == "Random Forest":
                    model_name = "Random Forest"
                    _, predictions, accuracy, report, matrix, importance = train_random_forest(
                        X_train, X_test, y_train, y_test
                    )
                else:
                    model_name = "Decision Tree Classifier"
                    _, predictions, accuracy, report, matrix, importance = train_decision_tree_classifier(
                        X_train, X_test, y_train, y_test
                    )
                render_classification_results(
                    model_name,
                    y_test,
                    predictions,
                    accuracy,
                    report,
                    matrix,
                    importance
                )
        except Exception as e:
            st.error(f"Model training failed: {e}")

    if automl_clicked:
        try:
            start = time.time()

            if problem == "Regression":
                results, best = automl_regression(
                    df,
                    target,
                    test_size=test_size / 100
                )
            else:
                results, best = automl_classification(
                    df,
                    target,
                    test_size=test_size / 100
                )
            execution_time = time.time() - start
            best_model_name = best["Model"]
            st.markdown(f"""
            <div class="insight-banner">
               <div>
                    <div class="insight-banner-title">Success</div>
                    <div class="insight-banner-text">
                        AutoML completed successfully. Best model: <strong>{best_model_name}</strong>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_automl_summary(results, best, problem, execution_time)
            ranked_results = render_automl_ranking(results, problem)
            if problem == "Regression":
                render_results_download(ranked_results, "regression_automl_report.csv")
            else:
                render_results_download(ranked_results, "classification_automl_report.csv")
            st.markdown("### AI AutoML Advisor")
            with st.spinner("Analyzing model performance..."):
                advisor_report = explain_automl_results(df, ranked_results, problem)
            st.markdown(advisor_report)
            if problem == "Regression":
                title, message, level = interpret_r2(best["R² Score"])
                st.markdown("### Best Model Interpretation")
                if level == "success":
                    st.success(message)
                elif level == "warning":
                    st.warning(message)
                else:
                    st.error(message)
        except Exception as e:
            st.error(f"AutoML failed: {e}")
