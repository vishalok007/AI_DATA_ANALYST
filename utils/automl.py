import pandas as pd

from utils.machine_learning import (
    prepare_data,
    train_decision_tree_classifier,
    train_decision_tree_regression,
    train_logistic,
    train_random_forest,
    train_regression,
)


def automl_regression(df, target, test_size=0.2):
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target,
        test_size,
    )

    results = []

    _, _, r2_lin, mae_lin, rmse_lin = train_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append({
        "Model": "Linear Regression",
        "R² Score": r2_lin,
        "MAE": mae_lin,
        "RMSE": rmse_lin,
    })

    _, _, r2_dt, mae_dt, rmse_dt = train_decision_tree_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append({
        "Model": "Decision Tree",
        "R² Score": r2_dt,
        "MAE": mae_dt,
        "RMSE": rmse_dt,
    })

    result_df = pd.DataFrame(results)

    best = result_df.sort_values(
        "R² Score",
        ascending=False,
    ).iloc[0]

    return result_df, best


def automl_classification(df, target, test_size=0.2):
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target,
        test_size,
    )

    results = []

    _, _, acc_log, report_log, _ = train_logistic(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    f1_log = report_log.get("weighted avg", {}).get("f1-score", acc_log)
    results.append({
        "Model": "Logistic Regression",
        "Accuracy": acc_log,
        "F1 Score": f1_log,
    })

    _, _, acc_rf, report_rf, _, _ = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    f1_rf = report_rf.get("weighted avg", {}).get("f1-score", acc_rf)
    results.append({
        "Model": "Random Forest",
        "Accuracy": acc_rf,
        "F1 Score": f1_rf,
    })

    _, _, acc_dt, report_dt, _, _ = train_decision_tree_classifier(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    f1_dt = report_dt.get("weighted avg", {}).get("f1-score", acc_dt)
    results.append({
        "Model": "Decision Tree",
        "Accuracy": acc_dt,
        "F1 Score": f1_dt,
    })

    result_df = pd.DataFrame(results)

    best = result_df.sort_values(
        "Accuracy",
        ascending=False,
    ).iloc[0]

    return result_df, best


def interpret_r2(r2):
    if r2 >= 0.90:
        return (
            "Excellent Model",
            "The model explains more than 90% of the variation in the target variable. This is an excellent predictive model suitable for most real-world applications.",
            "success",
        )
    elif r2 >= 0.75:
        return (
            "Very Good Model",
            "The model explains most of the variation in the target variable and is expected to perform well on unseen data.",
            "success",
        )
    elif r2 >= 0.50:
        return (
            "Moderate Model",
            "The model captures a reasonable amount of the variation, but there is still room for improvement through better features or algorithms.",
            "warning",
        )
    elif r2 >= 0.25:
        return (
            "Weak Model",
            "The model has limited predictive ability. Consider feature engineering, data cleaning, or trying different algorithms.",
            "warning",
        )
    else:
        return (
            "Poor Model",
            "The model explains very little of the target variation. Review the target variable, improve preprocessing, add better features, or consider another algorithm.",
            "error",
        )