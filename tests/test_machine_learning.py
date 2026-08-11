import numpy as np
import pandas as pd
from utils.automl import automl_classification, automl_regression
from utils.machine_learning import (
    prepare_data,
    train_decision_tree_classifier,
    train_decision_tree_regression,
    train_logistic,
    train_random_forest,
    train_regression,
)


def create_sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feat_num": np.random.randn(n),
        "feat_cat": np.random.choice(["X", "Y", "Z"], size=n),
        "target_num": np.random.randn(n) * 5 + 10,
        "target_cat": np.random.choice(["ClassA", "ClassB"], size=n),
    })


def test_prepare_data():
    df = create_sample_df()
    X_tr, X_te, y_tr, y_te = prepare_data(df, "target_num")
    assert len(X_tr) + len(X_te) == 100
    assert "target_num" not in X_tr.columns


def test_regression_models():
    df = create_sample_df()
    X_tr, X_te, y_tr, y_te = prepare_data(df, "target_num")

    _, preds_lin, r2_lin, mae_lin, rmse_lin = train_regression(X_tr, X_te, y_tr, y_te)
    assert len(preds_lin) == len(y_te)
    assert isinstance(r2_lin, float)

    _, preds_dt, r2_dt, mae_dt, rmse_dt = train_decision_tree_regression(X_tr, X_te, y_tr, y_te)
    assert len(preds_dt) == len(y_te)

    res, best = automl_regression(df, "target_num")
    assert len(res) == 2
    assert "Model" in best.index


def test_classification_models():
    df = create_sample_df()
    X_tr, X_te, y_tr, y_te = prepare_data(df, "target_cat")

    _, preds_log, acc_log, _, _ = train_logistic(X_tr, X_te, y_tr, y_te)
    assert 0.0 <= acc_log <= 1.0

    _, preds_rf, acc_rf, _, _, imp_rf = train_random_forest(X_tr, X_te, y_tr, y_te)
    assert not imp_rf.empty

    _, preds_dt, acc_dt, _, _, imp_dt = train_decision_tree_classifier(X_tr, X_te, y_tr, y_te)
    assert not imp_dt.empty

    res, best = automl_classification(df, "target_cat")
    assert len(res) == 3
    assert "Model" in best.index
