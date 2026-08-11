import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def preprocess_raw_features(X):
    """Clean raw feature data types before splitting without fitting imputers or scalers."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        elif X[col].dtype == "object":
            # Attempt datetime parsing
            try:
                dt = pd.to_datetime(X[col], dayfirst=True, errors="raise")
                X[col] = dt.astype("int64") // 10**9
            except Exception:
                pass
    return X


def prepare_data(df, target, test_size=0.2):
    """Safely split raw dataset into train and test sets to prevent data leakage."""
    df = df.dropna(subset=[target]).copy()

    X = df.drop(columns=[target])
    y = df[target]

    X = preprocess_raw_features(X)

    # Encode target if object or categorical
    if y.dtype == "object" or str(y.dtype) == "category":
        try:
            dt_y = pd.to_datetime(y, dayfirst=True, errors="raise")
            y = dt_y.astype("int64") // 10**9
        except Exception:
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

    return train_test_split(X, y, test_size=test_size, random_state=42)


def build_preprocessor(X_train, is_linear=False):
    """Build a ColumnTransformer for numeric and categorical features."""
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    transformers = []

    if numeric_cols:
        if is_linear:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
        else:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        if is_linear:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        else:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def train_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression model with standard scaling and one-hot encoding."""
    preprocessor = build_preprocessor(X_train, is_linear=True)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return pipeline, predictions, r2, mae, rmse


def train_decision_tree_regression(X_train, X_test, y_train, y_test):
    """Train Decision Tree Regressor with ordinal encoding."""
    preprocessor = build_preprocessor(X_train, is_linear=False)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return pipeline, predictions, r2, mae, rmse


def train_logistic(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with standard scaling and one-hot encoding."""
    preprocessor = build_preprocessor(X_train, is_linear=True)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    return pipeline, predictions, accuracy, report, matrix


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest Classifier with ordinal encoding and feature importance calculation."""
    preprocessor = build_preprocessor(X_train, is_linear=False)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return pipeline, predictions, accuracy, report, matrix, importance


def train_decision_tree_classifier(X_train, X_test, y_train, y_test):
    """Train Decision Tree Classifier with ordinal encoding and feature importance calculation."""
    preprocessor = build_preprocessor(X_train, is_linear=False)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return pipeline, predictions, accuracy, report, matrix, importance