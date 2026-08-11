import pandas as pd
import numpy as np
from utils.datatype import is_id_column


def get_auto_recommendations(df, profile=None):
    """
    Intelligently analyze dataset features to automatically recommend
    the most suitable visualizations upon CSV upload.
    """
    if df is None or df.empty:
        return []

    # Get non-ID numeric columns
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not is_id_column(c, df)
    ]

    # Get suitable categorical columns (low to medium cardinality)
    categorical_cols = []
    for c in df.select_dtypes(exclude="number").columns:
        if not is_id_column(c, df):
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 20:
                categorical_cols.append(c)

    # Get datetime columns
    datetime_cols = []
    for c in df.columns:
        if "date" in c.lower() or c.lower().endswith("_dt"):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() >= 0.5:
                    datetime_cols.append(c)
            except Exception:
                pass

    recommendations = []

    # -------------------------------------------------------------
    # 1. Feature Distribution Recommendation (Highest Variance Column)
    # -------------------------------------------------------------
    if numeric_cols:
        variances = df[numeric_cols].var(numeric_only=True).dropna().sort_values(ascending=False)
        top_num = variances.index[0] if not variances.empty else numeric_cols[0]
        
        recommendations.append({
            "id": "distribution_rec",
            "title": f"Distribution Analysis of {top_num}",
            "chart_type": "Histogram & Violin",
            "badge": "Highest Variance Feature",
            "reason": f"'{top_num}' shows significant numerical variance across records. Ideal for detecting skewness, spread, and natural clusters.",
            "x_col": top_num,
            "category": "Distribution"
        })

    # -------------------------------------------------------------
    # 2. Category Breakdown Recommendation (Top Category vs Top Numeric)
    # -------------------------------------------------------------
    if categorical_cols and numeric_cols:
        top_cat = categorical_cols[0]
        top_num = numeric_cols[0]
        
        # Pick category with cleanest group distribution
        cat_uniques = {c: df[c].nunique() for c in categorical_cols}
        best_cat = min(cat_uniques, key=lambda k: abs(cat_uniques[k] - 6))
        
        recommendations.append({
            "id": "category_rec",
            "title": f"Category Breakdown: {top_num} by {best_cat}",
            "chart_type": "Bar Chart & Donut",
            "badge": "Top Category Comparison",
            "reason": f"Comparing '{top_num}' across '{best_cat}' reveals group patterns, category performance, and structural proportions.",
            "x_col": best_cat,
            "y_col": top_num,
            "agg": "Mean",
            "category": "Category Comparison"
        })

    # -------------------------------------------------------------
    # 3. Bivariate Relationship Recommendation (Correlated Pair)
    # -------------------------------------------------------------
    if len(numeric_cols) >= 2:
        num1 = numeric_cols[0]
        num2 = numeric_cols[1]
        
        # Find highest absolute correlation pair if possible
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            if not corr_matrix.empty and corr_matrix.max().max() > 0:
                max_corr_pair = corr_matrix.unstack().idxmax()
                num1, num2 = max_corr_pair[0], max_corr_pair[1]
        except Exception:
            pass

        color_col = categorical_cols[0] if categorical_cols else None

        recommendations.append({
            "id": "bivariate_rec",
            "title": f"Bivariate Association: {num2} vs {num1}",
            "chart_type": "Scatter Plot with Color Grouping",
            "badge": "Strongest Associated Features",
            "reason": f"'{num2}' and '{num1}' exhibit linear or non-linear co-variation. Useful for identifying correlation, outliers, and trend lines.",
            "x_col": num1,
            "y_col": num2,
            "color_col": color_col,
            "category": "Relationship"
        })

    # -------------------------------------------------------------
    # 4. Time Series Recommendation (If Date Present)
    # -------------------------------------------------------------
    if datetime_cols and numeric_cols:
        top_date = datetime_cols[0]
        top_num = numeric_cols[0]

        recommendations.append({
            "id": "trend_rec",
            "title": f"Time Trend Analysis: {top_num} Over {top_date}",
            "chart_type": "Interactive Line Chart",
            "badge": "Temporal Signal Detected",
            "reason": f"Tracks historical shifts, seasonality, and long-term momentum of '{top_num}' over time.",
            "x_col": top_date,
            "y_col": top_num,
            "category": "Time Trend"
        })

    return recommendations


def recommend_charts(profile):
    """
    Backwards compatible recommender list.
    """
    recommendations = []
    numeric = profile.get("numeric_columns", [])
    categorical = profile.get("categorical_columns", [])
    datetime = profile.get("datetime_columns", [])

    if len(numeric) >= 1:
        recommendations.append("histogram")
    if len(numeric) >= 2:
        recommendations.append("scatter")
    if len(numeric) >= 1 and len(categorical) >= 1:
        recommendations.append("bar")
    if len(datetime) >= 1 and len(numeric) >= 1:
        recommendations.append("line")
    if len(numeric) >= 2:
        recommendations.append("correlation")

    return recommendations