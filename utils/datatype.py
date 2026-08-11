import pandas as pd
def is_id_column(col, df=None):
    """
    Check if a column is a Row Identifier, Primary Key, or Index column.
    """
    if col is None:
        return False
    col_str = str(col).strip()
    col_lower = col_str.lower()

    # Exact or common ID column name patterns
    if col_lower in ["id", "id_", "uuid", "guid", "index", "unnamed: 0", "row_id", "rowid", "row_num", "rownum", "serial", "pk", "key"]:
        return True

    if col_lower.endswith("_id") or col_lower.startswith("id_"):
        return True

    # Known ID column names (e.g. PassengerId, UserId, CustomerId, OrderId, etc.)
    id_terms = ["passengerid", "userid", "customerid", "orderid", "productid", "itemid", "employeeid", "transactionid", "ticketid", "invoiceid"]
    if any(term in col_lower for term in id_terms):
        return True

    # Sequential integer check: 1 to N values
    if df is not None and len(df) > 5 and col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_vals = df[col].dropna()
            if len(valid_vals) == len(df) and df[col].nunique() == len(df):
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                if min_val in [0, 1] and (max_val - min_val + 1) == len(df):
                    return True

    return False


def detect_column_types(df):
    """
    Detect intelligent column types.
    """
    result = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "identifier": [],
        "text": []
    }
    for column in df.columns:
        series = df[column]

        # Identifier check using enhanced is_id_column helper
        if is_id_column(column, df):
            result["identifier"].append(column)
            continue

        name = column.lower()
        if (
            "date" in name
            or name.endswith("_dt")
            or name.endswith("date")
        ):
            result["datetime"].append(column)
            continue
        # Boolean
        unique = set(series.dropna().unique())
        if unique.issubset(
            {0, 1, True, False, "Yes", "No", "yes", "no"}
        ):
            result["boolean"].append(column)
            continue
        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            result["numeric"].append(column)
            continue
        if series.dtype == "object":
            avg_len = series.astype(str).str.len().mean()
            if avg_len > 40:
                result["text"].append(column)
            else:
                result["categorical"].append(column)
    return result