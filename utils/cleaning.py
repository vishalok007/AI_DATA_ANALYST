import pandas as pd


def remove_duplicates(df):

    cleaned_df = df.drop_duplicates()

    removed_rows = len(df) - len(cleaned_df)

    return cleaned_df, removed_rows
def drop_missing_values(df):

    cleaned_df = df.dropna()

    removed_rows = len(df) - len(cleaned_df)

    return cleaned_df, removed_rows
def fill_mean(df, column):

    cleaned_df = df.copy()

    cleaned_df[column] = cleaned_df[column].fillna(
        cleaned_df[column].mean()
    )

    return cleaned_df

def fill_median(df, column):

    cleaned_df = df.copy()

    cleaned_df[column] = cleaned_df[column].fillna(
        cleaned_df[column].median()
    )

    return cleaned_df
def fill_mode(df, column):

    cleaned_df = df.copy()

    cleaned_df[column] = cleaned_df[column].fillna(
        cleaned_df[column].mode()[0]
    )

    return cleaned_df
def fill_constant(df, column, value):

    cleaned_df = df.copy()

    cleaned_df[column] = cleaned_df[column].fillna(value)

    return cleaned_df
def drop_columns(df, columns):

    cleaned_df = df.drop(columns=columns)

    return cleaned_df

def convert_dtype(df, column, dtype):

    cleaned_df = df.copy()

    try:

        if dtype == "Numeric":

            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )

        elif dtype == "String":

            cleaned_df[column] = cleaned_df[column].astype(str)

        elif dtype == "Datetime":

            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                errors="coerce"
            )

        elif dtype == "Boolean":

            cleaned_df[column] = cleaned_df[column].astype(bool)

    except Exception as e:

        return df, str(e)

    return cleaned_df, None