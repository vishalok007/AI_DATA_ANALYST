import numbers
import re
from difflib import SequenceMatcher

import pandas as pd
from utils.prompts import SYSTEM_PROMPT
from utils.ai_provider import generate_ai_response, generate_ai_response_with_source


STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "by", "with", "and",
    "is", "are", "was", "were", "be", "as", "at", "from", "show", "me",
    "what", "which", "who", "has", "have", "had", "this", "that", "these",
    "those", "dataset", "data", "column", "value", "values", "row", "rows"
}

MAX_KEYWORDS = ["highest", "largest", "maximum", "max", "most", "top", "biggest"]
MIN_KEYWORDS = ["lowest", "smallest", "minimum", "min", "least", "bottom"]
COUNT_KEYWORDS = ["how many", "count", "number of"]
NEGATIVE_HINTS = ["not", "did not", "didn't", "dont", "don't", "failed", "dead", "died", "lost"]
ENTITY_PRIORITY_HINTS = [
    "country", "territory", "name", "state", "province", "city", "region",
    "store", "branch", "capital", "continent", "category", "type"
]


def _normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _tokenize(value):
    return [token for token in _normalize_text(value).split() if token and token not in STOPWORDS]


def _token_similarity(left, right):
    return SequenceMatcher(None, str(left), str(right)).ratio()


def _format_value(value):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, numbers.Real):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):,}"
        return f"{numeric_value:,.4f}"

    return str(value)


def _question_direction(question):
    normalized = _normalize_text(question)

    if any(keyword in normalized for keyword in MAX_KEYWORDS):
        return "max"
    if any(keyword in normalized for keyword in MIN_KEYWORDS):
        return "min"
    return None


def _score_column_match(question, column_name):
    question_normalized = _normalize_text(question)
    column_normalized = _normalize_text(column_name)
    question_tokens = set(_tokenize(question))
    column_tokens = set(_tokenize(column_name))

    score = 0.0

    if column_normalized and column_normalized in question_normalized:
        score += 6.0

    overlap = question_tokens.intersection(column_tokens)
    score += len(overlap) * 3.0

    for column_token in column_tokens:
        if len(column_token) >= 4 and column_token in question_normalized:
            score += 1.5
        for question_token in question_tokens:
            if _token_similarity(question_token, column_token) >= 0.8:
                score += 1.2

    if question_tokens and column_tokens:
        score += SequenceMatcher(None, " ".join(sorted(question_tokens)), column_normalized).ratio()

    return score


def _find_metric_column(df, question):
    numeric_candidates = []

    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        numeric_candidates.append((column, _score_column_match(question, column), series))

    if not numeric_candidates:
        return None, None

    numeric_candidates.sort(key=lambda item: item[1], reverse=True)
    best_column, best_score, best_series = numeric_candidates[0]

    if best_score <= 0 and len(numeric_candidates) > 1:
        return None, None

    return best_column, best_series


def _find_entity_column(df, question, metric_column):
    question_normalized = _normalize_text(question)
    candidate_columns = [column for column in df.columns if column != metric_column]

    if not candidate_columns:
        return None

    preferred_terms = [term for term in ENTITY_PRIORITY_HINTS if term in question_normalized]

    def entity_score(column_name):
        normalized = _normalize_text(column_name)
        score = _score_column_match(question, column_name)

        for index, term in enumerate(ENTITY_PRIORITY_HINTS):
            if term in normalized:
                score += max(0.5, 4.0 - (index * 0.2))

        for term in preferred_terms:
            if term in normalized:
                score += 6.0

        dtype = df[column_name].dtype
        if dtype == "object" or str(dtype).startswith("string") or str(dtype) == "category":
            score += 2.0

        unique_values = df[column_name].nunique(dropna=True)
        if 1 < unique_values < max(len(df), 2):
            score += 0.5

        return score

    ranked = sorted(candidate_columns, key=entity_score, reverse=True)
    best_column = ranked[0]

    if entity_score(best_column) <= 0:
        return None

    return best_column


def _answer_extreme_question(df, question):
    direction = _question_direction(question)
    if direction is None:
        return None

    metric_column, metric_series = _find_metric_column(df, question)
    if metric_column is None or metric_series is None:
        return None

    valid_series = metric_series.dropna()
    if valid_series.empty:
        return None

    row_index = valid_series.idxmax() if direction == "max" else valid_series.idxmin()
    entity_column = _find_entity_column(df, question, metric_column)

    metric_value = df.loc[row_index, metric_column]
    entity_value = df.loc[row_index, entity_column] if entity_column else None

    comparator = "highest" if direction == "max" else "lowest"

    if entity_column and pd.notna(entity_value):
        return (
            "**Dataset-based answer (local analysis)**\n\n"
            f"The **{entity_value}** row has the **{comparator} {metric_column}** "
            f"in the dataset, with a value of **{_format_value(metric_value)}**.\n\n"
            f"- Entity column: `{entity_column}`\n"
            f"- Metric column: `{metric_column}`"
        )

    return (
        "**Dataset-based answer (local analysis)**\n\n"
        f"The **{comparator} {metric_column}** in the dataset is **{_format_value(metric_value)}**."
    )


def _answer_binary_count_question(df, question):
    normalized_question = _normalize_text(question)
    if not any(keyword in normalized_question for keyword in COUNT_KEYWORDS):
        return None

    best_column = None
    best_score = 0.0

    for column in df.columns:
        unique_values = set(pd.Series(df[column]).dropna().unique().tolist())
        if len(unique_values) == 0 or len(unique_values) > 12:
            continue

        score = _score_column_match(question, column)
        if score > best_score:
            best_score = score
            best_column = column

    if best_column is None or best_score < 0.8:
        return None

    series = df[best_column].dropna()
    numeric_series = pd.to_numeric(series, errors="coerce")
    unique_numeric = set(numeric_series.dropna().unique().tolist())

    if unique_numeric.issubset({0, 1}) and len(unique_numeric) > 0:
        positive_count = int((numeric_series == 1).sum())
        negative_count = int((numeric_series == 0).sum())
        target_count = negative_count if any(hint in normalized_question for hint in NEGATIVE_HINTS) else positive_count
        target_label = "0 values" if target_count == negative_count and any(hint in normalized_question for hint in NEGATIVE_HINTS) else "1 values"

        return (
            "**Dataset-based answer (local analysis)**\n\n"
            f"The dataset contains **{target_count:,}** records matching your question in the **{best_column}** column.\n\n"
            f"- Counted values: `{target_label}` in `{best_column}`\n"
            f"- Total rows reviewed: `{len(df):,}`"
        )

    return None


from datetime import datetime


def _build_dataset_prompt(summary, question, dataset_name):
    current_date_str = datetime.now().strftime("%A, %B %d, %Y")
    return f"""
{SYSTEM_PROMPT}
Current Live Date: {current_date_str}
Dataset Name: {dataset_name}

Dataset Context & Summary:
{summary}

User Question:
{question}

Instructions:
- If the user asks for today's date, time, or day, state clearly that today is {current_date_str}.
- If the question is about the dataset, provide an accurate answer based on the dataset summary.
- If the question is a dataset-specific quantitative query not present in the summary, reply in one line using:
  DATASET_UNAVAILABLE: <short reason>
- If the question is a general, external, or non-CSV question (such as date, coding, data science concepts, math), answer it directly and accurately.
"""


def _build_external_prompt(question, dataset_name):
    current_date_str = datetime.now().strftime("%A, %B %d, %Y")
    return f"""
{SYSTEM_PROMPT}
Current Live Date: {current_date_str}

User Question:
{question}

Instructions:
- Answer the user's question directly, accurately, and professionally.
- If the user asks for today's date, time, or day, state clearly that today is {current_date_str}.
- Keep the answer clear, helpful, and concise.
"""


def ask_ai(df, question, dataset_name="dataset"):
    direct_answer = _answer_extreme_question(df, question)
    if direct_answer:
        return direct_answer

    direct_count_answer = _answer_binary_count_question(df, question)
    if direct_count_answer:
        return direct_count_answer

    summary = create_dataset_summary(df)
    dataset_prompt = _build_dataset_prompt(summary, question, dataset_name)
    dataset_answer, dataset_provider = generate_ai_response_with_source(dataset_prompt)

    if str(dataset_answer).strip().upper().startswith("DATASET_UNAVAILABLE:"):
        external_prompt = _build_external_prompt(question, dataset_name)
        external_answer, external_provider = generate_ai_response_with_source(external_prompt)
        return str(external_answer).strip()

    return str(dataset_answer).strip()



def _low_cardinality_breakdown(df, max_unique=8, max_columns=6):
    sections = []
    inspected = 0

    for column in df.columns:
        if inspected >= max_columns:
            break

        series = df[column].dropna()
        unique_count = series.nunique()
        if unique_count == 0 or unique_count > max_unique:
            continue

        counts = series.value_counts().head(max_unique)
        sections.append(f"Value Counts for {column}:\n{counts.to_string()}")
        inspected += 1

    return "\n\n".join(sections)


def create_dataset_summary(df):
    preview_rows = df.head(5).to_markdown(index=False)
    low_cardinality = _low_cardinality_breakdown(df)
    summary = f"""
Dataset Shape:
{df.shape}
Columns:
{list(df.columns)}
Data Types:
{df.dtypes.to_string()}
Missing Values:
{df.isnull().sum().to_string()}
Statistics:
{df.describe(include='all').to_string()}
Sample Rows:
{preview_rows}
Low Cardinality Summaries:
{low_cardinality if low_cardinality else 'None'}
"""
    return summary


def generate_business_insights(df):
    summary = create_dataset_summary(df)
    prompt = f"""
{SYSTEM_PROMPT}
Dataset Information:
{summary}

Generate a professional report with the following sections.

1. Executive Summary
2. Business Insights
3. Data Quality Issues
4. Interesting Patterns
5. Recommended Charts
6. Machine Learning Suggestions
7. Actionable Recommendations
Use markdown headings and bullet points.
"""
    return generate_ai_response(prompt)


def recommend_charts(df):
    summary = create_dataset_summary(df)

    prompt = f"""
{SYSTEM_PROMPT}

Dataset Information

{summary}

Recommend the best charts for this dataset.

For each chart provide:
- Chart Name
- X-axis
- Y-axis
- Reason

Return in Markdown.
"""

    return generate_ai_response(prompt)


def explain_chart_recommendations(df):
    charts = recommend_charts(df)
    summary = create_dataset_summary(df)
    prompt = f"""
{SYSTEM_PROMPT}
Dataset Summary
{summary}
The following charts were recommended by the analytics engine.
{charts}
Explain:
1. Why each chart is useful.
2. Which business question it answers.
3. What insights users should look for.
4. Which audience would benefit
   (Manager, Analyst, Executive).
Return in Markdown.
"""
    return generate_ai_response(prompt)


def explain_automl_results(df, results_df, task_type):
    summary = create_dataset_summary(df)
    prompt = f"""
{SYSTEM_PROMPT}
Dataset Summary
{summary}
Task Type:
{task_type}
AutoML Results
{results_df.to_markdown(index=False)}
Generate a professional report.
Include:

# Best Model
# Model Comparison
# Performance Evaluation
# Overfitting / Underfitting Risk
# Business Impact
# Recommendations

Use Markdown headings and bullet points.
"""
    return generate_ai_response(prompt)


def generate_cleaning_advice(df):
    summary = create_dataset_summary(df)

    prompt = f"""
{SYSTEM_PROMPT}

Dataset Summary

{summary}

You are a Senior Data Quality Consultant.

Analyze the dataset and prepare a professional report.

Include:

# Missing Values
- Which columns need attention?
- Should values be removed or imputed?

# Duplicate Records
- Should duplicates be removed?
- Possible impact

# Outliers
- Which numerical columns are likely to contain outliers?
- Recommended treatment

# Data Types
- Any incorrect data types?
- Recommended conversions

# Feature Engineering
Suggest useful new features if applicable.

# Data Quality Score
Rate the dataset out of 100.

# Priority Actions
List the cleaning tasks from highest to lowest priority.
Return the report in Markdown.
"""
    return generate_ai_response(prompt)


def generate_feature_engineering_advice(df):
    summary = create_dataset_summary(df)

    prompt = f"""
{SYSTEM_PROMPT}

Dataset Summary

{summary}

You are an expert Machine Learning Engineer.

Analyze the dataset and prepare a Feature Engineering report.

Include the following sections.

# Date Features
Suggest useful date-derived features if date columns exist.

# Numerical Features
Suggest interaction terms, ratios, or polynomial features.

# Categorical Features
Recommend encoding techniques:
- One-Hot Encoding
- Label Encoding
- Target Encoding
Explain which is appropriate.

# Feature Scaling
Recommend:
- StandardScaler
- MinMaxScaler
- RobustScaler
Explain why.

# Feature Selection
Identify features that may be informative or redundant.

# Features to Remove
Mention columns that are likely to add little value.

# Expected Impact
Explain how the recommendations could improve model performance.

Return the answer in Markdown.
"""
    return generate_ai_response(prompt)
