"""
AI Data Analyst — PDF Report Generator (rebuilt)
==================================================
Produced from the actually-loaded dataset in the AI Chat tab.

Sections populated (every page is filled, no blank pages):
  1. Cover page               (title, dataset name, date, key counts)
  2. Executive summary        (rows, columns, types, completeness, score)
  3. Dataset overview         (column-by-column snapshot table)
  4. Data quality report      (missing %, duplicates, quality score)
  5. Statistical analysis     (per-column min/max/mean/std/median)
  6. Categorical breakdown    (top categories with share %)
  7. Correlation highlights   (top + and - numeric correlations)
  8. Business findings &      (auto-generated, deterministic
     recommendations           comments + prioritised actions)
"""

from datetime import datetime
import pandas as pd
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


# ---------------------------------------------------------------------------
# Brand palette aligned with the Streamlit app
# ---------------------------------------------------------------------------
BRAND_PRIMARY = HexColor("#6D28D9")   # deep purple
BRAND_SECONDARY = HexColor("#2563EB") # electric blue
BRAND_ACCENT = HexColor("#22C55E")     # green
BRAND_DANGER = HexColor("#EF4444")     # red
BRAND_WARN = HexColor("#F59E0B")        # amber
BRAND_TEXT = HexColor("#0F172A")
BRAND_MUTED = HexColor("#475569")
TABLE_HEAD_BG = HexColor("#EEF2FF")
TABLE_ALT_BG = HexColor("#F8FAFC")
SOFT_BORDER = HexColor("#CBD5E1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            textColor=BRAND_PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=BRAND_SECONDARY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=BRAND_PRIMARY,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BRAND_SECONDARY,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=BRAND_TEXT,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=BRAND_TEXT,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=BRAND_MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _stylise_table(table, header_bg=BRAND_PRIMARY, alt_bg=TABLE_ALT_BG,
                   head_text_color=colors.white, col_widths=None):
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), head_text_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, SOFT_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, alt_bg]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ]
    if col_widths:
        pass
    table.setStyle(TableStyle(style_cmds))
    return table


def _add_section_heading(story, styles, title, intro=None):
    if story:
        story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(title, styles["h1"]))
    if intro:
        story.append(Paragraph(intro, styles["body"]))


def _safe_format(value, fmt="{:,.2f}"):
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return fmt.format(value)
    return str(value)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_pdf_report(df, output_path, dataset_name="Dataset"):
    """Build a structured, populated PDF report from the dataframe."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.7 * cm,
        title=f"AI Data Analyst Report — {dataset_name}",
        author="AI Data Analyst Platform",
    )

    styles = _build_styles()
    story = []

    add_cover_page(story, styles, df, dataset_name)
    add_executive_summary(story, styles, df, dataset_name)
    add_dataset_overview(story, styles, df)
    add_data_quality_section(story, styles, df)
    add_statistical_analysis(story, styles, df)
    add_categorical_breakdown(story, styles, df)
    add_correlation_section(story, styles, df)
    add_business_findings(story, styles, df, dataset_name)
    add_recommendations(story, styles, df, dataset_name)

    _add_footer_to_pages(doc, dataset_name)

    doc.build(story)


def _add_footer_to_pages(doc, dataset_name):
    def on_page(canvas, _doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(BRAND_MUTED)
        canvas.drawString(
            2 * cm, 1.0 * cm,
            f"AI Data Analyst Report — {dataset_name}",
        )
        canvas.drawRightString(
            A4[0] - 2 * cm, 1.0 * cm,
            f"Page {_doc_.page}",
        )
        canvas.setStrokeColor(SOFT_BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
        canvas.restoreState()

    doc.onLaterPages = on_page
    doc.onFirstPage = on_page


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------
def add_cover_page(story, styles, df, dataset_name):
    story.append(Spacer(1, 0.4 * cm))
    brand_strip = Table(
        [[Paragraph(
            "<font color='#FFFFFF'><b>AI DATA ANALYST</b></font>",
            ParagraphStyle("brand", parent=styles["body"],
                           alignment=TA_CENTER, textColor=colors.white,
                           fontSize=11)
        )]],
        colWidths=[17 * cm],
        rowHeights=[1.0 * cm],
    )
    brand_strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_PRIMARY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(brand_strip)
    story.append(Spacer(1, 1.2 * cm))

    story.append(Paragraph("AI DATA ANALYST REPORT", styles["title"]))
    story.append(Paragraph("Automated analytical summary", styles["subtitle"]))
    story.append(Spacer(1, 0.7 * cm))

    hero_table = Table(
        [
            ["Dataset", dataset_name],
            ["Prepared by", "AI Data Analyst Platform"],
            ["Generated on", datetime.now().strftime("%d %B %Y, %I:%M %p")],
            ["Rows", f"{df.shape[0]:,}"],
            ["Columns", f"{df.shape[1]}"],
            ["Memory footprint",
             f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"],
        ],
        colWidths=[6 * cm, 11 * cm],
        rowHeights=[0.85 * cm] * 6,
    )
    hero_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), TABLE_HEAD_BG),
        ("TEXTCOLOR", (0, 0), (0, -1), BRAND_PRIMARY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, SOFT_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(hero_table)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "This report is generated automatically from the active dataset. "
        "Every section below has been filled using the actual data shape, "
        "the per-column profile, the missing-value footprint, and a "
        "deterministic statistical analysis.",
        styles["callout_body"],
    ))


# ---------------------------------------------------------------------------
# Page 2 — Executive Summary
# ---------------------------------------------------------------------------
def add_executive_summary(story, styles, df, dataset_name):
    story.append(PageBreak())
    story.append(Paragraph("Executive Summary", styles["h1"]))

    rows, cols = df.shape
    numerical = len(df.select_dtypes(include="number").columns)
    categorical = len(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_cols = len(df.select_dtypes(include=["datetime"]).columns)
    missing_cells = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    completeness = round(
        ((rows * cols) - missing_cells) / max(rows * cols, 1) * 100, 2
    )
    quality_score = max(0, min(100, completeness - duplicates))

    story.append(Paragraph(
        f"This report covers the dataset <b>{dataset_name}</b>, which contains "
        f"{rows:,} rows and {cols} columns. Completeness stands at "
        f"<b>{completeness}%</b>, the overall data-quality score is "
        f"<b>{quality_score:.1f}/100</b>, and {duplicates} duplicate rows were "
        f"detected. The table below summarises the structural profile of the "
        f"dataset.",
        styles["body"],
    ))

    story.append(Spacer(1, 0.3 * cm))
    summary_rows = [
        ["Metric", "Value", "Interpretation"],
        ["Rows", f"{rows:,}", "Total records"],
        ["Columns", f"{cols}", "Total fields"],
        ["Numerical columns", f"{numerical}", "Suitable for correlation & ML"],
        ["Categorical columns", f"{categorical}", "May need encoding"],
        ["Datetime columns", f"{datetime_cols}", "Can yield time features"],
        ["Missing cells", f"{missing_cells:,}",
         f"{100 - completeness:.2f}% of cells" if rows * cols else "—"],
        ["Duplicate rows", f"{duplicates}", "Should be reviewed"],
        ["Completeness", f"{completeness:.2f}%", "Non-null cell share"],
        ["Data Quality Score", f"{quality_score:.1f}/100", "Composite of above"],
        ["Memory footprint",
         f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB", "In-memory size"],
    ]
    summary_table = Table(
        summary_rows,
        colWidths=[4.6 * cm, 3.6 * cm, 7.8 * cm],
        repeatRows=1,
    )
    _stylise_table(summary_table)
    story.append(summary_table)

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Column list", styles["h2"]))
    column_text = ", ".join([f"<b>{c}</b>" for c in df.columns])
    story.append(Paragraph(column_text, styles["body"]))


# ---------------------------------------------------------------------------
# Page 3 — Dataset overview (column-by-column snapshot)
# ---------------------------------------------------------------------------
def add_dataset_overview(story, styles, df):
    story.append(PageBreak())
    _add_section_heading(
        story,
        styles,
        "Dataset Overview",
        "Snapshot of every column: stored type, non-null count, share of "
        "missing values, number of unique values and a sample of the most "
        "frequent category. Rows are limited to the first 30 columns for "
        "readability; remaining columns are summarised at the end.",
    )

    header = ["#", "Column", "Dtype", "Non-null", "Missing %", "Unique", "Top value"]
    rows_data = [header]
    limited_cols = list(df.columns[:30])
    for idx, col in enumerate(limited_cols, start=1):
        s = df[col]
        non_null = int(s.notna().sum())
        missing_pct = (1 - non_null / max(len(s), 1)) * 100
        try:
            nunique = int(s.nunique(dropna=True))
        except TypeError:
            nunique = int(s.astype(str).nunique())
        try:
            top_value = s.dropna().value_counts().idxmax()
            top_value = str(top_value)
            if len(top_value) > 28:
                top_value = top_value[:25] + "…"
        except Exception:
            top_value = "—"
        try:
            dtype = str(s.dtype)
        except Exception:
            dtype = "—"
        rows_data.append([
            str(idx), str(col), dtype,
            f"{non_null:,}", f"{missing_pct:.2f}%",
            f"{nunique:,}", top_value,
        ])

    overview_table = Table(
        rows_data,
        colWidths=[0.9 * cm, 4.0 * cm, 2.3 * cm, 1.9 * cm, 1.9 * cm, 1.5 * cm, 3.3 * cm],
        repeatRows=1,
    )
    _stylise_table(overview_table)
    story.append(overview_table)

    if len(df.columns) > 30:
        remaining = ", ".join(str(c) for c in df.columns[30:])
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "<b>Additional columns (truncated from this page):</b> "
            + remaining,
            styles["body"],
        ))


# ---------------------------------------------------------------------------
# Page 4 — Data Quality
# ---------------------------------------------------------------------------
def add_data_quality_section(story, styles, df):
    _add_section_heading(
        story,
        styles,
        "Data Quality Report",
        "Quality is broken down into duplicates, types, and missing values. "
        "The score uses a simple composite: completeness minus duplicate "
        "rows, clamped between 0 and 100.",
    )

    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    quality_score = max(0, min(100, ((rows * cols) - missing)
                               / max(rows * cols, 1) * 100 - duplicates))

    rows_data = [
        ["Metric", "Value", "Status"],
        ["Total cells", f"{rows * cols:,}", "—"],
        ["Missing cells", f"{missing:,}",
         "Healthy" if missing == 0 else "Needs cleanup"],
        ["Missing share",
         f"{(missing / max(rows * cols, 1) * 100):.2f}%",
         "Healthy" if missing / max(rows * cols, 1) * 100 < 5 else "Watch"],
        ["Duplicate rows", f"{duplicates}",
         "Healthy" if duplicates == 0 else "Review"],
        ["Data Quality Score",
         f"{quality_score:.1f}/100",
         "Excellent" if quality_score >= 90
         else "Good" if quality_score >= 75
         else "Needs work"],
    ]
    quality_table = Table(
        rows_data,
        colWidths=[4.8 * cm, 3.4 * cm, 8 * cm],
        repeatRows=1,
    )
    _stylise_table(quality_table,
                   header_bg=BRAND_SECONDARY,
                   alt_bg=TABLE_ALT_BG)
    story.append(quality_table)

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Missing values by column", styles["h2"]))

    missing_series = df.isnull().sum().sort_values(ascending=False)
    rows_data = [["Column", "Missing", "Missing %", "Severity"]]
    for col, count in missing_series.items():
        pct = count / max(len(df), 1) * 100
        if count == 0:
            severity = "✓ Clean"
            severity_color = BRAND_ACCENT
        elif pct < 5:
            severity = "Low"
            severity_color = BRAND_WARN
        elif pct < 20:
            severity = "Moderate"
            severity_color = HexColor("#F97316")
        else:
            severity = "High"
            severity_color = BRAND_DANGER
        rows_data.append([str(col), f"{int(count):,}",
                          f"{pct:.2f}%", severity])

    missing_table = Table(
        rows_data,
        colWidths=[5.4 * cm, 2.7 * cm, 2.7 * cm, 4.9 * cm],
        repeatRows=1,
    )
    _stylise_table(missing_table)
    style_overrides = []
    for i, (_, _, _, sev) in enumerate(rows_data[1:], start=1):
        if sev == "✓ Clean":
            style_overrides.append(("TEXTCOLOR", (3, i), (3, i), BRAND_ACCENT))
        elif sev == "Moderate":
            style_overrides.append(("TEXTCOLOR", (3, i), (3, i),
                                   HexColor("#F97316")))
        elif sev == "High":
            style_overrides.append(("TEXTCOLOR", (3, i), (3, i), BRAND_DANGER))
        elif sev in ("Watch", "Low"):
            style_overrides.append(("TEXTCOLOR", (3, i), (3, i), BRAND_WARN))
    if style_overrides:
        missing_table.setStyle(TableStyle(style_overrides))
    story.append(missing_table)


# ---------------------------------------------------------------------------
# Page 5 — Statistical analysis
# ---------------------------------------------------------------------------
def add_statistical_analysis(story, styles, df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        _add_section_heading(
            story,
            styles,
            "Statistical Analysis",
            "No numeric columns were found in this dataset, so descriptive "
            "statistics cannot be generated. Categorical-only datasets still "
            "benefit from the categorical breakdown, correlation, and "
            "business findings sections.",
        )
        return

    _add_section_heading(
        story,
        styles,
        "Statistical Analysis",
        f"Descriptive statistics for the {numeric_df.shape[1]} numeric "
        f"column(s) in the dataset.",
    )

    rows_data = [["Column", "Count", "Mean", "Median", "Std", "Min", "Max"]]
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if s.empty:
            rows_data.append([str(col), "0", "—", "—", "—", "—", "—"])
            continue
        rows_data.append([
            str(col),
            f"{int(s.count()):,}",
            _safe_format(s.mean(), "{:,.2f}"),
            _safe_format(s.median(), "{:,.2f}"),
            _safe_format(s.std(), "{:,.2f}"),
            _safe_format(s.min(), "{:,.2f}"),
            _safe_format(s.max(), "{:,.2f}"),
        ])

    stat_table = Table(
        rows_data,
        colWidths=[3.8 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm,
                   2.1 * cm, 2.2 * cm, 2.2 * cm],
        repeatRows=1,
    )
    _stylise_table(stat_table, header_bg=BRAND_SECONDARY)
    story.append(stat_table)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>How to read this:</b> Mean and median diverge when the column "
        "contains outliers. A standard deviation larger than half of the mean "
        "suggests high spread and likely outliers.",
        styles["body"],
    ))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Outlier candidates (IQR rule)", styles["h2"]))
    outlier_rows = [["Column", "Lower bound", "Upper bound",
                    "Outliers", "Outlier %"]]
    found_any = False
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        if outliers > 0:
            found_any = True
            outlier_rows.append([
                str(col),
                _safe_format(lower, "{:,.2f}"),
                _safe_format(upper, "{:,.2f}"),
                f"{int(outliers):,}",
                f"{outliers / max(len(s), 1) * 100:.2f}%",
            ])
    if not found_any:
        outlier_rows.append(["—", "—", "—", "0", "0.00%"])
    outlier_table = Table(
        outlier_rows,
        colWidths=[4.2 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 3.2 * cm],
        repeatRows=1,
    )
    _stylise_table(outlier_table, header_bg=BRAND_WARN)
    story.append(outlier_table)


# ---------------------------------------------------------------------------
# Page 6 — Categorical breakdown
# ---------------------------------------------------------------------------
def add_categorical_breakdown(story, styles, df):
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    _add_section_heading(story, styles, "Categorical Breakdown")
    if cat_df.shape[1] == 0:
        story.append(Paragraph(
            "No categorical columns were detected in this dataset. The "
            "categorical breakdown is therefore empty.",
            styles["body"],
        ))
        return
    story.append(Paragraph(
        f"Top categories for each of the {cat_df.shape[1]} categorical "
        f"column(s). Use this to detect dominant classes, rare classes and "
        f"encoding candidates.",
        styles["body"],
    ))
    for col in cat_df.columns:
        story.append(Paragraph(
            f"<b>{col}</b> "
            f"<font size='8.5' color='#475569'>"
            f"({df[col].nunique(dropna=True)} unique, "
            f"{int(df[col].isnull().sum())} missing)</font>",
            styles["h2"],
        ))
        try:
            vc = df[col].value_counts(dropna=True).head(8)
        except TypeError:
            vc = df[col].astype(str).value_counts(dropna=True).head(8)
        rows_data = [["Rank", "Value", "Count", "Share %"]]
        total = max(vc.sum(), 1)
        for rank, (value, count) in enumerate(vc.items(), start=1):
            value_text = str(value)
            if len(value_text) > 32:
                value_text = value_text[:29] + "…"
            rows_data.append([
                str(rank), value_text, f"{int(count):,}",
                f"{count / total * 100:.2f}%"
            ])
        cat_table = Table(
            rows_data,
            colWidths=[1.4 * cm, 6.2 * cm, 2.7 * cm, 3.4 * cm],
            repeatRows=1,
        )
        _stylise_table(cat_table, header_bg=BRAND_SECONDARY)
        story.append(cat_table)
        story.append(Spacer(1, 0.3 * cm))


# ---------------------------------------------------------------------------
# Page 7 — Correlation highlights
# ---------------------------------------------------------------------------
def add_correlation_section(story, styles, df):
    numeric_df = df.select_dtypes(include="number")
    _add_section_heading(story, styles, "Correlation Highlights")
    if numeric_df.shape[1] < 2:
        story.append(Paragraph(
            "At least two numeric columns are required to compute "
            "correlations. This dataset does not meet that requirement.",
            styles["body"],
        ))
        return
    corr = numeric_df.corr(numeric_only=True)
    story.append(Paragraph(
        f"Computed across the {numeric_df.shape[1]} numeric columns. The "
        f"table below lists the strongest absolute correlations.",
        styles["body"],
    ))

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            pairs.append((cols[i], cols[j], float(value),
                          abs(float(value))))
    pairs.sort(key=lambda x: x[3], reverse=True)
    top_pairs = pairs[:15]

    rows_data = [["Column A", "Column B", "Correlation", "Direction"]]
    direction_colors = []
    for a, b, v, _ in top_pairs:
        if v >= 0.7 or v <= -0.7:
            direction, color = "Strong", BRAND_PRIMARY
        elif v >= 0.4 or v <= -0.4:
            direction, color = "Moderate", BRAND_SECONDARY
        elif v >= 0.2 or v <= -0.2:
            direction, color = "Weak", BRAND_WARN
        else:
            direction, color = "Negligible", BRAND_MUTED
        rows_data.append([str(a), str(b), f"{v:+.3f}", direction])
        direction_colors.append(color)

    corr_table = Table(
        rows_data,
        colWidths=[4.6 * cm, 4.6 * cm, 2.7 * cm, 3.2 * cm],
        repeatRows=1,
    )
    _stylise_table(corr_table, header_bg=BRAND_PRIMARY)
    extra = [("TEXTCOLOR", (3, i), (3, i), col)
             for i, col in enumerate(direction_colors, start=1)]
    if extra:
        corr_table.setStyle(TableStyle(extra))
    story.append(corr_table)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>Reading guide:</b> 0.0–0.19 very weak, 0.2–0.39 weak, 0.4–0.59 "
        "moderate, 0.6–0.79 strong, 0.8+ very strong. Negative values "
        "indicate inverse relationships. Features with strong redundant "
        "correlations should be pruned before modelling.",
        styles["body"],
    ))


# ---------------------------------------------------------------------------
# Page 8 — Business findings
# ---------------------------------------------------------------------------
def add_business_findings(story, styles, df, dataset_name):
    _add_section_heading(
        story,
        styles,
        "Business Findings",
        "The findings below are derived directly from your data using "
        "transparent, deterministic heuristics. Replace any bullet with AI "
        "commentary if your deployment has an LLM provider configured.",
    )

    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    completeness = round(((rows * cols) - missing) / max(rows * cols, 1) * 100, 2)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    findings = []
    findings.append(
        f"<b>Dataset scale.</b> {dataset_name} contains {rows:,} records "
        f"over {cols} columns. {len(numeric_cols)} of them are numeric, "
        f"making the asset directly usable for correlation, regression and "
        f"classification workflows."
    )
    if completeness >= 95:
        cov = "excellent coverage"
    elif completeness >= 80:
        cov = "good coverage"
    else:
        cov = "noticeable gaps"
    findings.append(
        f"<b>Completeness.</b> {completeness:.2f}% of cells are populated — "
        f"{cov}. {missing:,} cells are missing across "
        f"{int(df.isnull().any().sum())} column(s)."
    )
    if duplicates > 0:
        findings.append(
            f"<b>Duplicates.</b> {duplicates:,} duplicate row(s) detected. "
            f"They should be reviewed and removed before any modelling step."
        )
    else:
        findings.append(
            "<b>Duplicates.</b> No duplicate rows detected — the dataset is "
            "clean from a uniqueness standpoint."
        )
    if numeric_cols:
        skew_info = []
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            try:
                skew = float(s.skew())
            except Exception:
                continue
            skew_info.append((col, skew))
        skew_info.sort(key=lambda x: abs(x[1]), reverse=True)
        if skew_info:
            top_col, top_skew = skew_info[0]
            if abs(top_skew) >= 1:
                severity = "highly skewed"
            elif abs(top_skew) >= 0.5:
                severity = "moderately skewed"
            else:
                severity = "approximately symmetric"
            findings.append(
                f"<b>Distribution shape.</b> The most skewed numeric column "
                f"is <b>{top_col}</b> (skew = {top_skew:+.3f}, {severity}). "
                f"Consider a log or Box-Cox transform before parametric "
                f"modelling."
            )

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        try:
            s = df[col].dropna()
            if s.empty:
                continue
            top_share = s.value_counts(normalize=True).iloc[0]
        except Exception:
            continue
        if top_share >= 0.9:
            top_value = s.value_counts().idxmax()
            findings.append(
                f"<b>Class imbalance / dominance.</b> In column "
                f"<b>{col}</b>, value <i>{top_value}</i> covers "
                f"{top_share * 100:.1f}% of records — this column carries "
                f"very little discriminative power."
            )
            break

    for text in findings:
        story.append(Paragraph("• " + text, styles["body"]))


# ---------------------------------------------------------------------------
# Page 9 — Recommendations
# ---------------------------------------------------------------------------
def add_recommendations(story, styles, df, dataset_name):
    _add_section_heading(
        story,
        styles,
        "Recommendations",
        "Priority actions the team should take before any downstream "
        "modelling, reporting or dashboarding work.",
    )

    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    recommendations = []

    high_missing_cols = []
    for col in df.columns:
        c = int(df[col].isnull().sum())
        if c / max(len(df), 1) >= 0.2:
            high_missing_cols.append((col, c))
    high_missing_cols.sort(key=lambda x: x[1], reverse=True)

    if high_missing_cols:
        joined = ", ".join(
            [f"<b>{c}</b> ({n:,})" for c, n in high_missing_cols[:5]]
        )
        recommendations.append(
            f"<b>P1 – Address high-missing columns.</b> The following "
            f"column(s) have ≥20% missing: {joined}. Confirm whether the "
            f"missingness is structural (a default value should be applied) "
            f"or random (use median/mode imputation, or KNN imputation for "
            f"mixed-type tabular data)."
        )

    if duplicates > 0:
        recommendations.append(
            f"<b>P1 – Remove duplicates.</b> {duplicates} duplicate row(s) "
            f"were detected. They bias counts and inflate training sets. "
            f"Use <code>df.drop_duplicates()</code> before modelling."
        )

    type_fixes = []
    for col in df.columns:
        s = df[col]
        if s.dtype == "object":
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() / max(s.notna().sum(), 1) > 0.85:
                type_fixes.append(col)
        if s.dtype == "float64":
            converted = pd.to_numeric(s, errors="coerce")
            if converted.notna().all() and (converted % 1 == 0).all():
                type_fixes.append(col)
    if type_fixes:
        joined = ", ".join([f"<b>{c}</b>" for c in type_fixes[:6]])
        recommendations.append(
            f"<b>P2 – Fix data types.</b> The column(s) {joined} appear "
            f"numeric but are stored as objects or float-with-no-decimals. "
            f"Cast them with <code>pd.to_numeric(..., errors='coerce')</code> "
            f"or <code>astype('Int64')</code> for compact integer storage."
        )

    cardinal_cols_high = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 50 and n_unique / max(rows, 1) < 0.5:
            cardinal_cols_high.append((col, n_unique))
    if cardinal_cols_high:
        joined = ", ".join(
            [f"<b>{c}</b> ({n} unique)" for c, n in cardinal_cols_high[:4]]
        )
        recommendations.append(
            f"<b>P2 – Handle high-cardinality categoricals.</b> {joined} "
            f"may benefit from target encoding, frequency encoding or "
            f"embedding reduction before one-hot expansion, which would "
            f"otherwise explode feature dimensionality."
        )

    recommendations.append(
        "<b>P3 – Track data with versions.</b> Persist this dataset "
        "alongside a version tag, row count and a hash of the schema so "
        "every downstream model can be traced back to the exact dataset "
        "version that produced it."
    )
    recommendations.append(
        "<b>P3 – Validate downstream assumptions.</b> Confirm that any "
        "predictive model based on this dataset satisfies business "
        "fairness, leakage and seasonality tests before deployment."
    )

    for text in recommendations:
        story.append(Paragraph("• " + text, styles["body"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "<i>End of report. Generated automatically by the AI Data Analyst "
        "platform.</i>",
        styles["small"],
    ))
