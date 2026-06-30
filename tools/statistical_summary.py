import pandas as pd
from langchain_core.tools import tool


def summarize_numeric(df: pd.DataFrame) -> dict:
    """Descriptive stats for numeric columns, including skew/kurtosis."""
    numeric_cols = df.select_dtypes(include="number").columns
    summary = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        summary[col] = {
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "skewness": round(float(series.skew()), 3) if len(series) > 2 else 0.0,
            "kurtosis": round(float(series.kurt()), 3) if len(series) > 3 else 0.0,
        }
    return summary


def summarize_categorical(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Top value counts for categorical/object columns."""
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns
    summary = {}

    for col in cat_cols:
        value_counts = df[col].value_counts().head(top_n)
        summary[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
        }
    return summary


def flag_skewed_columns(numeric_summary: dict, threshold: float = 1.0) -> list[str]:
    """
    Columns with |skewness| > threshold are flagged — useful signal
    for the agent to suggest log-transform or note in the report.
    """
    return [
        col for col, stats in numeric_summary.items()
        if abs(stats.get("skewness", 0)) > threshold
    ]


@tool
def statistical_summary_tool(file_path: str) -> dict:
   
    from utils.file_loader import load_dataset

    df = load_dataset(file_path)
    numeric_summary = summarize_numeric(df)
    categorical_summary = summarize_categorical(df)

    return {
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "skewed_columns": flag_skewed_columns(numeric_summary),
    }


if __name__ == "__main__":
    from utils.file_loader import load_dataset
    import json

    df = load_dataset("data/sample_dataset.csv")
    result = {
        "numeric": summarize_numeric(df),
        "categorical": summarize_categorical(df),
    }
    result["skewed_columns"] = flag_skewed_columns(result["numeric"])
    print(json.dumps(result, indent=2))