import pandas as pd
import numpy as np
from langchain_core.tools import tool


def detect_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> dict:
    series = df[col].dropna()
    if len(series) < 4:
        return {"method": "iqr", "outlier_count": 0, "outlier_pct": 0.0, "bounds": None}

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    outliers = series[(series < lower) | (series > upper)]
    return {
        "method": "iqr",
        "outlier_count": int(len(outliers)),
        "outlier_pct": round((len(outliers) / len(series)) * 100, 2),
        "bounds": {"lower": round(float(lower), 3), "upper": round(float(upper), 3)},
    }


def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> dict:
    
    series = df[col].dropna()
    if len(series) < 4 or series.std() == 0:
        return {"method": "zscore", "outlier_count": 0, "outlier_pct": 0.0, "threshold": threshold}

    z_scores = np.abs((series - series.mean()) / series.std())
    outliers = series[z_scores > threshold]

    return {
        "method": "zscore",
        "outlier_count": int(len(outliers)),
        "outlier_pct": round((len(outliers) / len(series)) * 100, 2),
        "threshold": threshold,
    }


def detect_outliers(df: pd.DataFrame) -> dict:
    """Run both IQR and Z-score detection across all numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    result = {}

    for col in numeric_cols:
        result[col] = {
            "iqr": detect_outliers_iqr(df, col),
            "zscore": detect_outliers_zscore(df, col),
        }

    return result


def flag_high_outlier_columns(outlier_result: dict, pct_threshold: float = 5.0) -> list[str]:
    """Columns where IQR-based outlier % exceeds threshold."""
    flagged = []
    for col, methods in outlier_result.items():
        if methods["iqr"]["outlier_pct"] > pct_threshold:
            flagged.append(col)
    return flagged


@tool
def outlier_detection_tool(file_path: str) -> dict:
    """
    Detect outliers in numeric columns using IQR and Z-score methods.
    Use this AFTER correlation analysis to identify data points that
    may need investigation or cleaning before modeling.
    """
    from utils.file_loader import load_dataset

    df = load_dataset(file_path)
    outlier_result = detect_outliers(df)
    flagged = flag_high_outlier_columns(outlier_result)

    return {
        "by_column": outlier_result,
        "flagged_columns": flagged,
    }