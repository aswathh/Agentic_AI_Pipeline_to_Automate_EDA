import pandas as pd
from langchain_core.tools import tool


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Core profiling logic (pure function, easy to unit test).
    """
    n_rows = len(df)

    column_profile = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        column_profile[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_%": round((missing_count / n_rows) * 100, 2) if n_rows else 0.0,
            "n_unique": int(df[col].nunique()),
        }

    result = {
        "shape": {"rows": n_rows, "columns": len(df.columns)},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": column_profile,
        "high_missing_columns": [
            col for col, stats in column_profile.items() if stats["missing_%"] > 30
        ],
    }
    return result


@tool
def data_profiling_tool(file_path: str) -> dict:
    
    from utils.file_loader import load_dataset

    df = load_dataset(file_path)
    return profile_dataframe(df)


if __name__ == "__main__":
    from utils.file_loader import load_dataset

    df = load_dataset("data/sample_dataset.csv")
    result = profile_dataframe(df)

    import json
    print(json.dumps(result, indent=2))