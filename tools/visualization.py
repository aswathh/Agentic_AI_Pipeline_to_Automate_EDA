from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

OUTPUT_DIR = Path("outputs/plots")
MAX_CATEGORICAL_PLOTS = 5
MAX_NUMERIC_PLOTS = 8


def plot_numeric_distribution(df: pd.DataFrame, col: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"dist_{col}.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Distribution: {col}")

    sns.boxplot(x=df[col].dropna(), ax=axes[1], color="lightcoral")
    axes[1].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return str(save_path)


def plot_categorical_distribution(df: pd.DataFrame, col: str, top_n: int = 10) -> str:
    """Bar chart of top N value counts for a categorical column."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"bar_{col}.png"

    value_counts = df[col].value_counts().head(top_n)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=value_counts.values, y=value_counts.index, color="seagreen", orient="h")
    plt.title(f"Top Values: {col}")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return str(save_path)


def generate_all_visualizations(df: pd.DataFrame) -> dict:
    """Auto-generate plots, capped to avoid excessive files on wide datasets."""
    numeric_cols = list(df.select_dtypes(include="number").columns)[:MAX_NUMERIC_PLOTS]
    cat_cols = list(df.select_dtypes(include=["object", "category", "str"]).columns)[:MAX_CATEGORICAL_PLOTS]

    numeric_paths = [plot_numeric_distribution(df, col) for col in numeric_cols]
    categorical_paths = [plot_categorical_distribution(df, col) for col in cat_cols]

    return {
        "numeric_plots": numeric_paths,
        "categorical_plots": categorical_paths,
        "total_plots": len(numeric_paths) + len(categorical_paths),
    }


@tool
def visualization_tool(file_path: str) -> dict:
    """
    Auto-generate visualizations: histogram + boxplot for numeric
    columns, bar chart for categorical columns. Use this AFTER outlier
    detection to visually confirm distributions and outliers already
    flagged numerically.
    """
    from utils.file_loader import load_dataset

    df = load_dataset(file_path)
    return generate_all_visualizations(df)