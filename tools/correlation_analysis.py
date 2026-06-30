from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/headless use
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

OUTPUT_DIR = Path("outputs/plots")


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson"):
    """Pairwise correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr(method=method)
    

def find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list[dict]:
   
    strong_pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr_matrix.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                strong_pairs.append({
                    "column_1": cols[i],
                    "column_2": cols[j],
                    "correlation": round(float(value), 3),
                })

    return sorted(strong_pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def save_correlation_heatmap(corr_matrix: pd.DataFrame, filename: str = "correlation_heatmap.png") -> str:
    """Render and save a correlation heatmap. Returns the saved file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / filename

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return str(save_path)


@tool
def correlation_analysis_tool(file_path: str) -> dict:
    
    from utils.file_loader import load_dataset

    df = load_dataset(file_path)
    corr_matrix = compute_correlation_matrix(df)

    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return {
            "correlation_matrix": {},
            "strong_correlations": [],
            "heatmap_path": None,
            "note": "Not enough numeric columns to compute correlation.",
        }

    heatmap_path = save_correlation_heatmap(corr_matrix)
    strong_pairs = find_strong_correlations(corr_matrix)

    return {
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "strong_correlations": strong_pairs,
        "heatmap_path": heatmap_path,
    }