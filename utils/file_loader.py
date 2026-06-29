from pathlib import Path
from typing import Union
import pandas as pd

supported_extensions = {".csv",".xlsx",".xls"}

class FileLoadError(Exception):
    """Raised when a file cannot be loaded  or fails validation."""
    pass
def load_dataset(file_path: Union[str,Path]):
    path = Path(file_path)

    if not path.exists():
        raise FileLoadError(f"File not found:{path}")
    if path.suffix.lower() not in supported_extensions:
        raise FileLoadError(
            f"Unsupported file type '{path.suffix}'."
            f"supported types : {supported_extensions}"
        )
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        raise FileLoadError(f"Failed to read the file'{path.name}':{e}") from e
    
   
    
def validate_dataset(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise FileLoadError("Loaded dataset is empty")
    if len(df.columns)==0:
        raise FileLoadError("Loaded dataset has no columns")
    
    validate_dataset(df)
    return df

def get_basic_metadata(df: pd.DataFrame)-> dict:
    return {
        "n_rows" : len(df),
        "n_columns" : len(df.columns),
        "columns" : list(df.columns),
        "dtypes" : {col : str(dtype) for col,dtype in df.dtypes.items()},
    }
