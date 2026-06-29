from typing import TypedDict, Optional, Any
import pandas as pd


class EDAAgentState(TypedDict, total=False):
   
    

   #total=False -> not all keys need to be present at every step.
    
   

    #  Input 
    file_path: str                          # path to uploaded dataset
    df: Optional[pd.DataFrame]               # loaded dataframe (in-memory)

    #  Metadata
    metadata: Optional[dict]                 # n_rows, n_columns, dtypes, etc.

    # Tool outputs 
    profiling_result: Optional[dict]         # nulls, duplicates, dtype summary
    statistical_summary: Optional[dict]      # mean, median, std, skew, etc.
    correlation_result: Optional[dict]       # correlation matrix + heatmap path
    outlier_result: Optional[dict]           # IQR/Z-score flagged columns/rows
    visualization_paths: Optional[list[str]] # saved plot file paths

    #  Agent reasoning trail 
    steps_completed: list[str]               # ["profiling", "stats", ...]
    next_action: Optional[str]               # what the agent decided to do next
    messages: list[Any]                      # LLM conversation history

    # Final output
    final_report: Optional[str]              # compiled markdown/text report
    error: Optional[str]                     # captured error, if any step fails


def create_initial_state(file_path: str) -> EDAAgentState:
    """
    Factory function to create a fresh state when a new dataset is submitted.
    """
    return EDAAgentState(
        file_path=file_path,
        df=None,
        metadata=None,
        profiling_result=None,
        statistical_summary=None,
        correlation_result=None,
        outlier_result=None,
        visualization_paths=[],
        steps_completed=[],
        next_action=None,
        messages=[],
        final_report=None,
        error=None,
    )