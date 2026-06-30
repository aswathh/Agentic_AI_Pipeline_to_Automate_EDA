from langgraph.graph import StateGraph, END

from agent.state import EDAAgentState, create_initial_state
from utils.file_loader import load_dataset, get_basic_metadata

from tools.data_profiling import profile_dataframe
from tools.statistical_summary import summarize_numeric, summarize_categorical, flag_skewed_columns
from tools.correlation_analysis import compute_correlation_matrix, find_strong_correlations, save_correlation_heatmap
from tools.outlier_detection import detect_outliers, flag_high_outlier_columns
from tools.visualization import generate_all_visualizations
from tools.report_generator import generate_report, save_report


def load_data_node(state: EDAAgentState) -> dict:
    try:
        df = load_dataset(state["file_path"])
        return {
            "df": df,
            "metadata": get_basic_metadata(df),
            "steps_completed": state.get("steps_completed", []) + ["load_data"],
        }
    except Exception as e:
        return {"error": f"load_data failed: {e}"}


def profiling_node(state: EDAAgentState) -> dict:
    try:
        result = profile_dataframe(state["df"])
        return {
            "profiling_result": result,
            "steps_completed": state.get("steps_completed", []) + ["profiling"],
        }
    except Exception as e:
        return {"error": f"profiling failed: {e}"}


def statistics_node(state: EDAAgentState) -> dict:
    try:
        df = state["df"]
        numeric_summary = summarize_numeric(df)
        result = {
            "numeric": numeric_summary,
            "categorical": summarize_categorical(df),
            "skewed_columns": flag_skewed_columns(numeric_summary),
        }
        return {
            "statistical_summary": result,
            "steps_completed": state.get("steps_completed", []) + ["statistics"],
        }
    except Exception as e:
        return {"error": f"statistics failed: {e}"}


def correlation_node(state: EDAAgentState) -> dict:
    try:
        df = state["df"]
        corr_matrix = compute_correlation_matrix(df)
        if corr_matrix.empty or len(corr_matrix.columns) < 2:
            result = {"correlation_matrix": {}, "strong_correlations": [], "heatmap_path": None}
        else:
            result = {
                "correlation_matrix": corr_matrix.round(3).to_dict(),
                "strong_correlations": find_strong_correlations(corr_matrix),
                "heatmap_path": save_correlation_heatmap(corr_matrix),
            }
        return {
            "correlation_result": result,
            "steps_completed": state.get("steps_completed", []) + ["correlation"],
        }
    except Exception as e:
        return {"error": f"correlation failed: {e}"}


def outliers_node(state: EDAAgentState) -> dict:
    try:
        df = state["df"]
        outlier_raw = detect_outliers(df)
        result = {
            "by_column": outlier_raw,
            "flagged_columns": flag_high_outlier_columns(outlier_raw),
        }
        return {
            "outlier_result": result,
            "steps_completed": state.get("steps_completed", []) + ["outliers"],
        }
    except Exception as e:
        return {"error": f"outliers failed: {e}"}


def visualization_node(state: EDAAgentState) -> dict:
    try:
        df = state["df"]
        viz_result = generate_all_visualizations(df)
        all_paths = viz_result["numeric_plots"] + viz_result["categorical_plots"]
        return {
            "visualization_paths": all_paths,
            "steps_completed": state.get("steps_completed", []) + ["visualization"],
        }
    except Exception as e:
        return {"error": f"visualization failed: {e}"}


def report_node(state: EDAAgentState) -> dict:
    try:
        report_text = generate_report(
            state["profiling_result"],
            state["statistical_summary"],
            state["correlation_result"],
            state["outlier_result"],
        )
        save_report(report_text)
        return {
            "final_report": report_text,
            "steps_completed": state.get("steps_completed", []) + ["report"],
        }
    except Exception as e:
        return {"error": f"report generation failed: {e}"}


def check_error(state: EDAAgentState) -> str:
    if state.get("error"):
        return "end"
    return "continue"


def build_eda_graph():
    graph = StateGraph(EDAAgentState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("profiling", profiling_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("correlation", correlation_node)
    graph.add_node("outliers", outliers_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("load_data")

    node_order = [
        "load_data", "profiling", "statistics",
        "correlation", "outliers", "visualization", "report",
    ]
    for i, node in enumerate(node_order):
        next_node = node_order[i + 1] if i + 1 < len(node_order) else END
        graph.add_conditional_edges(
            node,
            check_error,
            {"continue": next_node, "end": END},
        )

    return graph.compile()


def run_eda_pipeline(file_path: str) -> EDAAgentState:
    app = build_eda_graph()
    initial_state = create_initial_state(file_path)
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    result = run_eda_pipeline("data/sample_dataset.csv")
    print("Steps completed:", result.get("steps_completed"))
    if result.get("error"):
        print("ERROR:", result["error"])
    else:
        print("\n--- FINAL REPORT ---\n")
        print(result["final_report"])