import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

OUTPUT_DIR = Path("outputs/reports")

SYSTEM_PROMPT = """You are a senior data analyst writing an EDA (Exploratory \
Data Analysis) summary report for a technical but non-statistician audience.

Given structured JSON findings from profiling, statistics, correlation, and \
outlier detection, write a concise markdown report with these sections:

## Dataset Overview
## Data Quality Issues
## Key Statistical Insights
## Correlations & Relationships
## Outliers
## Recommendations

Be specific — cite actual column names and numbers from the data. Keep it \
under 400 words. Do not invent findings not present in the JSON."""


def build_report_prompt(profiling_result, statistical_result, correlation_result, outlier_result) -> str:
    """Assemble all tool outputs into a single prompt for the LLM."""
    combined_findings = {
        "profiling": profiling_result,
        "statistics": statistical_result,
        "correlation": correlation_result,
        "outliers": outlier_result,
    }
    return f"Here are the EDA findings as JSON:\n\n```json\n{json.dumps(combined_findings, indent=2)}\n```\n\nWrite the report now."


def generate_report(profiling_result, statistical_result, correlation_result, outlier_result) -> str:
    """Call the LLM to synthesize all findings into a markdown report."""
    from agent.llm import get_llm

    llm = get_llm()
    prompt = build_report_prompt(profiling_result, statistical_result, correlation_result, outlier_result)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content


def save_report(report_text: str, filename: str = "eda_report.md") -> str:
    """Save the generated report to disk and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / filename
    save_path.write_text(report_text, encoding="utf-8")
    return str(save_path)


@tool
def report_generator_tool(profiling_result: dict, statistical_result: dict, correlation_result: dict, outlier_result: dict) -> dict:
    """
    Generate the final EDA insights report by synthesizing outputs from
    profiling, statistics, correlation, and outlier detection tools using
    an LLM. Use this as the LAST step, after all other analysis tools
    have run.
    """
    report_text = generate_report(profiling_result, statistical_result, correlation_result, outlier_result)
    report_path = save_report(report_text)

    return {"report_text": report_text, "report_path": report_path}