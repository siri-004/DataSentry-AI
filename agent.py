import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from langgraph.graph import StateGraph, START, END

from graph_state import AgentState, FindingDict
from data_loader import load_dataset, DataLoaderError
from profiler import profile_dataset
from checks import run_all_checks
from llm_analyzer import investigate_findings_with_llm, get_llm
from report_generator import generate_report_dict
from config import Config

logger = logging.getLogger(__name__)

# ==========================================
# 1. NODE: Load Dataset
# ==========================================
def load_dataset_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Reads dataset file, validates dimensions, and identifies column types."""
    logger.info("Executing Node 1: load_dataset_node")
    df = state.get("df")
    summary = state.get("dataset_summary", {})
    overrides = state.get("config_overrides", {})

    if df is None:
        file_bytes = state.get("file_bytes")
        filename = state.get("filename")
        if not file_bytes:
            raise DataLoaderError("No dataset or file bytes provided to the agent.")
        df, summary = load_dataset(file_bytes, filename=filename, config_overrides=overrides)
    else:
        if not summary:
            summary = {
                "filename": state.get("filename", "dataframe_input.csv"),
                "file_size_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }

    # Check if LLM is available
    llm_instance = get_llm(overrides)
    llm_available = llm_instance is not None

    return {
        "df": df,
        "dataset_summary": summary,
        "workflow_stage": "Dataset Loaded",
        "llm_available": llm_available,
    }

# ==========================================
# 2. NODE: Profile Dataset
# ==========================================
def profile_dataset_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Calculates missing counts, unique values, duplicate counts, and statistics."""
    logger.info("Executing Node 2: profile_dataset_node")
    df = state["df"]
    profile = profile_dataset(df)
    return {
        "profile": profile,
        "workflow_stage": "Dataset Profiled",
    }

# ==========================================
# 3. NODE: Detect Issues
# ==========================================
def detect_issues_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Runs deterministic checks for empirical data-quality problems."""
    logger.info("Executing Node 3: detect_issues_node")
    df = state["df"]
    overrides = state.get("config_overrides", {})
    raw_issues = run_all_checks(df, config_overrides=overrides)
    return {
        "raw_issues": raw_issues,
        "workflow_stage": f"Issues Detected ({len(raw_issues)} findings)",
    }

# ==========================================
# 4. NODE: Investigate Findings (LLM)
# ==========================================
def investigate_findings_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Uses LLM to interpret findings without inventing data."""
    logger.info("Executing Node 4: investigate_findings_node")
    raw_issues = state.get("raw_issues", [])
    summary = state.get("dataset_summary", {})
    profile = state.get("profile", {})
    overrides = state.get("config_overrides", {})

    enhanced_findings, exec_summary, llm_used = investigate_findings_with_llm(
        findings=raw_issues,
        dataset_summary=summary,
        profile=profile,
        config_overrides=overrides,
    )

    return {
        "investigated_findings": enhanced_findings,
        "workflow_stage": "Findings Interpreted with AI" if llm_used else "Findings Analyzed (Deterministic)",
        "final_report": {"executive_summary": exec_summary} if exec_summary else {},
    }

# ==========================================
# 5. NODE: Prioritize Issues
# ==========================================
def prioritize_issues_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Assigns severity and ranks findings, keeping severity separate from statistical outliers."""
    logger.info("Executing Node 5: prioritize_issues_node")
    findings = state.get("investigated_findings") or state.get("raw_issues", [])

    # Order priority: critical > high > medium > low > info
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

    # Sort: Confirmed Violations first, then severity order, then affected rows descending
    sorted_findings = sorted(
        findings,
        key=lambda f: (
            0 if f.get("category") == "Confirmed Violation" else 1,
            severity_order.get(str(f.get("severity", "low")).lower(), 5),
            -f.get("affected_rows", 0)
        )
    )

    return {
        "prioritized_findings": sorted_findings,
        "workflow_stage": "Issues Prioritized",
    }

# ==========================================
# 6. NODE: Recommend Fixes
# ==========================================
def recommend_fixes_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Enriches context-appropriate corrections without touching the original data."""
    logger.info("Executing Node 6: recommend_fixes_node")
    prioritized = state.get("prioritized_findings", [])
    recommendations: List[FindingDict] = []

    for f in prioritized:
        f_copy = dict(f)
        # Ensure clear remediation instructions and safety notice
        fix_type = f_copy.get("proposed_fix_type")
        col = f_copy.get("column")
        if not f_copy.get("recommended_action"):
            if fix_type == "drop_duplicates":
                f_copy["recommended_action"] = "Remove redundant duplicate rows after verifying deduplication key."
            elif fix_type == "fill_missing":
                f_copy["recommended_action"] = f"Impute missing values in '{col}' or specify a default placeholder."
            else:
                f_copy["recommended_action"] = f"Review affected records in '{col}'."

        recommendations.append(f_copy)  # type: ignore

    return {
        "recommendations": recommendations,
        "workflow_stage": "Recommendations Formulated",
    }

# ==========================================
# 7. NODE: Generate Report
# ==========================================
def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """Node 7: Synthesizes the final structured report."""
    logger.info("Executing Node 7: generate_report_node")
    summary = state.get("dataset_summary", {})
    profile = state.get("profile", {})
    recommendations = state.get("recommendations", [])
    report_meta = state.get("final_report", {})
    exec_summary = report_meta.get("executive_summary") if isinstance(report_meta, dict) else None
    llm_enhanced = state.get("llm_available", False) and any(f.get("verified_by_llm", False) for f in recommendations)

    final_report = generate_report_dict(
        dataset_summary=summary,
        profile=profile,
        findings=recommendations,
        executive_summary=exec_summary,
        llm_enhanced=llm_enhanced,
    )

    return {
        "final_report": final_report,
        "workflow_stage": "Investigation Complete",
    }


# ==========================================
# Conditional Edge Router
# ==========================================
def route_after_detect(state: AgentState) -> str:
    """
    Decides whether to route through LLM investigation or skip straight to prioritization.
    Skips LLM if no issues were detected OR if LLM is unavailable.
    """
    raw_issues = state.get("raw_issues", [])
    llm_available = state.get("llm_available", False)

    if len(raw_issues) > 0 and llm_available:
        return "investigate_findings"
    return "prioritize_issues"


# ==========================================
# Build and Compile Graph
# ==========================================
def build_datasentry_graph():
    """Builds and compiles the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # Add all 7 nodes
    workflow.add_node("load_dataset", load_dataset_node)
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("detect_issues", detect_issues_node)
    workflow.add_node("investigate_findings", investigate_findings_node)
    workflow.add_node("prioritize_issues", prioritize_issues_node)
    workflow.add_node("recommend_fixes", recommend_fixes_node)
    workflow.add_node("generate_report", generate_report_node)

    # Add edges
    workflow.add_edge(START, "load_dataset")
    workflow.add_edge("load_dataset", "profile_dataset")
    workflow.add_edge("profile_dataset", "detect_issues")

    # Conditional routing from detect_issues
    workflow.add_conditional_edges(
        "detect_issues",
        route_after_detect,
        {
            "investigate_findings": "investigate_findings",
            "prioritize_issues": "prioritize_issues",
        }
    )

    workflow.add_edge("investigate_findings", "prioritize_issues")
    workflow.add_edge("prioritize_issues", "recommend_fixes")
    workflow.add_edge("recommend_fixes", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# Global compiled app
datasentry_graph = build_datasentry_graph()


def run_datasentry_investigation(
    df: Optional[pd.DataFrame] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level entry point to run the DataSentry AI agent graph.
    """
    initial_state: AgentState = {
        "df": df,
        "file_bytes": file_bytes,
        "filename": filename,
        "config_overrides": config_overrides or {},
        "raw_issues": [],
        "investigated_findings": [],
        "prioritized_findings": [],
        "recommendations": [],
        "final_report": {},
        "workflow_stage": "Starting",
        "llm_available": False,
    }

    final_state = datasentry_graph.invoke(initial_state)
    return final_state
