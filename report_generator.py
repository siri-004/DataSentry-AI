from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from graph_state import FindingDict

def generate_report_dict(
    dataset_summary: Dict[str, Any],
    profile: Dict[str, Any],
    findings: List[FindingDict],
    executive_summary: Optional[str] = None,
    llm_enhanced: bool = False
) -> Dict[str, Any]:
    """
    Constructs the final structured investigation report.
    """
    total_findings = len(findings)
    confirmed_count = sum(1 for f in findings if f.get("category") == "Confirmed Violation")
    anomaly_count = sum(1 for f in findings if f.get("category") == "Potential Anomaly")

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for f in findings:
        sev = str(f.get("severity", "low")).lower()
        if sev in severity_counts:
            severity_counts[sev] += 1

    # Overall health assessment label
    if severity_counts["critical"] > 0:
        overall_assessment = "Critical Data Integrity Risk — Urgent Review Required"
    elif severity_counts["high"] > 0 or confirmed_count > 5:
        overall_assessment = "Moderate to High Data Quality Violations Detected"
    elif total_findings > 0:
        overall_assessment = "Minor Potential Anomalies Detected — Human Review Recommended"
    else:
        overall_assessment = "Dataset Clean — No Critical Violations Detected"

    # Default executive summary if none from LLM
    if not executive_summary:
        if total_findings == 0:
            executive_summary = "All deterministic data-quality checks passed successfully with zero rule violations or notable statistical anomalies."
        else:
            executive_summary = (
                f"DataSentry AI identified {total_findings} data-quality findings across {dataset_summary.get('columns', 0)} columns "
                f"and {dataset_summary.get('rows', 0):,} records ({confirmed_count} confirmed rule violations and {anomaly_count} potential anomalies)."
            )

    limitations = [
        "Checks are based on structural patterns, statistical distributions, and standard domain heuristics.",
        "Statistical outliers using IQR indicate values in the tails of the distribution; they are not inherently errors and may represent legitimate high-value transactions.",
        "Proposed remediations must be reviewed and approved by domain experts before production deployment.",
        "Confidential columns or unstructured free-text fields may contain domain-specific semantics beyond automated rule heuristics."
    ]

    report = {
        "dataset_summary": {
            "filename": dataset_summary.get("filename", "unknown"),
            "rows": dataset_summary.get("rows", 0),
            "columns": dataset_summary.get("columns", 0),
            "file_size_kb": dataset_summary.get("file_size_kb", 0.0),
            "memory_usage_mb": profile.get("memory_usage_mb", 0.0),
            "total_missing_cells": profile.get("total_missing_cells", 0),
            "exact_duplicate_rows": profile.get("exact_duplicate_rows", 0),
        },
        "overall_assessment": overall_assessment,
        "executive_summary": executive_summary,
        "findings_summary": {
            "total_findings": total_findings,
            "confirmed_violations": confirmed_count,
            "potential_anomalies": anomaly_count,
            "by_severity": severity_counts,
        },
        "findings": findings,
        "limitations": limitations,
        "metadata": {
            "agent_name": "DataSentry AI",
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_enhanced": llm_enhanced,
        }
    }

    return report


def findings_to_dataframe(findings: List[FindingDict]) -> pd.DataFrame:
    """
    Converts list of findings to a flattened pandas DataFrame for UI table rendering and CSV export.
    """
    if not findings:
        return pd.DataFrame(columns=[
            "ID", "Category", "Issue Type", "Column", "Severity", 
            "Affected Rows", "Affected %", "Evidence", "Business Impact", "Recommended Action"
        ])

    rows = []
    for f in findings:
        rows.append({
            "ID": f.get("id"),
            "Category": f.get("category", "Potential Anomaly"),
            "Issue Type": f.get("issue_type", "").replace("_", " ").title(),
            "Column": f.get("column") or "(All Columns)",
            "Severity": str(f.get("severity", "medium")).upper(),
            "Affected Rows": f.get("affected_rows", 0),
            "Affected %": f"{f.get('affected_percentage', 0.0):.1f}%",
            "Evidence": f.get("evidence", ""),
            "Business Impact": f.get("business_impact", ""),
            "Recommended Action": f.get("recommended_action", ""),
            "Proposed Fix": f.get("proposed_fix_type", ""),
        })

    return pd.DataFrame(rows)
