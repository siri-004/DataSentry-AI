from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd

class FindingDict(TypedDict, total=False):
    """Represents a single data-quality finding or anomaly."""
    id: str
    issue_type: str            # e.g., 'missing_values', 'exact_duplicates', 'business_duplicates',
                               # 'invalid_dates', 'negative_values', 'inconsistent_categories',
                               # 'data_type_mismatch', 'statistical_outliers'
    category: str              # 'Confirmed Violation' vs 'Potential Anomaly'
    column: Optional[str]      # Relevant column name or None if table-level
    affected_rows: int         # Count of affected rows
    affected_percentage: float # Percentage of total dataset rows
    severity: str              # 'critical', 'high', 'medium', 'low', 'info'
    evidence: str              # Clear, empirical evidence backing this finding
    sample_records: List[Dict[str, Any]] # Up to a few sample rows / row indices
    business_impact: str       # Consequence of this issue on business decisions
    recommended_action: str    # Context-appropriate action recommended
    proposed_fix_type: str     # 'drop_duplicates', 'fill_missing', 'standardize_case',
                               # 'strip_whitespace', 'cap_outliers', 'remove_invalid'
    fix_params: Dict[str, Any] # Parameters needed by the remediation engine to execute the fix
    confidence: str            # 'high', 'medium', 'low'
    verified_by_llm: bool      # True if enhanced/reviewed by LLM, False if rule-based fallback


class AgentState(TypedDict, total=False):
    """State object for the LangGraph workflow."""
    # Input data
    file_bytes: Optional[bytes]
    filename: Optional[str]
    df: Optional[pd.DataFrame]

    # Workflow runtime configuration overrides
    config_overrides: Dict[str, Any]

    # Analysis results
    dataset_summary: Dict[str, Any]
    profile: Dict[str, Any]
    raw_issues: List[FindingDict]
    investigated_findings: List[FindingDict]
    prioritized_findings: List[FindingDict]
    recommendations: List[FindingDict]
    final_report: Dict[str, Any]

    # Workflow operational metadata
    workflow_stage: str
    llm_available: bool
    error: Optional[str]
