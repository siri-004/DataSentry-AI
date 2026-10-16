import os
import pytest
import pandas as pd
from agent import datasentry_graph, run_datasentry_investigation

def test_agent_graph_execution_without_llm():
    """Verify that agent graph runs smoothly end-to-end even with no LLM key configured."""
    df = pd.DataFrame({
        "customer_id": ["C1", None, "C2", "C1"],
        "category": ["Electronics", " electronics", "Books", "Electronics"],
        "quantity": [1, 2, -3, 1],
        "price": [100.0, 50.0, 25.0, 100.0],
        "order_date": ["2024-01-01", "not_a_date", "2024-01-03", "2024-01-01"],
    })

    result_state = run_datasentry_investigation(
        df=df,
        filename="test_sales.csv",
        config_overrides={"gemini_api_key": "", "openai_api_key": ""}
    )

    # Verify state keys
    assert "df" in result_state
    assert "dataset_summary" in result_state
    assert "profile" in result_state
    assert "raw_issues" in result_state
    assert "final_report" in result_state
    assert result_state["workflow_stage"] == "Investigation Complete"

    # Verify report structure
    report = result_state["final_report"]
    assert "dataset_summary" in report
    assert "overall_assessment" in report
    assert "findings" in report
    assert "limitations" in report
    assert "findings_summary" in report

    # Verify findings are populated
    findings = report["findings"]
    assert len(findings) > 0
    issue_types = {f["issue_type"] for f in findings}
    assert "missing_values" in issue_types
    assert "exact_duplicates" in issue_types
    assert "negative_values" in issue_types
    assert "invalid_dates" in issue_types

def test_agent_on_sample_messy_sales():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "messy_sales.csv")
    assert os.path.exists(sample_path)
    df = pd.read_csv(sample_path)

    result_state = run_datasentry_investigation(
        df=df,
        filename="messy_sales.csv",
        config_overrides={"openai_api_key": ""}
    )

    report = result_state["final_report"]
    findings = report["findings"]
    issue_types = {f["issue_type"] for f in findings}

    # Verify core detectors caught issues
    expected_issues = {
        "missing_values",
        "exact_duplicates",
        "inconsistent_categories",
        "negative_values",
        "invalid_dates",
        "statistical_outliers"
    }
    for expected in expected_issues:
        assert expected in issue_types, f"Expected {expected} to be detected in messy_sales.csv"
