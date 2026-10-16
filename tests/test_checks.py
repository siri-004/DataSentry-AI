import pytest
import pandas as pd
import numpy as np
from checks import (
    check_missing_values,
    check_exact_duplicates,
    check_business_duplicates,
    check_invalid_dates,
    check_inconsistent_categories,
    check_negative_values,
    check_statistical_outliers,
    check_data_type_mismatch,
    run_all_checks,
)

@pytest.fixture
def clean_dataframe():
    """A completely clean, well-formed sales dataset."""
    return pd.DataFrame({
        "order_id": [f"ORD-{i:03d}" for i in range(1, 21)],
        "customer_id": [f"CUST-{i:03d}" for i in range(1, 21)],
        "category": ["Electronics", "Clothing", "Home & Kitchen", "Books"] * 5,
        "quantity": [1, 2, 3, 4] * 5,
        "price": [10.0, 25.5, 50.0, 100.0] * 5,
        "order_date": pd.date_range("2024-01-01", periods=20, freq="D").astype(str),
    })

def test_clean_data_no_violations(clean_dataframe):
    """Clean data should not produce false positive violations."""
    findings = run_all_checks(clean_dataframe)
    # Check that there are no critical or high severity violations
    severe_findings = [f for f in findings if f.get("severity") in {"critical", "high"}]
    assert len(severe_findings) == 0, f"Expected 0 severe findings on clean data, got: {severe_findings}"
    assert all(f.get("category") != "Confirmed Violation" for f in findings)

def test_check_missing_values():
    df = pd.DataFrame({
        "customer_id": ["C1", None, "C3", "   "],
        "notes": ["ok", "pending", None, "done"]
    })
    findings = check_missing_values(df, required_columns=["customer_id"])
    assert len(findings) == 2
    cid_finding = next(f for f in findings if f["column"] == "customer_id")
    assert cid_finding["category"] == "Confirmed Violation"
    assert cid_finding["affected_rows"] == 2  # None and "   "

def test_check_exact_duplicates():
    df = pd.DataFrame({
        "a": [1, 2, 1, 3],
        "b": ["x", "y", "x", "z"]
    })
    findings = check_exact_duplicates(df)
    assert len(findings) == 1
    assert findings[0]["affected_rows"] == 1
    assert findings[0]["category"] == "Confirmed Violation"

def test_check_business_duplicates():
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "order_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "amount": [100.0, 100.0, 200.0],
        "notes": ["First attempt", "Second attempt with note", "Third"]
    })
    findings = check_business_duplicates(df, key_columns=["customer_id", "order_date", "amount"])
    assert len(findings) == 1
    assert findings[0]["category"] == "Potential Anomaly"
    assert findings[0]["affected_rows"] == 2

def test_check_invalid_dates():
    df = pd.DataFrame({
        "transaction_date": ["2024-01-15", "2024-02-31", "not_a_date", "2024-03-01", "2099-12-31"]
    })
    findings = check_invalid_dates(df, date_columns=["transaction_date"])
    assert len(findings) == 1
    assert findings[0]["category"] == "Confirmed Violation"
    # '2024-02-31', 'not_a_date', and '2099-12-31' (>2050)
    assert findings[0]["affected_rows"] == 3

def test_check_inconsistent_categories():
    df = pd.DataFrame({
        "product_type": ["Electronics", " electronics", "ELECTRONICS", "Clothing", "Clothing"]
    })
    findings = check_inconsistent_categories(df, categorical_columns=["product_type"])
    assert len(findings) == 1
    assert findings[0]["column"] == "product_type"
    assert findings[0]["affected_rows"] >= 3

def test_check_negative_values():
    df = pd.DataFrame({
        "quantity": [1, 5, -2, 0, 10],
        "unit_price": [15.0, -10.0, 25.0, 30.0, 50.0]
    })
    findings = check_negative_values(df, nonnegative_columns=["quantity", "unit_price"])
    assert len(findings) == 2
    cols_found = {f["column"] for f in findings}
    assert cols_found == {"quantity", "unit_price"}
    assert all(f["category"] == "Confirmed Violation" for f in findings)

def test_outlier_detection_labelled_as_potential_anomaly_not_error():
    """Verify statistical outliers are labeled as Potential Anomaly and not confirmed error."""
    # 20 regular values + 1 extreme value
    normal_values = list(np.random.normal(loc=100, scale=10, size=25))
    extreme_values = normal_values + [10000.0]
    df = pd.DataFrame({"sales_amount": extreme_values})

    findings = check_statistical_outliers(df, numeric_columns=["sales_amount"])
    assert len(findings) == 1
    finding = findings[0]
    assert finding["category"] == "Potential Anomaly"
    assert "human review" in finding["recommended_action"].lower()
    assert finding["affected_rows"] >= 1
