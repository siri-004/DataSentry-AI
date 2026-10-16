import pandas as pd
from remediation import apply_remediations

def test_apply_remediations_does_not_mutate_original():
    orig_df = pd.DataFrame({
        "id": ["1", "2", "2", "3"],
        "category": ["electronics", "Books", "Books", " Electronics"],
        "quantity": [-5, 10, 10, 20],
    })
    orig_df_backup = orig_df.copy(deep=True)

    approved = [
        {"proposed_fix_type": "drop_duplicates", "column": None, "fix_params": {"keep": "first"}},
        {"proposed_fix_type": "standardize_categories", "column": "category", "fix_params": {"case": "title"}},
        {"proposed_fix_type": "correct_negatives", "column": "quantity", "fix_params": {}},
    ]

    cleaned_df, logs, diff = apply_remediations(orig_df, approved)

    # Assert original df was NOT mutated
    pd.testing.assert_frame_equal(orig_df, orig_df_backup)

    # Assert cleaned df has transformations applied
    assert len(cleaned_df) == 3  # dropped 1 duplicate from 4 rows
    assert all(cleaned_df["quantity"] >= 0)  # negative converted
    assert "Electronics" in list(cleaned_df["category"])
    assert len(logs) == 3
    assert diff["rows_removed"] == 1
