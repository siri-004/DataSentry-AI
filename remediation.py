from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import copy

def apply_remediations(
    original_df: pd.DataFrame,
    approved_findings: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Applies user-approved remediations to a copy of the dataset.
    Never modifies the original dataframe in-place.

    Args:
        original_df: The source pandas DataFrame.
        approved_findings: List of finding dictionaries that the user selected for remediation.

    Returns:
        Tuple of:
          - cleaned_df: The newly transformed DataFrame.
          - change_log: List of human-readable explanations of every action taken.
          - diff_summary: Summary of before vs after metrics.
    """
    df = original_df.copy(deep=True)
    change_log: List[str] = []

    initial_rows = len(df)
    initial_nulls = int(df.isna().sum().sum())

    for finding in approved_findings:
        fix_type = finding.get("proposed_fix_type")
        col = finding.get("column")
        params = finding.get("fix_params", {})

        if fix_type == "drop_duplicates":
            pre_count = len(df)
            df = df.drop_duplicates(keep=params.get("keep", "first")).reset_index(drop=True)
            dropped = pre_count - len(df)
            change_log.append(f"Dropped {dropped} exact duplicate row(s).")

        elif fix_type == "drop_missing_id":
            if col and col in df.columns:
                pre_count = len(df)
                # Drop rows where required ID is null or blank
                mask = df[col].notna() & (df[col].astype(str).str.strip().str.lower().isin(["", "nan", "null", "none", "n/a"]) == False)
                df = df[mask].reset_index(drop=True)
                dropped = pre_count - len(df)
                change_log.append(f"Removed {dropped} row(s) missing primary business identifier in '{col}'.")

        elif fix_type == "fill_missing":
            if col and col in df.columns:
                null_mask = df[col].isna() | (df[col].astype(str).str.strip().str.lower().isin(["", "nan", "null", "none", "n/a"]))
                fill_count = int(null_mask.sum())
                if fill_count > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_val = df[col].median()
                        df[col] = df[col].fillna(fill_val)
                        change_log.append(f"Imputed {fill_count} missing value(s) in '{col}' using median ({round(float(fill_val), 2)}).")
                    else:
                        fill_val = "Unknown"
                        df.loc[null_mask, col] = fill_val
                        change_log.append(f"Filled {fill_count} missing value(s) in '{col}' with placeholder '{fill_val}'.")

        elif fix_type == "standardize_categories":
            if col and col in df.columns:
                pre_series = df[col].astype(str)
                # Strip whitespace and apply title-case
                df[col] = df[col].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)
                altered = int((pre_series != df[col].astype(str)).sum())
                change_log.append(f"Standardized category whitespace and title-casing for {altered} value(s) in '{col}'.")

        elif fix_type == "correct_negatives":
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                neg_mask = df[col] < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
                    change_log.append(f"Converted {neg_count} negative value(s) in '{col}' to positive absolute values.")

        elif fix_type == "coerce_dates":
            if col and col in df.columns:
                pre_dates = df[col].copy()
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                change_log.append(f"Coerced date format in '{col}' to ISO YYYY-MM-DD.")

        elif fix_type == "cast_numeric":
            if col and col in df.columns:
                cleaned_str = df[col].astype(str).str.replace(r"[\$,%€£¥]", "", regex=True).str.replace(",", "").str.strip()
                df[col] = pd.to_numeric(cleaned_str, errors="coerce")
                change_log.append(f"Cleaned non-numeric symbols and cast column '{col}' to numeric.")

        elif fix_type == "cap_outliers":
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                lower = params.get("lower_bound")
                upper = params.get("upper_bound")
                if lower is not None and upper is not None:
                    outlier_mask = (df[col] < lower) | (df[col] > upper)
                    outlier_count = int(outlier_mask.sum())
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    change_log.append(f"Capped {outlier_count} statistical outlier(s) in '{col}' between [{round(lower, 2)}, {round(upper, 2)}].")

    final_rows = len(df)
    final_nulls = int(df.isna().sum().sum())

    diff_summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "nulls_resolved": max(0, initial_nulls - final_nulls),
        "actions_executed": len(change_log),
    }

    return df, change_log, diff_summary
