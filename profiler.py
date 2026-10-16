from typing import Dict, Any, List
import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive profile and summary statistics of the dataset.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary containing overall metrics and per-column profiles.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols if total_cols > 0 else 0

    # Duplicates
    exact_duplicate_rows = int(df.duplicated().sum())
    exact_duplicate_percentage = round((exact_duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    # Memory usage in MB
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    column_profiles: Dict[str, Dict[str, Any]] = {}
    total_missing_cells = 0

    for col in df.columns:
        series = df[col]
        # Treat None, NaN, and blank strings as missing
        missing_count = int(series.isna().sum())
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Count blank/whitespace-only strings
            blank_str_count = int(series.astype(str).str.strip().eq("").sum() - series.isna().sum())
            if blank_str_count > 0:
                missing_count += blank_str_count

        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        total_missing_cells += missing_count

        unique_count = int(series.nunique(dropna=True))
        unique_pct = round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        col_type = str(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series)

        profile: Dict[str, Any] = {
            "name": col,
            "dtype": col_type,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "is_numeric": is_numeric,
            "is_datetime": is_datetime,
            "is_bool": is_bool,
        }

        # Numeric stats
        if is_numeric:
            clean_series = series.dropna()
            if not clean_series.empty:
                q25 = float(clean_series.quantile(0.25))
                q75 = float(clean_series.quantile(0.75))
                iqr = float(q75 - q25)
                profile.update({
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "mean": round(float(clean_series.mean()), 2),
                    "median": round(float(clean_series.median()), 2),
                    "std": round(float(clean_series.std()), 2) if len(clean_series) > 1 else 0.0,
                    "q25": round(q25, 2),
                    "q75": round(q75, 2),
                    "iqr": round(iqr, 2),
                    "negative_count": int((clean_series < 0).sum()),
                    "zero_count": int((clean_series == 0).sum()),
                })
        elif is_datetime:
            clean_series = series.dropna()
            if not clean_series.empty:
                profile.update({
                    "min_date": str(clean_series.min()),
                    "max_date": str(clean_series.max()),
                })
        else:
            # String / Categorical analysis
            clean_series = series.dropna().astype(str)
            if not clean_series.empty:
                # Top value
                mode_vals = clean_series.mode()
                top_value = str(mode_vals.iloc[0]) if not mode_vals.empty else ""
                top_freq = int((clean_series == top_value).sum()) if top_value else 0

                # String lengths
                lengths = clean_series.str.len()
                min_len = int(lengths.min()) if not lengths.empty else 0
                max_len = int(lengths.max()) if not lengths.empty else 0

                # Sample unique values for LLM / preview (max 5)
                samples = list(clean_series.unique()[:5])

                profile.update({
                    "top_value": top_value,
                    "top_frequency": top_freq,
                    "min_length": min_len,
                    "max_length": max_len,
                    "sample_values": samples,
                })

        column_profiles[col] = profile

    dataset_missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "dataset_missing_percentage": dataset_missing_pct,
        "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_percentage": exact_duplicate_percentage,
        "memory_usage_mb": memory_usage_mb,
        "column_profiles": column_profiles,
    }
