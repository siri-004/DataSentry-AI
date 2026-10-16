import uuid
import re
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np
from config import Config
from graph_state import FindingDict

def _generate_id(prefix: str = "DQ") -> str:
    """Generate a short unique identifier for a finding."""
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"

def check_missing_values(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> List[FindingDict]:
    """
    Detects missing or blank values across all columns.
    Flags required identifier columns with High/Critical severity as Confirmed Violations.
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    # Determine required columns
    req_cols = set(required_columns or [])
    if not required_columns:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern == col_lower or f"_{pattern}" in col_lower or f"{pattern}_" in col_lower 
                   for pattern in Config.DEFAULT_REQUIRED_ID_PATTERNS):
                req_cols.add(col)

    for col in df.columns:
        series = df[col]
        # Null mask
        null_mask = series.isna()
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Also catch whitespace-only or empty strings or literal 'null'/'n/a'
            str_vals = series.astype(str).str.strip().str.lower()
            empty_mask = series.notna() & str_vals.isin(["", "nan", "none", "null", "n/a", "na"])
            combined_mask = null_mask | empty_mask
        else:
            combined_mask = null_mask

        missing_count = int(combined_mask.sum())
        if missing_count > 0:
            pct = round((missing_count / total_rows) * 100, 2)
            is_required = col in req_cols

            if is_required:
                category = "Confirmed Violation"
                severity = "critical" if pct > 10 else "high"
                impact = f"Column '{col}' appears to be a primary business identifier. Missing values will disrupt entity tracking, joins, and reporting integrity."
                rec = f"Review data source logs for missing '{col}'. Consider filtering unlinked records or obtaining valid keys from upstream systems."
            else:
                category = "Confirmed Violation" if pct > 30 else "Potential Anomaly"
                if pct > 40:
                    severity = "high"
                elif pct > 10:
                    severity = "medium"
                else:
                    severity = "low"
                impact = f"Missing data in '{col}' ({pct}% of rows) can bias aggregations or result in null values in downstream pipelines."
                rec = f"Impute missing values using mode/median or use a default sentinel placeholder (e.g. 'Unknown') based on business needs."

            # Collect up to 5 sample indices
            sample_indices = list(df[combined_mask].index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("MISS"),
                issue_type="missing_values",
                category=category,
                column=col,
                affected_rows=missing_count,
                affected_percentage=pct,
                severity=severity,
                evidence=f"{missing_count:,} out of {total_rows:,} records ({pct}%) have missing or blank values in '{col}'.",
                sample_records=sample_data,
                business_impact=impact,
                recommended_action=rec,
                proposed_fix_type="fill_missing" if not is_required else "drop_missing_id",
                fix_params={"column": col, "missing_count": missing_count, "is_required": is_required},
                confidence="high",
                verified_by_llm=False
            ))

    return findings


def check_exact_duplicates(df: pd.DataFrame) -> List[FindingDict]:
    """
    Detects completely identical rows across all columns.
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())

    if dup_count > 0:
        pct = round((dup_count / total_rows) * 100, 2)
        severity = "high" if pct > 5 else "medium"
        sample_indices = list(df[dup_mask].index[:5])
        sample_data = df.loc[sample_indices].to_dict(orient="records")

        findings.append(FindingDict(
            id=_generate_id("DUP"),
            issue_type="exact_duplicates",
            category="Confirmed Violation",
            column=None,
            affected_rows=dup_count,
            affected_percentage=pct,
            severity=severity,
            evidence=f"{dup_count:,} duplicate rows detected ({pct}% of the dataset) with identical values in all columns.",
            sample_records=sample_data,
            business_impact="Exact duplicate records distort metrics such as revenue, lead counts, and inventory levels through double-counting.",
            recommended_action="Deduplicate dataset by retaining only the first occurrence of each unique record.",
            proposed_fix_type="drop_duplicates",
            fix_params={"keep": "first"},
            confidence="high",
            verified_by_llm=False
        ))

    return findings


def check_business_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None
) -> List[FindingDict]:
    """
    Identifies potential duplicate business records where a subset of core keys match
    (e.g., customer_id + date + amount), even if other columns slightly differ.
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows < 2 or len(df.columns) < 2:
        return findings

    # Infer candidate keys if not provided
    keys_to_use = key_columns or []
    if not keys_to_use:
        # Look for customer_id, order_date/date, and amount/price/total
        col_map = {c.lower(): c for c in df.columns}
        cand_id = next((col_map[c] for c in col_map if any(k in c for k in ["customer", "client", "user", "order"])), None)
        cand_date = next((col_map[c] for c in col_map if "date" in c or "time" in c), None)
        cand_num = next((col_map[c] for c in col_map if any(k in c for k in ["amount", "price", "total", "sales"])), None)
        
        candidates = [c for c in [cand_id, cand_date, cand_num] if c is not None]
        if len(candidates) >= 2:
            keys_to_use = candidates

    if len(keys_to_use) >= 2:
        # Check if subset of keys is duplicated, but row is not an exact duplicate
        dup_keys_mask = df.duplicated(subset=keys_to_use, keep=False)
        exact_dup_mask = df.duplicated(keep=False)
        partial_dup_mask = dup_keys_mask & (~exact_dup_mask)
        partial_count = int(partial_dup_mask.sum())

        if partial_count > 0:
            pct = round((partial_count / total_rows) * 100, 2)
            sample_indices = list(df[partial_dup_mask].index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("BDUP"),
                issue_type="business_duplicates",
                category="Potential Anomaly",
                column=", ".join(keys_to_use),
                affected_rows=partial_count,
                affected_percentage=pct,
                severity="medium",
                evidence=f"{partial_count:,} records share matching key fields ({', '.join(keys_to_use)}) but have differing non-key attributes.",
                sample_records=sample_data,
                business_impact="May indicate duplicate order submissions, accidental double-clicks, or amended orders requiring human verification.",
                recommended_action="Manually review records sharing matching key values to determine if they represent unique valid transactions or duplicate entries.",
                proposed_fix_type="review_manual",
                fix_params={"key_columns": keys_to_use},
                confidence="medium",
                verified_by_llm=False
            ))

    return findings


def check_invalid_dates(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None
) -> List[FindingDict]:
    """
    Detects invalid, unparseable, or nonsensical date strings (e.g. '2024-02-31', 'invalid_date', or far future).
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    # Identify date columns by name or dtype
    candidate_date_cols: Set[str] = set(date_columns or [])
    if not date_columns:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ["date", "time", "created_at", "updated_at", "timestamp", "dob", "birthday"]):
                candidate_date_cols.add(col)

    for col in candidate_date_cols:
        if col not in df.columns:
            continue
        series = df[col]
        # Ignore already null values (covered by missing values check)
        non_null_mask = series.notna() & (series.astype(str).str.strip() != "")
        if non_null_mask.sum() == 0:
            continue

        raw_strings = series[non_null_mask].astype(str)
        # Try to parse
        parsed_dates = pd.to_datetime(raw_strings, errors="coerce")
        invalid_mask = parsed_dates.isna()
        invalid_count = int(invalid_mask.sum())

        # Check for absurd dates (e.g. year < 1900 or year > 2050)
        valid_dates = parsed_dates.dropna()
        absurd_date_count = 0
        if not valid_dates.empty:
            absurd_mask = (valid_dates.dt.year < 1900) | (valid_dates.dt.year > 2050)
            absurd_date_count = int(absurd_mask.sum())

        total_invalid = invalid_count + absurd_date_count
        if total_invalid > 0:
            pct = round((total_invalid / total_rows) * 100, 2)
            invalid_indices = list(raw_strings[invalid_mask].index[:5])
            sample_data = df.loc[invalid_indices].to_dict(orient="records") if invalid_indices else []

            findings.append(FindingDict(
                id=_generate_id("DATE"),
                issue_type="invalid_dates",
                category="Confirmed Violation",
                column=col,
                affected_rows=total_invalid,
                affected_percentage=pct,
                severity="high",
                evidence=f"{total_invalid:,} values in '{col}' cannot be parsed as valid calendar dates (e.g., malformed strings or out-of-range years).",
                sample_records=sample_data,
                business_impact="Time-series analyses, cohort retention, and fiscal period reporting will fail or drop these invalid timestamps.",
                recommended_action=f"Standardize date formatting in '{col}' to ISO-8601 (YYYY-MM-DD) and correct erroneous values like non-existent days.",
                proposed_fix_type="coerce_dates",
                fix_params={"column": col},
                confidence="high",
                verified_by_llm=False
            ))

    return findings


def check_inconsistent_categories(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None
) -> List[FindingDict]:
    """
    Detects text category inconsistencies, including mixed capitalization (e.g. 'North' vs 'north')
    and accidental leading/trailing whitespace (e.g. ' Electronics ' vs 'Electronics').
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    cols_to_check = categorical_columns or [
        c for c in df.columns 
        if (pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]))
        and df[c].nunique() < min(100, total_rows * 0.5)
    ]

    for col in cols_to_check:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        raw_uniques = list(series.unique())
        # Check whitespace discrepancies
        stripped_uniques = set(s.strip() for s in raw_uniques)
        has_whitespace_padding = len(raw_uniques) > len(stripped_uniques)

        # Check capitalization discrepancies
        lower_map: Dict[str, List[str]] = {}
        for val in stripped_uniques:
            key = val.lower()
            lower_map.setdefault(key, []).append(val)

        case_inconsistencies = {k: v for k, v in lower_map.items() if len(v) > 1}

        if has_whitespace_padding or case_inconsistencies:
            # Count affected rows
            affected_rows = 0
            if case_inconsistencies:
                affected_rows = int(series.str.strip().str.lower().isin(case_inconsistencies.keys()).sum())
            elif has_whitespace_padding:
                affected_rows = int((series != series.str.strip()).sum())

            pct = round((affected_rows / total_rows) * 100, 2)
            evidence_details = []
            if case_inconsistencies:
                sample_cases = [f"'{'/'.join(variants)}'" for variants in list(case_inconsistencies.values())[:3]]
                evidence_details.append(f"Mixed casing variants found: {', '.join(sample_cases)}")
            if has_whitespace_padding:
                evidence_details.append("Leading/trailing whitespace causing artificial category fragmentation")

            sample_indices = list(series[series.str.strip().str.lower().isin(case_inconsistencies.keys())].index[:5]) if case_inconsistencies else list(series[series != series.str.strip()].index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("CAT"),
                issue_type="inconsistent_categories",
                category="Confirmed Violation" if has_whitespace_padding else "Potential Anomaly",
                column=col,
                affected_rows=affected_rows,
                affected_percentage=pct,
                severity="medium" if pct > 5 else "low",
                evidence=f"{affected_rows:,} records in '{col}' exhibit inconsistent category formatting. " + "; ".join(evidence_details),
                sample_records=sample_data,
                business_impact="Segmented dashboards and group-by aggregations will splinter single categories into separate rows (e.g. 'North' vs 'north').",
                recommended_action=f"Trim whitespace and apply uniform title-casing or lower-casing to '{col}'.",
                proposed_fix_type="standardize_categories",
                fix_params={"column": col, "case": "title", "strip": True},
                confidence="high",
                verified_by_llm=False
            ))

    return findings


def check_negative_values(
    df: pd.DataFrame,
    nonnegative_columns: Optional[List[str]] = None
) -> List[FindingDict]:
    """
    Identifies negative values in columns where negative values violate domain business rules
    (e.g., negative quantities, unit prices, or sales revenue).
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    cols_to_check: Set[str] = set(nonnegative_columns or [])
    if not nonnegative_columns:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern == col_lower or f"_{pattern}" in col_lower or f"{pattern}_" in col_lower 
                   for pattern in Config.DEFAULT_NON_NEGATIVE_PATTERNS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    cols_to_check.add(col)

    for col in cols_to_check:
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        neg_mask = series < 0
        neg_count = int(neg_mask.sum())

        if neg_count > 0:
            pct = round((neg_count / total_rows) * 100, 2)
            sample_indices = list(df[neg_mask].index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("NEG"),
                issue_type="negative_values",
                category="Confirmed Violation",
                column=col,
                affected_rows=neg_count,
                affected_percentage=pct,
                severity="high" if pct > 1 else "medium",
                evidence=f"{neg_count:,} records ({pct}%) in '{col}' have negative values (min value: {series.min()}).",
                sample_records=sample_data,
                business_impact=f"Negative values in '{col}' violate fundamental business constraints (e.g. negative sales/units) and distort total financial sums.",
                recommended_action=f"Verify whether negative entries in '{col}' signify returns/refunds or typographical errors. Apply absolute value or filter invalid entries.",
                proposed_fix_type="correct_negatives",
                fix_params={"column": col},
                confidence="high",
                verified_by_llm=False
            ))

    return findings


def check_statistical_outliers(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    multiplier: float = 1.5
) -> List[FindingDict]:
    """
    Identifies statistical outliers using the Interquartile Range (IQR) method.
    DOES NOT automatically treat outliers as errors; categorizes them as 'Potential Anomaly'
    for human review.
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows < 10:
        return findings

    num_cols = numeric_columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        if iqr == 0:
            continue

        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            pct = round((outlier_count / total_rows) * 100, 2)
            # Only flag if there are notable outliers and not just normal distribution tails
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            extreme_val = outliers.max() if abs(outliers.max() - q75) > abs(outliers.min() - q25) else outliers.min()
            
            sample_indices = list(df[outlier_mask].index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("OUT"),
                issue_type="statistical_outliers",
                category="Potential Anomaly",
                column=col,
                affected_rows=outlier_count,
                affected_percentage=pct,
                severity="medium" if pct > 5 else "low",
                evidence=f"{outlier_count:,} records in '{col}' fall outside the expected IQR range [{lower_bound:.2f}, {upper_bound:.2f}]. Example extreme value: {extreme_val}.",
                sample_records=sample_data,
                business_impact="Statistically anomalous values may skew regression models, averages, and forecasting, or indicate high-value VIP orders or data entry typos.",
                recommended_action=f"Human review recommended for '{col}' outliers. Verify if high values are genuine high-tier transactions or erroneous input.",
                proposed_fix_type="cap_outliers",
                fix_params={"column": col, "lower_bound": lower_bound, "upper_bound": upper_bound},
                confidence="medium",
                verified_by_llm=False
            ))

    return findings


def check_data_type_mismatch(df: pd.DataFrame) -> List[FindingDict]:
    """
    Detects columns stored as object/string when they are predominantly numeric
    (e.g., currency symbols '$49.99', commas '1,200', or percentage signs '15%').
    """
    findings: List[FindingDict] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    for col in df.columns:
        series = df[col].dropna()
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            continue
        if len(series) < 5:
            continue

        raw_strings = series.astype(str).str.strip()
        # Clean currency and comma characters
        cleaned = raw_strings.str.replace(r"[\$,%€£¥]", "", regex=True).str.replace(",", "")
        
        # Test numeric parseability
        numeric_test = pd.to_numeric(cleaned, errors="coerce")
        valid_numeric_count = int(numeric_test.notna().sum())
        ratio = valid_numeric_count / len(series)

        # If > 70% of non-null values look numeric, but column is object
        if 0.7 <= ratio < 1.0 or (ratio == 1.0 and any(sym in raw_strings.iloc[0] for sym in ["$", "%", "€"])):
            affected = len(series) - valid_numeric_count if ratio < 1.0 else len(series)
            pct = round((affected / total_rows) * 100, 2)
            
            sample_indices = list(series.index[:5])
            sample_data = df.loc[sample_indices].to_dict(orient="records")

            findings.append(FindingDict(
                id=_generate_id("TYPE"),
                issue_type="data_type_mismatch",
                category="Confirmed Violation",
                column=col,
                affected_rows=affected,
                affected_percentage=pct,
                severity="medium",
                evidence=f"Column '{col}' is typed as text/object but {round(ratio * 100)}% of values are numeric format (e.g. currency signs, commas, or mixed text).",
                sample_records=sample_data,
                business_impact="Cannot perform numerical calculations, summaries, or charts directly on this column until parsed.",
                recommended_action=f"Clean special characters (e.g. '$', commas) and cast '{col}' to numeric float/int.",
                proposed_fix_type="cast_numeric",
                fix_params={"column": col},
                confidence="high",
                verified_by_llm=False
            ))

    return findings


def run_all_checks(
    df: pd.DataFrame,
    config_overrides: Optional[Dict[str, Any]] = None
) -> List[FindingDict]:
    """
    Executes all deterministic checks and returns an aggregated list of findings.
    """
    cfg = config_overrides or {}
    findings: List[FindingDict] = []

    # 1. Missing values
    findings.extend(check_missing_values(df, required_columns=cfg.get("required_columns")))

    # 2. Exact duplicates
    findings.extend(check_exact_duplicates(df))

    # 3. Potential business duplicates
    findings.extend(check_business_duplicates(df, key_columns=cfg.get("business_key_columns")))

    # 4. Invalid dates
    findings.extend(check_invalid_dates(df, date_columns=cfg.get("date_columns")))

    # 5. Inconsistent categories
    findings.extend(check_inconsistent_categories(df, categorical_columns=cfg.get("categorical_columns")))

    # 6. Negative values
    findings.extend(check_negative_values(df, nonnegative_columns=cfg.get("nonnegative_columns")))

    # 7. Statistical outliers (IQR)
    iqr_mult = cfg.get("iqr_multiplier", Config.DEFAULT_IQR_MULTIPLIER)
    findings.extend(check_statistical_outliers(df, numeric_columns=cfg.get("numeric_columns"), multiplier=iqr_mult))

    # 8. Data type mismatches
    findings.extend(check_data_type_mismatch(df))

    return findings
