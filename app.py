import os
import json
import io
import streamlit as st
import pandas as pd

from config import Config
from data_loader import load_dataset, DataLoaderError
from profiler import profile_dataset
from agent import datasentry_graph, run_datasentry_investigation
from report_generator import findings_to_dataframe
from remediation import apply_remediations

# Set page config
st.set_page_config(
    page_title="DataSentry AI — Silent Data Error Detective",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Global fonts and background nuances */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero section */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Severity badges */
    .badge-critical {
        background-color: #EF4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #F97316;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #FBBF24;
        color: #1E293B;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low {
        background-color: #38BDF8;
        color: #0F172A;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Status alerts */
    .status-banner {
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
    }
    .status-llm-active {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
    }
    .status-llm-inactive {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #FBBF24;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR: Configuration and Business Rules
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("DataSentry AI")
    st.caption("v1.0 • Autonomous Data Quality Agent")
    st.markdown("---")

    st.subheader("🤖 AI Engine Status")
    provider_info = Config.get_active_provider_info()
    if provider_info["active"]:
        st.success(f"**{provider_info['provider']} Active**\n\n• Model: `{provider_info['model']}`\n• Loaded securely from `.env`")
    else:
        st.warning("**Deterministic Engine Active**\n\nNo API key found in `.env`. Running local rule-based diagnostics.")

    st.markdown("---")
    st.subheader("⚙️ Business Rules")
    
    iqr_multiplier = st.slider(
        "Outlier Threshold (IQR Multiplier)",
        min_value=1.0,
        max_value=3.0,
        value=Config.DEFAULT_IQR_MULTIPLIER,
        step=0.25,
        help="Higher values flag only extreme statistical outliers"
    )

    req_cols_str = st.text_input(
        "Required Identifier Columns",
        placeholder="e.g. customer_id, order_id",
        help="Comma-separated. Missing values in these will be flagged as High/Critical Confirmed Violations."
    )
    non_neg_cols_str = st.text_input(
        "Non-Negative Columns",
        placeholder="e.g. quantity, price, amount",
        help="Comma-separated. Negative values in these will be flagged as Confirmed Violations."
    )

    st.markdown("---")
    st.markdown("### 📚 Quick Guide")
    st.info(
        "1. Upload CSV or Excel file (or click 'Load Sample Dataset').\n"
        "2. Review dataset dimensions.\n"
        "3. Click **Investigate Dataset** to run the 7-node LangGraph agent.\n"
        "4. Review evidence & select proposed fixes to export clean data."
    )

# -------------------------------------------------------------
# MAIN VIEW: Header & Dataset Upload
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">DataSentry AI</div>
    <div class="hero-subtitle">The Silent Data Error Detective — Discovering hidden corruptions, duplicates, and business rule violations before they skew decision-making.</div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None
if "dataset_filename" not in st.session_state:
    st.session_state.dataset_filename = None
if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None
if "cleaned_preview" not in st.session_state:
    st.session_state.cleaned_preview = None

# Top Action Row: File Upload and Sample Dataset button
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, XLSX, or XLS)",
        type=["csv", "xlsx", "xls"],
        help="Files are analyzed in-memory and never permanently stored."
    )

with col_sample:
    st.write("&nbsp;")
    if st.button("🧪 Load Sample Messy Dataset", use_container_width=True):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "messy_sales.csv")
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            st.session_state.dataset_df = sample_df
            st.session_state.dataset_filename = "messy_sales.csv"
            st.session_state.investigation_result = None
            st.session_state.cleaned_preview = None
            st.success("Loaded 'messy_sales.csv' successfully!")
            st.rerun()

# Handle uploaded file
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        df_loaded, meta = load_dataset(file_bytes, filename=uploaded_file.name)
        if st.session_state.dataset_filename != uploaded_file.name:
            st.session_state.dataset_df = df_loaded
            st.session_state.dataset_filename = uploaded_file.name
            st.session_state.investigation_result = None
            st.session_state.cleaned_preview = None
    except DataLoaderError as e:
        st.error(f"Error loading file: {str(e)}")
        st.session_state.dataset_df = None

# -------------------------------------------------------------
# DATASET PREVIEW & INVESTIGATION TRIGGER
# -------------------------------------------------------------
current_df = st.session_state.dataset_df
if current_df is not None:
    st.subheader(f"📊 Dataset Overview: `{st.session_state.dataset_filename}`")
    
    # Overview cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(current_df):,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(current_df.columns):,}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        null_count = int(current_df.isna().sum().sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{null_count:,}</div>
            <div class="metric-label">Missing Cells</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        dup_count = int(current_df.duplicated().sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{dup_count:,}</div>
            <div class="metric-label">Duplicate Rows</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Dataset Sample (First 10 rows)", expanded=False):
        st.dataframe(current_df.head(10), use_container_width=True)

    # Investigation Action Button
    st.write("")
    run_col1, run_col2 = st.columns([1, 2])
    with run_col1:
        investigate_clicked = st.button("🛡️ Investigate Dataset with LangGraph", type="primary", use_container_width=True)

    if investigate_clicked:
        with st.spinner("Executing LangGraph investigation workflow across 7 nodes..."):
            progress_bar = st.progress(0, text="Initializing Agent Workflow...")
            
            # Prepare overrides
            req_cols = [c.strip() for c in req_cols_str.split(",") if c.strip()] if req_cols_str else None
            non_neg_cols = [c.strip() for c in non_neg_cols_str.split(",") if c.strip()] if non_neg_cols_str else None

            config_overrides = {
                "iqr_multiplier": iqr_multiplier,
                "required_columns": req_cols,
                "nonnegative_columns": non_neg_cols,
            }

            progress_bar.progress(20, text="Node 1 & 2: Loading & Profiling Dataset...")
            progress_bar.progress(50, text="Node 3: Running Deterministic Data-Quality Checks...")
            progress_bar.progress(70, text="Node 4: Investigating Findings (AI / Deterministic)...")
            progress_bar.progress(90, text="Node 5 & 6: Prioritizing & Formulating Remediations...")

            # Run full LangGraph StateGraph
            result_state = run_datasentry_investigation(
                df=current_df,
                filename=st.session_state.dataset_filename,
                config_overrides=config_overrides
            )

            progress_bar.progress(100, text="Node 7: Structured Report Generated!")
            st.session_state.investigation_result = result_state
            st.session_state.cleaned_preview = None
            st.rerun()

# -------------------------------------------------------------
# INVESTIGATION REPORT & FINDINGS DASHBOARD
# -------------------------------------------------------------
if st.session_state.investigation_result is not None:
    result_state = st.session_state.investigation_result
    report = result_state.get("final_report", {})
    findings = report.get("findings", [])
    summary_counts = report.get("findings_summary", {})
    llm_enhanced = report.get("metadata", {}).get("llm_enhanced", False)

    st.markdown("---")
    st.header("📋 Investigation Report")

    # Status Banner
    if llm_enhanced:
        st.markdown("""
        <div class="status-banner status-llm-active">
            🤖 <strong>AI-Enhanced Investigation Active:</strong> Business impact and remediation strategies have been enriched by Google Gemini based strictly on empirical evidence.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-banner status-llm-inactive">
            ⚡ <strong>Deterministic Investigation Active:</strong> Findings verified by the rule engine. <em>(AI narrative unavailable — add your GEMINI_API_KEY in .env to activate generative business impact synthesis)</em>.
        </div>
        """, unsafe_allow_html=True)

    # Executive Summary Card
    st.info(f"**Executive Assessment:** {report.get('overall_assessment')}\n\n{report.get('executive_summary')}")

    # Summary metric counters
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Findings", summary_counts.get("total_findings", 0))
    with m2:
        st.metric("Confirmed Violations", summary_counts.get("confirmed_violations", 0), delta_color="inverse")
    with m3:
        st.metric("Potential Anomalies", summary_counts.get("potential_anomalies", 0))
    with m4:
        crit_high = summary_counts.get("by_severity", {}).get("critical", 0) + summary_counts.get("by_severity", {}).get("high", 0)
        st.metric("Critical / High Severity", crit_high)

    # Filter Controls
    st.subheader("🔍 Detailed Findings")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        )
    with f_col2:
        categories = list({f.get("category", "Potential Anomaly") for f in findings})
        category_filter = st.multiselect("Filter by Category", options=categories, default=categories)
    with f_col3:
        issue_types = list({f.get("issue_type", "") for f in findings})
        issue_filter = st.multiselect("Filter by Issue Type", options=issue_types, default=issue_types)

    # Filter findings
    filtered_findings = [
        f for f in findings
        if str(f.get("severity", "")).upper() in severity_filter
        and f.get("category") in category_filter
        and f.get("issue_type") in issue_filter
    ]

    st.write(f"Showing **{len(filtered_findings)}** of **{len(findings)}** findings:")

    # Findings Table
    findings_df = findings_to_dataframe(filtered_findings)
    st.dataframe(findings_df, use_container_width=True)

    # Findings Details Cards
    for idx, f in enumerate(filtered_findings):
        sev = str(f.get("severity", "low")).upper()
        badge_class = f"badge-{sev.lower()}"
        cat = f.get("category")
        col_name = f.get("column") or "Table-wide"
        
        with st.expander(f"[{sev}] {cat}: {f.get('issue_type', '').replace('_', ' ').title()} — {col_name} ({f.get('affected_rows', 0)} affected rows)", expanded=False):
            st.markdown(f"**Evidence:** {f.get('evidence')}")
            st.markdown(f"**Business Impact:** {f.get('business_impact')}")
            st.markdown(f"**Recommended Action:** {f.get('recommended_action')}")
            st.markdown(f"**Confidence:** `{f.get('confidence', 'high')}` | **Verified by AI:** `{f.get('verified_by_llm', False)}`")

            sample_recs = f.get("sample_records", [])
            if sample_recs:
                st.caption("Sample Problematic Records (up to 5):")
                st.dataframe(pd.DataFrame(sample_recs), use_container_width=True)

    # -------------------------------------------------------------
    # EXPORT REPORT
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Investigation Report")
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        json_report_str = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="💾 Download Structured Report (JSON)",
            data=json_report_str,
            file_name=f"datasentry_report_{st.session_state.dataset_filename}.json",
            mime="application/json",
            use_container_width=True
        )

    with exp_col2:
        csv_findings_buffer = findings_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Findings Table (CSV)",
            data=csv_findings_buffer,
            file_name=f"datasentry_findings_{st.session_state.dataset_filename}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # -------------------------------------------------------------
    # INTERACTIVE REMEDIATION ENGINE
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🛠️ Remediation & Proposed Data Cleansing")
    st.markdown(
        "DataSentry AI does **never automatically overwrite** your original dataset. "
        "Select which proposed corrections you wish to apply below to generate a cleaned preview."
    )

    # Collect fixable findings
    fixable_findings = [f for f in findings if f.get("proposed_fix_type") and f.get("proposed_fix_type") != "review_manual"]

    if not fixable_findings:
        st.info("No automated remediations available for the current findings.")
    else:
        st.markdown("##### Select Corrections to Approve:")
        selected_findings = []
        for i, f in enumerate(fixable_findings):
            col_target = f.get("column") or "Dataset"
            label = f"**{f.get('proposed_fix_type', '').replace('_', ' ').title()}** on `{col_target}` — {f.get('recommended_action')}"
            is_checked = st.checkbox(label, value=True, key=f"fix_chk_{f.get('id', i)}")
            if is_checked:
                selected_findings.append(f)

        if st.button("🚀 Apply Approved Fixes & Preview Cleaned Dataset", type="primary"):
            if not selected_findings:
                st.warning("Please select at least one fix to apply.")
            else:
                cleaned_df, change_log, diff_summary = apply_remediations(current_df, selected_findings)
                st.session_state.cleaned_preview = {
                    "df": cleaned_df,
                    "change_log": change_log,
                    "diff": diff_summary
                }

        # Show preview if available
        if st.session_state.cleaned_preview is not None:
            clean_meta = st.session_state.cleaned_preview
            cleaned_df = clean_meta["df"]
            change_log = clean_meta["change_log"]
            diff = clean_meta["diff"]

            st.success(f"Successfully applied {diff['actions_executed']} remediation actions!")

            # Diff metrics
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Original Rows", diff["initial_rows"])
            d2.metric("Cleaned Rows", diff["final_rows"], delta=f"-{diff['rows_removed']}" if diff['rows_removed'] > 0 else "0")
            d3.metric("Original Nulls", diff["initial_nulls"])
            d4.metric("Cleaned Nulls", diff["final_nulls"], delta=f"-{diff['nulls_resolved']}" if diff['nulls_resolved'] > 0 else "0")

            with st.expander("📝 Detailed Remediation Audit Log", expanded=True):
                for log_entry in change_log:
                    st.write(f"- {log_entry}")

            st.write("##### Cleaned Dataset Preview:")
            st.dataframe(cleaned_df.head(15), use_container_width=True)

            # Export cleaned CSV
            clean_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=clean_csv_bytes,
                file_name=f"cleaned_{st.session_state.dataset_filename}",
                mime="text/csv",
                type="primary"
            )

else:
    if current_df is None:
        st.info("👆 Upload a CSV or Excel file above, or click **Load Sample Messy Dataset** to begin.")
