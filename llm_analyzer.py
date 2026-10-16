import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config
from graph_state import FindingDict

logger = logging.getLogger(__name__)

class FindingInsight(BaseModel):
    id: str = Field(description="The unique identifier matching the finding ID provided.")
    refined_business_impact: str = Field(description="Specific, realistic business consequence of this data issue.")
    refined_recommended_action: str = Field(description="Actionable, pragmatic recommendation for data engineers or business analysts.")
    refined_severity: Optional[str] = Field(None, description="Adjusted severity: 'critical', 'high', 'medium', 'low', 'info'.")
    confidence: str = Field("high", description="'high', 'medium', or 'low'.")

class LLMInvestigationResponse(BaseModel):
    overall_executive_summary: str = Field(description="A concise 2-sentence executive summary of the dataset's health.")
    insights: List[FindingInsight] = Field(description="List of enhanced insights mapped by finding ID.")


def get_llm(config_overrides: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """
    Initializes and returns an LLM instance (Gemini ChatGoogleGenerativeAI or ChatOpenAI).
    Strictly reads keys from environment variables or internal config.
    Returns None if no API key is provided or if key is a placeholder.
    """
    Config.reload_env()
    cfg = config_overrides or {}
    provider = cfg.get("llm_provider") or Config.LLM_PROVIDER
    temperature = cfg.get("llm_temperature", Config.LLM_TEMPERATURE)
    timeout = cfg.get("llm_timeout_seconds", Config.LLM_TIMEOUT_SECONDS)

    # 1. Google Gemini Provider
    gemini_key = cfg.get("gemini_api_key") or Config.GEMINI_API_KEY
    if gemini_key:
        placeholder_indicators = ["your-api-key", "placeholder", "xxx", "your_gemini_api_key_here"]
        if not any(p in gemini_key.lower() for p in placeholder_indicators):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_model = cfg.get("llm_model") or Config.LLM_MODEL
                if "gpt" in gemini_model.lower() or "2.0-flash" in gemini_model.lower():
                    gemini_model = "gemini-2.5-flash"
                
                return ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=gemini_key,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=1,
                )
            except Exception as e:
                logger.warning(f"Failed to instantiate ChatGoogleGenerativeAI: {e}")

    # 2. OpenAI-compatible Provider (Fallback or explicit)
    openai_key = cfg.get("openai_api_key") or Config.OPENAI_API_KEY
    if openai_key:
        placeholder_indicators = ["your-api-key", "sk-...", "placeholder", "xxx", "your_openai_api_key_here"]
        if not any(p in openai_key.lower() for p in placeholder_indicators):
            try:
                from langchain_openai import ChatOpenAI
                base_url = cfg.get("openai_base_url") or Config.OPENAI_BASE_URL
                model_name = cfg.get("llm_model") or Config.LLM_MODEL
                kwargs: Dict[str, Any] = {
                    "api_key": openai_key,
                    "model": model_name,
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_retries": 1,
                }
                if base_url:
                    kwargs["base_url"] = base_url
                return ChatOpenAI(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to instantiate ChatOpenAI: {e}")

    return None


def investigate_findings_with_llm(
    findings: List[FindingDict],
    dataset_summary: Dict[str, Any],
    profile: Dict[str, Any],
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[List[FindingDict], Optional[str], bool]:
    """
    Passes empirical findings and minimal aggregated statistics to the LLM
    to generate deep business impact analyses and recommendations.

    Returns:
        Tuple of (enhanced_findings, overall_executive_summary, llm_used_boolean).
    """
    if not findings:
        return findings, "No data quality issues were detected. Dataset appears consistent.", False

    llm = get_llm(config_overrides)
    if llm is None:
        logger.info("LLM is not configured. Retaining deterministic findings.")
        return findings, None, False

    # Construct privacy-preserving context payload (no full datasets, only summaries)
    findings_payload = []
    for f in findings:
        sample_vals = []
        if f.get("sample_records"):
            col = f.get("column")
            if col:
                sample_vals = [r.get(col) for r in f["sample_records"][:3] if col in r]

        findings_payload.append({
            "id": f.get("id"),
            "issue_type": f.get("issue_type"),
            "category": f.get("category"),
            "column": f.get("column"),
            "affected_rows": f.get("affected_rows"),
            "affected_percentage": f.get("affected_percentage"),
            "current_severity": f.get("severity"),
            "evidence": f.get("evidence"),
            "sample_values": sample_vals,
        })

    prompt_context = {
        "dataset_name": dataset_summary.get("filename", "uploaded_dataset"),
        "total_rows": dataset_summary.get("rows"),
        "total_columns": dataset_summary.get("columns"),
        "column_names": dataset_summary.get("column_names", []),
        "detected_findings": findings_payload,
    }

    system_message = (
        "You are DataSentry AI, an elite Data Quality Detective and Chief Data Officer. "
        "Your task is to interpret verified empirical data anomalies. "
        "CRITICAL RULES:\n"
        "1. DO NOT invent, hallucinate, or assume any errors that are not listed in 'detected_findings'.\n"
        "2. Ground every explanation firmly in the empirical evidence provided.\n"
        "3. Provide clear, concrete business consequences (e.g. accounting variances, model drift, pipeline failures).\n"
        "4. Provide realistic, non-destructive remediation recommendations.\n"
        "5. Return your output strictly as a valid JSON object matching the requested schema."
    )

    human_message = (
        "Analyze the following detected data quality issues and return a structured JSON response:\n\n"
        f"```json\n{json.dumps(prompt_context, indent=2, default=str)}\n```\n\n"
        "Return a JSON object with this exact format:\n"
        "{\n"
        '  "overall_executive_summary": "2 sentences describing dataset health and primary risks.",\n'
        '  "insights": [\n'
        '    {\n'
        '      "id": "<matching finding ID>",\n'
        '      "refined_business_impact": "<detailed business impact>",\n'
        '      "refined_recommended_action": "<actionable remediation step>",\n'
        '      "refined_severity": "critical|high|medium|low|info",\n'
        '      "confidence": "high|medium|low"\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    try:
        # Request completion
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ])

        raw_content = response.content
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
                else:
                    text_parts.append(str(item))
            raw_content = "\n".join(text_parts)

        # Extract JSON
        clean_json = raw_content.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        # Fallback to regex if wrapped with other text
        import re
        try:
            parsed_data = json.loads(clean_json)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", clean_json)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
            else:
                raise
        validated = LLMInvestigationResponse(**parsed_data)

        # Merge insights back
        insights_by_id = {item.id: item for item in validated.insights}
        enhanced_findings: List[FindingDict] = []

        for f in findings:
            f_copy = dict(f)  # type: ignore
            finding_id = f.get("id")
            if finding_id in insights_by_id:
                insight = insights_by_id[finding_id]
                if insight.refined_business_impact:
                    f_copy["business_impact"] = insight.refined_business_impact
                if insight.refined_recommended_action:
                    f_copy["recommended_action"] = insight.refined_recommended_action
                if insight.refined_severity in {"critical", "high", "medium", "low", "info"}:
                    f_copy["severity"] = insight.refined_severity
                f_copy["confidence"] = insight.confidence
                f_copy["verified_by_llm"] = True
            enhanced_findings.append(f_copy)  # type: ignore

        return enhanced_findings, validated.overall_executive_summary, True

    except Exception as e:
        logger.warning(f"LLM investigation failed or timed out: {e}. Falling back to deterministic outputs.")
        return findings, None, False
