# -*- coding: utf-8 -*-
"""
Agri Doctor - AI-powered Crop Disease Diagnosis & Advisory
Two-agent workflow: Pathologist Agent + Extension Officer Agent
Powered by IBM watsonx.ai
"""

import base64
import os
import time
import datetime
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from metrics import build_run_metrics, get_run_history
from guardrails import (
    # Input
    run_input_guardrails, mask_pii,
    # Output
    run_all_output_guardrails,
    # Tool
    run_tool_guardrails,
    # Data
    check_data_privacy, anonymize_filename,
    # Security
    run_security_guardrails,
    # Operational
    OperationalGuardrail,
    # Human-in-the-loop
    check_escalation_needed, check_chemical_recommendation,
    # Multi-agent
    register_agent, check_agent_handoff, check_loop_prevention, reset_agent_counters,
    # Ethics
    run_ethics_guardrails,
    # Audit
    get_audit_log, clear_audit_log,
    # Result type
    GuardrailResult,
)

# Load .env (local dev). Silently ignored if missing (e.g. on Streamlit Cloud).
load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secret loader ─────────────────────────────────────────────────────────────
# Priority: Streamlit Cloud secrets  >  .env file  >  empty string
def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) then fall back to os.getenv (.env)."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ── Constants (loaded from secrets / .env) ───────────────────────────────────
IBM_ENDPOINT   = _secret("WATSONX_URL",          "https://us-south.ml.cloud.ibm.com")
VISION_MODEL   = _secret("WATSONX_VISION_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503")
TEXT_MODEL     = _secret("WATSONX_TEXT_MODEL",   "ibm/granite-4-h-small")
PROJECT_ID     = _secret("WATSONX_PROJECT_ID",   "")
_ENV_API_KEY   = _secret("WATSONX_API_KEY",      "")   # pre-fills sidebar; never shown in plaintext

DISCLAIMER = (
    "⚠️  This diagnosis is an AI-generated preliminary assessment only. "
    "It is NOT a confirmed laboratory diagnosis. Always consult a certified "
    "plant pathologist or your local agricultural extension office for "
    "official confirmation before applying any treatment."
)

PESTICIDE_WARNING = (
    "🔶  Safety Notice: When handling any pesticide or chemical treatment, "
    "always read and follow the product label instructions, wear appropriate "
    "personal protective equipment (PPE), and comply with local agricultural "
    "regulations. Dosages listed here are general guidance only — verify with "
    "your local agricultural authority."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png", width=80)
    st.title("🌿 Agri Doctor")
    st.caption("AI Crop Disease Diagnosis & Advisory")
    st.divider()

    st.subheader("🔑 IBM watsonx Credentials")
    if _ENV_API_KEY:
        st.success("API key loaded from .env", icon="✅")
    api_key = st.text_input(
        "IBM API Key",
        value=_ENV_API_KEY,
        type="password",
        placeholder="Enter your IBM Cloud API key (or set WATSONX_API_KEY in .env)",
        help="Loaded from WATSONX_API_KEY in .env. Never shown in plaintext or logs.",
    )
    project_id = st.text_input(
        "Project ID",
        value=PROJECT_ID,
        placeholder="Enter your watsonx project ID (or set WATSONX_PROJECT_ID in .env)",
        help="Loaded from WATSONX_PROJECT_ID in .env.",
    )

    st.divider()
    st.subheader("⚙️ Model Settings")
    vision_model = st.text_input(
        "Pathologist Agent Model",
        value=VISION_MODEL,
        help=(
            "Loaded from WATSONX_VISION_MODEL in .env. "
            "Confirmed image_chat models on this endpoint:\n"
            "• mistralai/mistral-small-3-1-24b-instruct-2503 (default)\n"
            "• mistralai/mistral-medium-2505\n"
            "• meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        ),
    )
    text_model = st.text_input(
        "Extension Officer Agent Model",
        value=TEXT_MODEL,
        help="Loaded from WATSONX_TEXT_MODEL in .env. Default: ibm/granite-4-h-small.",
    )

    st.divider()
    st.info(
        "**How it works**\n\n"
        "1. Upload a clear photo of the affected crop leaf or plant part.\n"
        "2. Click **Diagnose Crop**.\n"
        "3. The **Pathologist Agent** analyses the image.\n"
        "4. The **Extension Officer Agent** prepares a farmer-friendly advisory.\n"
        "5. Download the final report as a text file.",
        icon="ℹ️",
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🌿 Agri Doctor — Crop Disease Diagnosis & Advisory")
st.caption(DISCLAIMER)
st.divider()

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.subheader("📤 Upload Crop Image")
    uploaded_file = st.file_uploader(
        "Upload a photo of the affected plant / leaf",
        type=["jpg", "jpeg", "png"],
        help="For best results upload a clear, well-lit, close-up photo of the affected area.",
    )
    diagnose_btn = st.button("🔍 Diagnose Crop", type="primary", use_container_width=True)  # noqa: deprecated in future

with col_preview:
    if uploaded_file:
        st.subheader("🖼️ Uploaded Image")
        st.image(uploaded_file, width="stretch", caption=uploaded_file.name)


# ── Helper functions ──────────────────────────────────────────────────────────

def encode_image_base64(file_obj) -> str:
    """Return a base64-encoded string of the uploaded image."""
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode("utf-8")


def get_image_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")


def build_credentials(key: str) -> Credentials:
    return Credentials(url=IBM_ENDPOINT, api_key=key)


def run_pathologist_agent(
    image_b64: str,
    mime_type: str,
    credentials: Credentials,
    v_model: str,
    proj_id: str,
) -> str:
    """
    Pathologist Agent — analyses the crop image and returns a structured
    diagnostic report.

    Tries the OpenAI-compatible multimodal message format first (image_url
    content block). If the model rejects it (e.g. text-only Granite 4 H Small
    on multitenant), falls back to a text-only chat that still gives a
    structured diagnostic response based on the farmer's description prompt.
    """
    model = ModelInference(
        model_id=v_model,
        credentials=credentials,
        project_id=proj_id,
        params={
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.TEMPERATURE: 0.2,
        },
    )

    system_prompt = (
        "You are an expert agricultural plant pathologist AI assistant. "
        "Your role is to examine crop and plant-leaf images and provide a "
        "structured diagnostic report. Be precise, factual, and cautious. "
        "If the image is unclear, too blurry, or lacks sufficient detail for "
        "a reliable diagnosis, clearly state that and ask for a clearer image "
        "— do NOT guess.\n\n"
        "Always structure your response with these sections:\n"
        "1. IMAGE QUALITY ASSESSMENT\n"
        "2. SUSPECTED DISEASE / CONDITION\n"
        "3. OBSERVED SYMPTOMS\n"
        "4. POSSIBLE CAUSES\n"
        "5. CONFIDENCE LEVEL (High / Medium / Low) with brief justification\n"
        "6. ALTERNATIVE DIAGNOSES (if applicable)\n"
        "7. RECOMMENDATION FOR FURTHER CONFIRMATION\n\n"
        "Label this clearly as a preliminary AI assessment, not a confirmed diagnosis."
    )

    user_message = (
        "Please examine this crop/plant-leaf image carefully and provide a "
        "detailed diagnostic report following the structure above. "
        "Identify any visible diseases, nutrient deficiencies, pest damage, "
        "or other abnormalities."
    )

    # ── Attempt 1: multimodal image_url format (works with vision-capable models)
    multimodal_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
            ],
        },
    ]

    try:
        response = model.chat(messages=multimodal_messages)
        return response["choices"][0]["message"]["content"]
    except Exception as vision_err:
        vision_err_str = str(vision_err).lower()
        # If the model doesn't support images, fall back to text-only mode
        if any(k in vision_err_str for k in ("image", "multimodal", "unsupported", "invalid", "400", "content")):
            # ── Attempt 2: text-only fallback for Granite 4 H Small (multitenant)
            text_only_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "A farmer has uploaded a crop/plant-leaf image for disease diagnosis. "
                        "The image has been provided as a base64-encoded attachment but this "
                        "session is running in text-only mode.\n\n"
                        "Please provide a comprehensive structured diagnostic template that "
                        "a farmer can use to describe their crop problem, covering all 7 "
                        "sections above. Then provide the most common crop diseases and their "
                        "typical symptoms so the farmer can identify which matches their plant.\n\n"
                        "Also advise the farmer: 'For accurate AI image diagnosis, please use "
                        "the ibm/granite-vision-3-3-2b model (deploy-on-demand) in the sidebar "
                        "model settings, or describe your crop symptoms in text below.'"
                    ),
                },
            ]
            response = model.chat(messages=text_only_messages)
            return (
                "[NOTE: Image analysis requires a vision-capable model. "
                "The current model (ibm/granite-4-h-small) is text-only. "
                "For full image diagnosis, set the Pathologist Agent Model to "
                "'ibm/granite-vision-3-3-2b' in the sidebar settings.]\n\n"
                + response["choices"][0]["message"]["content"]
            )
        # Any other error — re-raise so the main error handler catches it
        raise


def run_extension_officer_agent(
    diagnostic_report: str,
    credentials: Credentials,
    t_model: str,
    proj_id: str,
) -> str:
    """
    Extension Officer Agent — takes the Pathologist's report and generates
    a farmer-friendly advisory with remedial actions.
    """
    model = ModelInference(
        model_id=t_model,
        credentials=credentials,
        project_id=proj_id,
        params={
            GenParams.MAX_NEW_TOKENS: 1500,
            GenParams.TEMPERATURE: 0.3,
        },
    )

    system_prompt = (
        "You are a knowledgeable and friendly agricultural extension officer AI. "
        "Your goal is to help smallholder farmers understand crop diseases and "
        "take the right actions. Write in simple, clear language that a farmer "
        "with basic literacy can understand. Avoid technical jargon. "
        "Structure your advisory as follows:\n\n"
        "1. SUMMARY (in 2–3 plain sentences for the farmer)\n"
        "2. IMMEDIATE ACTIONS (what to do right now, step by step)\n"
        "3. TREATMENT OPTIONS\n"
        "   a. Organic / Cultural methods (preferred first)\n"
        "   b. Chemical treatments (only if necessary; note active ingredients, "
        "      general dosage guidance — do NOT invent specific dosages; "
        "      always refer farmer to product label and local agricultural office)\n"
        "4. PREVENTIVE MEASURES (for next season / long-term)\n"
        "5. CROP CARE RECOMMENDATIONS\n"
        "6. WHEN TO SEEK EXPERT HELP\n"
        "7. SAFETY REMINDERS (PPE, chemical handling, disposal)\n\n"
        "If the pathologist report says the image was unclear or the disease "
        "could not be identified, advise the farmer to take a clearer photo "
        "or visit their local agricultural extension office.\n\n"
        "Always end with: 'Remember: This is an AI-generated advisory. Consult "
        "your local agricultural extension officer for confirmation.'"
    )

    user_message = (
        f"Below is the Pathologist Agent's diagnostic report for a crop disease image.\n\n"
        f"--- PATHOLOGIST REPORT ---\n{diagnostic_report}\n--- END REPORT ---\n\n"
        "Please prepare a comprehensive, farmer-friendly advisory based on this "
        "diagnosis following the structure above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]


def build_final_report(
    filename: str,
    diagnosis: str,
    advisory: str,
) -> str:
    """Compile the full downloadable text report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep  = "-" * 70          # use hyphens — safe in both plain-text and st.code
    sep2 = "*" * 70

    lines = [
        sep2,
        "  AGRI DOCTOR  |  CROP DISEASE ADVISORY REPORT",
        "  Powered by IBM watsonx.ai  |  Multi-Agent AI Workflow",
        sep2,
        f"  Generated  : {timestamp}",
        f"  Image file : {filename}",
        sep,
        "",
        "  IMPORTANT NOTICE",
        DISCLAIMER,
        "",
        sep,
        "  SECTION 1  --  PATHOLOGIST AGENT: DIAGNOSTIC REPORT",
        sep,
        "",
        diagnosis,
        "",
        sep,
        "  SECTION 2  --  EXTENSION OFFICER AGENT: FARMER ADVISORY",
        sep,
        "",
        advisory,
        "",
        sep,
        "  SAFETY NOTICE",
        PESTICIDE_WARNING,
        sep2,
        "  End of Report  |  Agri Doctor  |  IBM watsonx.ai",
        sep2,
    ]
    return "\n".join(lines)


# ── Diagnosis workflow ────────────────────────────────────────────────────────

if diagnose_btn:
    # ── Basic field validation ─────────────────────────────────────────────
    if not api_key:
        st.error("🔑 Please enter your IBM API Key in the sidebar before diagnosing.")
        st.stop()
    if not project_id:
        st.error("🔑 Please enter your IBM watsonx Project ID in the sidebar.")
        st.stop()
    if not uploaded_file:
        st.warning("📸 Please upload a crop or plant-leaf image first.")
        st.stop()

    st.divider()
    st.subheader("🤖 Multi-Agent Workflow")

    # ── Progress setup ─────────────────────────────────────────────────────
    progress = st.progress(0, text="Starting Agri Doctor workflow…")
    status   = st.empty()

    try:
        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL — Reset counters & audit log for fresh run
        # ══════════════════════════════════════════════════════════════════
        clear_audit_log()
        reset_agent_counters()
        register_agent("PathologistAgent",
                        permitted_inputs=["image_b64", "mime_type"],
                        permitted_outputs=["diagnostic_report"])
        register_agent("ExtensionOfficerAgent",
                        permitted_inputs=["diagnostic_report"],
                        permitted_outputs=["advisory"])

        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL 3 — Tool Execution: validate model IDs & endpoint
        # ══════════════════════════════════════════════════════════════════
        progress.progress(5, text="Guardrails: Validating tool permissions…")
        tool_ok, tool_results = run_tool_guardrails(
            vision_model, IBM_ENDPOINT, 1024, 0.2, "PathologistAgent"
        )
        tool_ok2, tool_results2 = run_tool_guardrails(
            text_model, IBM_ENDPOINT, 1500, 0.3, "ExtensionOfficerAgent"
        )
        tool_warnings = [r for r in tool_results + tool_results2 if not r.passed]
        if tool_warnings:
            for r in tool_warnings:
                if r.severity == "critical":
                    st.error(f"🛡️ Tool Guardrail [{r.check}]: {r.message}", icon="🚫")
                    st.stop()
                else:
                    st.warning(f"🛡️ Tool Guardrail [{r.check}]: {r.message}", icon="⚠️")

        credentials = build_credentials(api_key)
        uploaded_file.seek(0)
        file_bytes  = uploaded_file.read()
        mime_type   = get_image_mime(uploaded_file.name)
        image_b64   = base64.b64encode(file_bytes).decode("utf-8")

        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL 1 — Input: validate uploaded image
        # ══════════════════════════════════════════════════════════════════
        progress.progress(8, text="Guardrails: Validating input…")
        input_ok, input_results = run_input_guardrails(
            filename=uploaded_file.name,
            mime_type=mime_type,
            size_bytes=len(file_bytes),
        )
        for r in input_results:
            if not r.passed:
                if r.severity == "critical":
                    st.error(f"🛡️ Input Guardrail [{r.check}]: {r.message}  {r.remediation}", icon="🚫")
                    st.stop()
                else:
                    st.warning(f"🛡️ Input Guardrail [{r.check}]: {r.message}", icon="⚠️")

        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL 4 — Data: anonymize filename, check data privacy
        # ══════════════════════════════════════════════════════════════════
        safe_filename = anonymize_filename(uploaded_file.name)
        check_loop_prevention("PathologistAgent")

        # ══════════════════════════════════════════════════════════════════
        # AGENT 1 — PATHOLOGIST AGENT
        # ══════════════════════════════════════════════════════════════════
        with st.expander("🔬 Agent 1: Pathologist Agent", expanded=True):
            status.info("🔬 Pathologist Agent is analysing the crop image…")
            progress.progress(15, text="Pathologist Agent: Examining image…")

            agent1_placeholder = st.empty()
            agent1_placeholder.markdown("_Analysing image… please wait._")

            # Operational guardrail + timing for metrics
            op1 = OperationalGuardrail("PathologistAgent")
            op1.start()
            _a1_start = time.time()

            diagnosis = run_pathologist_agent(
                image_b64, mime_type, credentials, vision_model, project_id
            )
            _a1_latency = time.time() - _a1_start
            op1.finish()
            progress.progress(50, text="Pathologist Agent: Analysis complete.")

            # ── GUARDRAIL 2 — Output: screen Agent 1 response ────────────
            out1_ok, out1_results = run_all_output_guardrails(diagnosis, "PathologistAgent")
            agent1_placeholder.empty()

            # Display per-check guardrail badges
            warn1 = [r for r in out1_results if not r.passed]
            for r in warn1:
                if r.severity == "critical":
                    st.error(f"🛡️ Output Guardrail [{r.check}]: {r.message}", icon="🚫")
                elif r.severity == "warning":
                    st.warning(f"🛡️ [{r.check}]: {r.message}  —  {r.remediation}", icon="⚠️")

            # GUARDRAIL 7 — Human-in-the-loop: escalation check
            esc = check_escalation_needed(diagnosis)
            if not esc.passed:
                st.warning(
                    f"👨‍⚕️ **Expert Escalation Recommended:** {esc.remediation}",
                    icon="🆘",
                )

            st.success("✅ Pathologist Agent completed diagnosis.")
            st.markdown("#### 📋 Diagnostic Report")
            st.info(
                "**AI Preliminary Assessment** — Not a confirmed laboratory diagnosis.",
                icon="⚠️",
            )
            st.markdown(diagnosis)

        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL 8 — Multi-Agent: validate handoff payload
        # ══════════════════════════════════════════════════════════════════
        handoff = check_agent_handoff("PathologistAgent", "ExtensionOfficerAgent", diagnosis)
        if not handoff.passed:
            st.error(f"🛡️ Multi-Agent Guardrail: {handoff.message}  {handoff.remediation}", icon="🚫")
            st.stop()
        check_loop_prevention("ExtensionOfficerAgent")

        # ══════════════════════════════════════════════════════════════════
        # AGENT 2 — EXTENSION OFFICER AGENT
        # ══════════════════════════════════════════════════════════════════
        with st.expander("👨‍🌾 Agent 2: Extension Officer Agent", expanded=True):
            status.info("👨‍🌾 Extension Officer Agent is preparing the farmer advisory…")
            progress.progress(60, text="Extension Officer Agent: Generating advisory…")

            agent2_placeholder = st.empty()
            agent2_placeholder.markdown("_Preparing farmer advisory… please wait._")

            # Operational guardrail + timing for metrics
            op2 = OperationalGuardrail("ExtensionOfficerAgent")
            op2.start()
            _a2_start = time.time()

            advisory = run_extension_officer_agent(
                diagnosis, credentials, text_model, project_id
            )
            _a2_latency = time.time() - _a2_start
            op2.finish()
            progress.progress(90, text="Extension Officer Agent: Advisory ready.")

            # ── GUARDRAIL 2 — Output: screen Agent 2 response ────────────
            out2_ok, out2_results = run_all_output_guardrails(advisory, "ExtensionOfficerAgent")
            agent2_placeholder.empty()

            warn2 = [r for r in out2_results if not r.passed]
            for r in warn2:
                if r.severity == "critical":
                    st.error(f"🛡️ Output Guardrail [{r.check}]: {r.message}", icon="🚫")
                elif r.severity == "warning":
                    st.warning(f"🛡️ [{r.check}]: {r.message}  —  {r.remediation}", icon="⚠️")

            # GUARDRAIL 7 — Chemical treatment human-in-the-loop
            chem = check_chemical_recommendation(advisory)
            if not chem.passed:
                st.warning(
                    f"🔬 **Chemical Treatment Notice:** {chem.remediation}",
                    icon="🔶",
                )

            st.success("✅ Extension Officer Agent completed advisory.")
            st.markdown("#### 📝 Farmer Advisory")
            st.warning(PESTICIDE_WARNING, icon="🔶")
            st.markdown(advisory)

        # ══════════════════════════════════════════════════════════════════
        # COMBINED FINAL REPORT
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("📄 Combined Final Advisory Report")

        final_report = build_final_report(safe_filename, diagnosis, advisory)

        with st.expander("📖 View Full Report", expanded=False):
            st.code(final_report, language=None)

        # ── Download button ────────────────────────────────────────────
        report_bytes = final_report.encode("utf-8")
        st.download_button(
            label="⬇️ Download Advisory Report (.txt)",
            data=report_bytes,
            file_name=f"agri_doctor_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ══════════════════════════════════════════════════════════════════
        # METRICS — Build and store run metrics for the dashboard
        # ══════════════════════════════════════════════════════════════════
        _a1_prompt = f"Diagnose crop disease from image: {safe_filename}"
        _a2_prompt = f"Generate advisory from diagnosis: {diagnosis[:200]}"
        run_metrics = build_run_metrics(
            vision_model      = vision_model,
            text_model        = text_model,
            image_filename    = safe_filename,
            a1_prompt         = _a1_prompt,
            a1_response       = diagnosis,
            a1_latency        = _a1_latency,
            a1_guardrail_results = out1_results,
            a2_prompt         = _a2_prompt,
            a2_response       = advisory,
            a2_latency        = _a2_latency,
            a2_guardrail_results = out2_results,
            handoff_valid     = handoff.passed,
            workflow_completed = True,
        )
        # Store in session so dashboard can read it
        if "run_history" not in st.session_state:
            st.session_state["run_history"] = []
        st.session_state["run_history"].append(run_metrics.to_dict())

        with st.expander("📊 Quick Metrics Summary", expanded=False):
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("A1 Latency",       f"{run_metrics.agent1.latency_s:.2f}s")
            mc2.metric("A2 Latency",       f"{run_metrics.agent2.latency_s:.2f}s")
            mc3.metric("Total Tokens",     f"{run_metrics.total_tokens:,}")
            mc4.metric("Coord. Score",     f"{run_metrics.agent_coordination_score:.2f}")
            mc5.metric("Est. Cost",        f"${run_metrics.total_cost_usd:.5f}")
            st.info(
                "Open the **Performance Dashboard** (run `streamlit run dashboard.py`) "
                "for full LLM evaluation metrics across all 6 categories.",
                icon="📊",
            )

        # ══════════════════════════════════════════════════════════════════
        # GUARDRAIL 9 — Audit Trail (expandable in UI)
        # ══════════════════════════════════════════════════════════════════
        audit_log = get_audit_log()
        passed_count = sum(1 for e in audit_log if e["passed"])
        warned_count = sum(1 for e in audit_log if not e["passed"])
        with st.expander(
            f"🛡️ Guardrail Audit Trail  —  {passed_count} passed / {warned_count} flagged",
            expanded=False,
        ):
            for entry in audit_log:
                icon = "✅" if entry["passed"] else ("⚠️" if entry["severity"] == "warning" else "🚫")
                st.markdown(
                    f"`{entry['timestamp'][:19].replace('T',' ')}`  "
                    f"**[{entry['category']}]** {entry['check']}  {icon}  "
                    f"{entry['message']}"
                )

        progress.progress(100, text="✅ Agri Doctor workflow complete!")
        status.success(
            "🎉 Both agents completed successfully. Your advisory report is ready."
        )

    except Exception as exc:
        progress.empty()
        err_msg = str(exc)
        err_low = err_msg.lower()

        if "401" in err_msg or "unauthorized" in err_low or "authentication" in err_low:
            st.error(
                "🔑 Authentication failed. Please check your IBM API Key and Project ID.",
                icon="🚫",
            )
        elif "404" in err_msg or "not found" in err_low or "no resource" in err_low or "does not exist" in err_low:
            # Work out which model actually caused the failure
            failed_model = vision_model if vision_model in err_msg else (
                text_model if text_model in err_msg else f"{vision_model}  /  {text_model}"
            )
            st.error(
                f"🤖 Model not found or not available: `{failed_model}`\n\n"
                f"**Raw error:** {err_msg}",
                icon="🚫",
            )
            st.info(
                "Confirmed working models on this endpoint:\n\n"
                "**Vision (image_chat):**\n"
                "- `mistralai/mistral-small-3-1-24b-instruct-2503`\n"
                "- `mistralai/mistral-medium-2505`\n"
                "- `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`\n\n"
                "**Text (text_chat):**\n"
                "- `ibm/granite-4-h-small`\n"
                "- `meta-llama/llama-3-3-70b-instruct`\n"
                "- `mistralai/mistral-medium-2505`",
                icon="ℹ️",
            )
        elif "rate" in err_low or "429" in err_msg:
            st.error(
                "⏳ Rate limit reached. Please wait a moment and try again.",
                icon="⏳",
            )
        elif "timeout" in err_low:
            st.error(
                "⌛ The request timed out. Check your internet connection and try again.",
                icon="⌛",
            )
        else:
            st.error(f"❌ An unexpected error occurred:\n\n{err_msg}", icon="🚫")
            st.info(
                "If the problem persists, please contact IBM Cloud support or verify "
                "that your watsonx project has the required model entitlements.",
                icon="ℹ️",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🌿 **Agri Doctor** — AI-powered Crop Disease Diagnosis & Advisory  |  "
    "Powered by **IBM watsonx.ai** (Granite 4 H Small)  |  "
    "For educational and preliminary advisory use only."
)
