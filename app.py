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

# ── UI — Custom CSS & background animations ──────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root theme ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ══════════════════════════════════════════════════
   ANIMATED PARTICLE BACKGROUND
══════════════════════════════════════════════════ */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2b1a 40%, #0a2e0a 70%, #1a3a0a 100%);
    min-height: 100vh;
}

/* Floating leaf particles */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(34,197,94,0.08) 0%, transparent 50%),
        radial-gradient(circle at 85% 10%, rgba(59,130,212,0.10) 0%, transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(34,197,94,0.06) 0%, transparent 55%),
        radial-gradient(circle at 90% 60%, rgba(16,185,129,0.08) 0%, transparent 40%),
        radial-gradient(circle at 10% 70%, rgba(59,130,212,0.07) 0%, transparent 45%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes bgPulse {
    0%   { opacity: 0.6; transform: scale(1); }
    100% { opacity: 1.0; transform: scale(1.05); }
}

/* Floating dots overlay */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1.5px 1.5px at 10% 15%, rgba(134,239,172,0.35) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 25% 45%, rgba(134,239,172,0.25) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 40% 25%, rgba(96,165,250,0.30) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 60% 70%, rgba(134,239,172,0.20) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 75% 35%, rgba(52,211,153,0.30) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 88% 80%, rgba(96,165,250,0.25) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 5%  85%, rgba(134,239,172,0.20) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 92% 20%, rgba(52,211,153,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 33% 90%, rgba(134,239,172,0.15) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 10%, rgba(96,165,250,0.20) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 55%, rgba(134,239,172,0.15) 0%, transparent 100%),
        radial-gradient(1px 1px at 18% 60%, rgba(52,211,153,0.18) 0%, transparent 100%);
    animation: floatDots 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes floatDots {
    0%   { transform: translateY(0px)   translateX(0px); }
    33%  { transform: translateY(-18px) translateX(8px); }
    66%  { transform: translateY(-8px)  translateX(-12px); }
    100% { transform: translateY(-22px) translateX(6px); }
}

/* ══════════════════════════════════════════════════
   HERO HEADER BANNER
══════════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(10,22,40,0.92) 0%,
        rgba(13,43,26,0.92) 50%,
        rgba(10,46,10,0.88) 100%);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(34,197,94,0.1);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(
        from 0deg at 50% 50%,
        transparent 0deg,
        rgba(34,197,94,0.04) 60deg,
        transparent 120deg,
        rgba(59,130,212,0.04) 180deg,
        transparent 240deg,
        rgba(34,197,94,0.03) 300deg,
        transparent 360deg
    );
    animation: rotateBg 20s linear infinite;
}

@keyframes rotateBg {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #86efac 0%, #3b82d4 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    animation: shimmer 4s ease-in-out infinite;
    background-size: 200% auto;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    50%  { background-position: 100% center; }
    100% { background-position: 0% center; }
}

.hero-subtitle {
    color: rgba(134,239,172,0.85);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0 0 1.2rem 0;
    letter-spacing: 0.02em;
}

.hero-badges {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.30);
    color: #86efac;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    transition: all 0.3s ease;
}

.hero-badge.blue {
    background: rgba(59,130,212,0.12);
    border-color: rgba(59,130,212,0.30);
    color: #93c5fd;
}

.hero-badge.teal {
    background: rgba(52,211,153,0.12);
    border-color: rgba(52,211,153,0.30);
    color: #6ee7b7;
}

/* ══════════════════════════════════════════════════
   GLASS CARDS
══════════════════════════════════════════════════ */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(134,239,172,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.glass-card:hover {
    border-color: rgba(134,239,172,0.35);
    background: rgba(255,255,255,0.06);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(34,197,94,0.12);
}

/* ══════════════════════════════════════════════════
   STREAMLIT COMPONENT OVERRIDES
══════════════════════════════════════════════════ */

/* Main content area */
.main .block-container {
    background: transparent;
    padding-top: 1.5rem;
}

/* Text colors on dark bg */
.stMarkdown, .stText, label, .stSelectbox label,
.stTextInput label, .stFileUploader label {
    color: rgba(220,240,220,0.92) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #86efac !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(8,18,32,0.97) 0%,
        rgba(8,25,14,0.97) 100%) !important;
    border-right: 1px solid rgba(34,197,94,0.18) !important;
}

[data-testid="stSidebar"] * {
    color: rgba(200,235,200,0.90) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #86efac !important;
}

/* Input fields */
.stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(134,239,172,0.25) !important;
    border-radius: 10px !important;
    color: #e2f5e2 !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput input:focus {
    border-color: rgba(134,239,172,0.60) !important;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.10) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 50%, #166534 100%) !important;
    border: 1px solid rgba(134,239,172,0.35) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.30) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(22,163,74,0.45) !important;
    border-color: rgba(134,239,172,0.60) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}

/* Secondary / other buttons */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(134,239,172,0.20) !important;
    border-radius: 10px !important;
    color: #86efac !important;
    transition: all 0.2s ease !important;
}

.stButton > button:not([kind="primary"]):hover {
    background: rgba(34,197,94,0.10) !important;
    border-color: rgba(134,239,172,0.45) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    border: 1px solid rgba(96,165,250,0.40) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(29,78,216,0.25) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(29,78,216,0.40) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(134,239,172,0.15) !important;
    border-radius: 12px !important;
    color: #86efac !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(34,197,94,0.08) !important;
    border-color: rgba(134,239,172,0.35) !important;
}

.streamlit-expanderContent {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid rgba(134,239,172,0.10) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(134,239,172,0.15) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover {
    border-color: rgba(134,239,172,0.35) !important;
    background: rgba(34,197,94,0.06) !important;
    transform: translateY(-2px) !important;
}

[data-testid="metric-container"] label {
    color: rgba(134,239,172,0.75) !important;
    font-size: 0.82rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2f5e2 !important;
    font-weight: 600 !important;
}

/* Info / warning / success / error boxes */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(6px) !important;
}

[data-testid="stNotification"] {
    border-radius: 12px !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #16a34a, #3b82d4, #34d399) !important;
    border-radius: 4px !important;
    background-size: 200% auto !important;
    animation: progressShimmer 2s linear infinite !important;
}

@keyframes progressShimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: rgba(34,197,94,0.04) !important;
    border: 2px dashed rgba(134,239,172,0.30) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(34,197,94,0.08) !important;
    border-color: rgba(134,239,172,0.60) !important;
}

/* Divider */
hr {
    border-color: rgba(134,239,172,0.15) !important;
}

/* Code blocks */
.stCode {
    background: rgba(0,0,0,0.40) !important;
    border: 1px solid rgba(134,239,172,0.15) !important;
    border-radius: 12px !important;
}

/* Dataframes */
.stDataFrame {
    border: 1px solid rgba(134,239,172,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }
::-webkit-scrollbar-thumb {
    background: rgba(134,239,172,0.30);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(134,239,172,0.55); }

/* ── Animated leaf particles (SVG-free, CSS only) ── */
.leaf-particle {
    position: fixed;
    width: 8px; height: 8px;
    background: rgba(134,239,172,0.25);
    border-radius: 50% 0 50% 0;
    pointer-events: none;
    z-index: 0;
    animation: leafFloat linear infinite;
}

@keyframes leafFloat {
    0%   { transform: translateY(110vh) rotate(0deg);   opacity: 0;   }
    10%  { opacity: 0.6; }
    90%  { opacity: 0.3; }
    100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
}

/* Agent step indicator */
.agent-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(134,239,172,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    transition: all 0.3s ease;
}

.agent-step:hover {
    background: rgba(34,197,94,0.07);
    border-color: rgba(134,239,172,0.35);
}

.step-num {
    background: linear-gradient(135deg, #16a34a, #3b82d4);
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}

/* Caption */
.stCaption {
    color: rgba(134,239,172,0.55) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(200,235,200,0.70) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(34,197,94,0.15) !important;
    color: #86efac !important;
}

/* Select box */
.stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(134,239,172,0.20) !important;
    border-radius: 10px !important;
}

/* ── Pulse animation on diagnose button ── */
@keyframes buttonPulse {
    0%, 100% { box-shadow: 0 4px 15px rgba(22,163,74,0.30); }
    50%       { box-shadow: 0 4px 30px rgba(22,163,74,0.60), 0 0 0 8px rgba(22,163,74,0.05); }
}

.stButton > button[kind="primary"] {
    animation: buttonPulse 2.5s ease-in-out infinite !important;
}

/* Override pulse on hover */
.stButton > button[kind="primary"]:hover {
    animation: none !important;
}
</style>

<div class="leaf-particle" style="left:5%;  animation-duration:14s; animation-delay:0s;   width:10px; height:10px;"></div>
<div class="leaf-particle" style="left:15%; animation-duration:18s; animation-delay:2s;   width:6px;  height:6px;"></div>
<div class="leaf-particle" style="left:28%; animation-duration:12s; animation-delay:4s;   width:8px;  height:8px;  background:rgba(96,165,250,0.20);"></div>
<div class="leaf-particle" style="left:42%; animation-duration:20s; animation-delay:1s;   width:5px;  height:5px;"></div>
<div class="leaf-particle" style="left:57%; animation-duration:16s; animation-delay:3s;   width:9px;  height:9px;"></div>
<div class="leaf-particle" style="left:68%; animation-duration:13s; animation-delay:5s;   width:6px;  height:6px;  background:rgba(52,211,153,0.22);"></div>
<div class="leaf-particle" style="left:78%; animation-duration:19s; animation-delay:0.5s; width:7px;  height:7px;"></div>
<div class="leaf-particle" style="left:88%; animation-duration:15s; animation-delay:2.5s; width:10px; height:10px; background:rgba(96,165,250,0.18);"></div>
<div class="leaf-particle" style="left:93%; animation-duration:11s; animation-delay:6s;   width:5px;  height:5px;"></div>
<div class="leaf-particle" style="left:35%; animation-duration:22s; animation-delay:7s;   width:7px;  height:7px;  background:rgba(52,211,153,0.15);"></div>
""", unsafe_allow_html=True)

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

# Hero banner
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🌿 Agri Doctor</div>
  <div class="hero-subtitle">
    AI-Powered Crop Disease Diagnosis &amp; Farmer Advisory System
  </div>
  <div class="hero-badges">
    <span class="hero-badge">🔬 Pathologist Agent</span>
    <span class="hero-badge">👨‍🌾 Extension Officer Agent</span>
    <span class="hero-badge blue">🤖 IBM watsonx.ai</span>
    <span class="hero-badge teal">🛡️ 9 Guardrail Categories</span>
    <span class="hero-badge blue">📊 LLM Evaluation Metrics</span>
    <span class="hero-badge">🌾 Multi-Agent AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="glass-card" style="padding:0.9rem 1.2rem; margin-bottom:0.5rem;">
  <span style="color:rgba(251,191,36,0.9); font-size:0.88rem;">
    ⚠️ &nbsp;<strong>Preliminary Assessment Only</strong> &nbsp;—&nbsp; {DISCLAIMER[3:]}
  </span>
</div>
""", unsafe_allow_html=True)

st.divider()

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("""
    <div class="glass-card" style="margin-bottom:0.5rem;">
      <span style="font-size:1.1rem; font-weight:600; color:#86efac;">
        📤 &nbsp;Upload Crop Image
      </span><br/>
      <span style="font-size:0.85rem; color:rgba(134,239,172,0.65);">
        JPG &bull; JPEG &bull; PNG &bull; Max 10 MB
      </span>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a photo of the affected plant or leaf",
        type=["jpg", "jpeg", "png"],
        help="For best results upload a clear, well-lit, close-up photo of the affected area.",
        label_visibility="collapsed",
    )
    diagnose_btn = st.button("🔍 Diagnose Crop", type="primary", use_container_width=True)

with col_preview:
    if uploaded_file:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:0.5rem; padding:0.7rem 1rem;">
          <span style="font-size:1rem; font-weight:600; color:#86efac;">🖼️ Uploaded Image</span>
        </div>
        """, unsafe_allow_html=True)
        st.image(uploaded_file, width="stretch", caption=uploaded_file.name)
    else:
        st.markdown("""
        <div class="glass-card" style="min-height:200px; display:flex; align-items:center;
             justify-content:center; text-align:center; padding:2rem;">
          <div>
            <div style="font-size:3rem; margin-bottom:0.5rem;">🌱</div>
            <div style="color:rgba(134,239,172,0.55); font-size:0.92rem;">
              Upload a crop image to see the preview here
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


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
