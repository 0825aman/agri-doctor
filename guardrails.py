# -*- coding: utf-8 -*-
"""
guardrails.py — Agri Doctor Guardrail Engine
=============================================
Implements all 9 guardrail categories for the two-agent AI workflow.

Categories
----------
1. Input Guardrails
2. Output Guardrails
3. Tool Execution Guardrails
4. Data Guardrails
5. Security Guardrails
6. Operational Guardrails
7. Human-in-the-Loop Guardrails
8. Multi-Agent Guardrails
9. Ethical & Compliance Guardrails
"""

from __future__ import annotations

import re
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Logging / Audit trail ─────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  [GUARDRAIL] %(levelname)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agri_doctor.guardrails")

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    category: str
    check: str
    message: str = ""
    severity: str = "info"        # info | warning | error | critical
    remediation: str = ""
    details: dict = field(default_factory=dict)

    def __bool__(self):
        return self.passed


# ─────────────────────────────────────────────────────────────────────────────
# Audit trail (in-memory; persisted to session_state by app.py)
# ─────────────────────────────────────────────────────────────────────────────

_audit_log: list[dict] = []

def _audit(category: str, check: str, result: GuardrailResult, context: str = "") -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "check": check,
        "passed": result.passed,
        "severity": result.severity,
        "message": result.message,
        "context": context,
    }
    _audit_log.append(entry)
    level = logging.WARNING if not result.passed else logging.INFO
    logger.log(level, "[%s] %s — %s", category, check, result.message or "OK")

def get_audit_log() -> list[dict]:
    return list(_audit_log)

def clear_audit_log() -> None:
    _audit_log.clear()


# =============================================================================
# 1. INPUT GUARDRAILS
# =============================================================================

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?(above|prior|previous|system)",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?:a|an)\s+",
    r"forget\s+(everything|all)\s+(you('ve)?\s+been\s+)?told",
    r"new\s+instructions?:",
    r"system\s*:\s*you\s+must",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"###\s*instruction",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"jailbreak",
    r"override\s+(safety|guardrail|instruction)",
]

# PII patterns (basic — no external library dependency)
_PII_PATTERNS = {
    "email":       r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone_intl":  r"\+?[\d\s\-\(\)]{10,15}",
    "aadhar":      r"\b\d{4}\s?\d{4}\s?\d{4}\b",   # India Aadhaar
    "pan":         r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",   # India PAN
    "credit_card": r"\b(?:\d[ \-]?){13,16}\b",
    "ssn":         r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
}

# Inappropriate / off-topic content keywords for an agriculture app
_INAPPROPRIATE_KEYWORDS = [
    "bomb", "weapon", "kill", "terrorist", "hack", "exploit",
    "nude", "porn", "sex", "violence", "drug", "narco",
]

# Allowed image MIME types
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
_MAX_IMAGE_SIZE_MB   = 10


def check_prompt_injection(text: str) -> GuardrailResult:
    """Detect prompt injection attempts in user-supplied text."""
    text_lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            result = GuardrailResult(
                passed=False, category="Input", check="PromptInjection",
                message=f"Possible prompt injection detected (pattern: '{pattern}').",
                severity="critical",
                remediation="Input blocked. Please enter a legitimate crop-disease query.",
            )
            _audit("Input", "PromptInjection", result)
            return result
    result = GuardrailResult(passed=True, category="Input", check="PromptInjection",
                             message="No injection patterns detected.")
    _audit("Input", "PromptInjection", result)
    return result


def check_pii(text: str) -> GuardrailResult:
    """Detect PII in user-supplied text and mask it."""
    found = {}
    for label, pattern in _PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[label] = len(matches)
    if found:
        result = GuardrailResult(
            passed=False, category="Input", check="PIIDetection",
            message=f"PII detected: {list(found.keys())}.",
            severity="warning",
            remediation="Personal information has been redacted before processing.",
            details={"types_found": found},
        )
        _audit("Input", "PIIDetection", result)
        return result
    result = GuardrailResult(passed=True, category="Input", check="PIIDetection",
                             message="No PII detected.")
    _audit("Input", "PIIDetection", result)
    return result


def mask_pii(text: str) -> str:
    """Replace PII patterns with [REDACTED-<TYPE>] tokens."""
    for label, pattern in _PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED-{label.upper()}]", text)
    return text


def check_inappropriate_content(text: str) -> GuardrailResult:
    """Filter clearly off-topic or inappropriate content."""
    text_lower = text.lower()
    hits = [kw for kw in _INAPPROPRIATE_KEYWORDS if kw in text_lower]
    if hits:
        result = GuardrailResult(
            passed=False, category="Input", check="InappropriateContent",
            message=f"Inappropriate keyword(s) detected: {hits}.",
            severity="error",
            remediation="This app is for agricultural crop-disease queries only.",
        )
        _audit("Input", "InappropriateContent", result)
        return result
    result = GuardrailResult(passed=True, category="Input", check="InappropriateContent",
                             message="Content is appropriate.")
    _audit("Input", "InappropriateContent", result)
    return result


def check_image_input(filename: str, mime_type: str, size_bytes: int) -> GuardrailResult:
    """Validate uploaded image format and size."""
    if mime_type not in _ALLOWED_IMAGE_TYPES:
        result = GuardrailResult(
            passed=False, category="Input", check="ImageValidation",
            message=f"Unsupported image type '{mime_type}'. Allowed: JPEG, PNG.",
            severity="error",
            remediation="Upload a JPG, JPEG, or PNG image.",
        )
        _audit("Input", "ImageValidation", result)
        return result
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > _MAX_IMAGE_SIZE_MB:
        result = GuardrailResult(
            passed=False, category="Input", check="ImageValidation",
            message=f"Image too large ({size_mb:.1f} MB). Max allowed: {_MAX_IMAGE_SIZE_MB} MB.",
            severity="error",
            remediation=f"Resize the image to under {_MAX_IMAGE_SIZE_MB} MB.",
        )
        _audit("Input", "ImageValidation", result)
        return result
    result = GuardrailResult(passed=True, category="Input", check="ImageValidation",
                             message=f"Image '{filename}' is valid ({size_mb:.2f} MB).")
    _audit("Input", "ImageValidation", result)
    return result


def run_input_guardrails(
    text: str = "",
    filename: str = "",
    mime_type: str = "",
    size_bytes: int = 0,
) -> tuple[bool, list[GuardrailResult]]:
    """Run all input guardrails. Returns (all_passed, results_list)."""
    results = []
    if text:
        results.append(check_prompt_injection(text))
        results.append(check_pii(text))
        results.append(check_inappropriate_content(text))
    if filename and mime_type:
        results.append(check_image_input(filename, mime_type, size_bytes))
    all_passed = all(r.passed for r in results)
    return all_passed, results


# =============================================================================
# 2. OUTPUT GUARDRAILS
# =============================================================================

_TOXIC_PATTERNS = [
    r"\b(kill|destroy|eliminate)\s+(all|the)\s+(humans?|people|farmers?)\b",
    r"\byou\s+(are|were)\s+(stupid|dumb|idiot|useless)\b",
    r"\b(f[u\*]+ck|sh[i\*]+t|b[i\*]+tch|a[s\*]+hole)\b",
]

_HALLUCINATION_MARKERS = [
    r"as\s+an?\s+AI\s+(language\s+)?model,?\s+I\s+(cannot|can't|don't)\s+see",
    r"I\s+(don't|cannot|can't)\s+actually\s+(see|view|analyze)\s+image",
    r"I\s+don't\s+have\s+(the\s+)?(ability|capability)\s+to\s+(process|analyze|view)",
    r"no\s+image\s+(was\s+)?(provided|attached|uploaded|given)",
]

_SENSITIVE_OUTPUT_PATTERNS = {
    "api_key":    r"[a-zA-Z0-9_\-]{32,}",
    "jwt_token":  r"eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+",
    "ibm_key":    r"[a-zA-Z0-9_\-]{44}==",
}

_MIN_RESPONSE_LENGTH = 30
_MAX_RESPONSE_LENGTH = 15000


def check_toxicity(text: str) -> GuardrailResult:
    """Screen output for toxic or harmful language."""
    text_lower = text.lower()
    for pattern in _TOXIC_PATTERNS:
        if re.search(pattern, text_lower):
            result = GuardrailResult(
                passed=False, category="Output", check="ToxicityFilter",
                message="Toxic content detected in AI response.",
                severity="critical",
                remediation="Response blocked. Please try again.",
            )
            _audit("Output", "ToxicityFilter", result)
            return result
    result = GuardrailResult(passed=True, category="Output", check="ToxicityFilter",
                             message="No toxic content detected.")
    _audit("Output", "ToxicityFilter", result)
    return result


def check_hallucination_markers(text: str) -> GuardrailResult:
    """Detect when the model admits it cannot see the image (hallucination risk)."""
    for pattern in _HALLUCINATION_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            result = GuardrailResult(
                passed=False, category="Output", check="HallucinationDetection",
                message="Model indicates it cannot process the image.",
                severity="warning",
                remediation="The vision model may not support image input. "
                            "Switch to a confirmed image_chat model in sidebar settings.",
                details={"matched_pattern": pattern},
            )
            _audit("Output", "HallucinationDetection", result)
            return result
    result = GuardrailResult(passed=True, category="Output", check="HallucinationDetection",
                             message="No hallucination markers detected.")
    _audit("Output", "HallucinationDetection", result)
    return result


def check_sensitive_output(text: str) -> GuardrailResult:
    """Ensure AI response does not accidentally leak credentials or keys."""
    for label, pattern in _SENSITIVE_OUTPUT_PATTERNS.items():
        if re.search(pattern, text):
            result = GuardrailResult(
                passed=False, category="Output", check="SensitiveInfoFilter",
                message=f"Possible sensitive data ({label}) detected in response.",
                severity="critical",
                remediation="Sensitive data redacted from output.",
            )
            _audit("Output", "SensitiveInfoFilter", result)
            return result
    result = GuardrailResult(passed=True, category="Output", check="SensitiveInfoFilter",
                             message="No sensitive information in response.")
    _audit("Output", "SensitiveInfoFilter", result)
    return result


def check_response_length(text: str) -> GuardrailResult:
    """Validate response is neither empty nor excessively long."""
    length = len(text.strip())
    if length < _MIN_RESPONSE_LENGTH:
        result = GuardrailResult(
            passed=False, category="Output", check="ResponseFormat",
            message=f"Response too short ({length} chars). Likely empty or incomplete.",
            severity="warning",
            remediation="Model returned an unusually short response. Please retry.",
        )
        _audit("Output", "ResponseFormat", result)
        return result
    if length > _MAX_RESPONSE_LENGTH:
        result = GuardrailResult(
            passed=False, category="Output", check="ResponseFormat",
            message=f"Response too long ({length} chars). Truncation may be needed.",
            severity="warning",
            remediation="Response has been flagged as unusually long.",
        )
        _audit("Output", "ResponseFormat", result)
        return result
    result = GuardrailResult(passed=True, category="Output", check="ResponseFormat",
                             message=f"Response length OK ({length} chars).")
    _audit("Output", "ResponseFormat", result)
    return result


def check_disclaimer_present(text: str, agent_name: str) -> GuardrailResult:
    """Ensure agent output contains required AI disclaimer."""
    disclaimer_markers = [
        "preliminary", "ai-generated", "not a confirmed", "consult",
        "extension officer", "plant pathologist", "laboratory",
    ]
    text_lower = text.lower()
    found = [m for m in disclaimer_markers if m in text_lower]
    if len(found) < 2:
        result = GuardrailResult(
            passed=False, category="Output", check="DisclaimerPresent",
            message=f"{agent_name} output missing required safety disclaimer.",
            severity="warning",
            remediation="Disclaimer appended automatically.",
        )
        _audit("Output", "DisclaimerPresent", result)
        return result
    result = GuardrailResult(passed=True, category="Output", check="DisclaimerPresent",
                             message=f"{agent_name} disclaimer present.")
    _audit("Output", "DisclaimerPresent", result)
    return result


def run_output_guardrails(text: str, agent_name: str = "Agent") -> tuple[bool, list[GuardrailResult]]:
    """Run all output guardrails. Returns (all_passed, results_list)."""
    results = [
        check_toxicity(text),
        check_hallucination_markers(text),
        check_sensitive_output(text),
        check_response_length(text),
        check_disclaimer_present(text, agent_name),
    ]
    all_passed = all(r.passed for r in results)
    return all_passed, results


# =============================================================================
# 3. TOOL EXECUTION GUARDRAILS
# =============================================================================

# Whitelist of permitted model IDs (confirmed on us-south endpoint)
_PERMITTED_MODELS = {
    "ibm/granite-4-h-small",
    "meta-llama/llama-3-3-70b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "mistralai/mistral-medium-2505",
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "mistral-large-2512",
    "openai/gpt-oss-120b",
}

_PERMITTED_ENDPOINTS = {
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com",
}

_MAX_TOKENS_VISION = 1024
_MAX_TOKENS_TEXT   = 1500
_MAX_TEMPERATURE   = 0.7


def check_model_permission(model_id: str) -> GuardrailResult:
    """Ensure only whitelisted models are called."""
    if model_id not in _PERMITTED_MODELS:
        result = GuardrailResult(
            passed=False, category="ToolExecution", check="ModelPermission",
            message=f"Model '{model_id}' is not in the permitted list.",
            severity="error",
            remediation=f"Use one of: {sorted(_PERMITTED_MODELS)}",
        )
        _audit("ToolExecution", "ModelPermission", result)
        return result
    result = GuardrailResult(passed=True, category="ToolExecution", check="ModelPermission",
                             message=f"Model '{model_id}' is permitted.")
    _audit("ToolExecution", "ModelPermission", result)
    return result


def check_endpoint_authorization(url: str) -> GuardrailResult:
    """Ensure API calls only go to authorized IBM endpoints."""
    if url not in _PERMITTED_ENDPOINTS:
        result = GuardrailResult(
            passed=False, category="ToolExecution", check="EndpointAuthorization",
            message=f"Endpoint '{url}' is not authorized.",
            severity="critical",
            remediation=f"Use one of: {sorted(_PERMITTED_ENDPOINTS)}",
        )
        _audit("ToolExecution", "EndpointAuthorization", result)
        return result
    result = GuardrailResult(passed=True, category="ToolExecution", check="EndpointAuthorization",
                             message=f"Endpoint '{url}' is authorized.")
    _audit("ToolExecution", "EndpointAuthorization", result)
    return result


def check_generation_params(max_tokens: int, temperature: float, agent_name: str) -> GuardrailResult:
    """Validate token and temperature limits."""
    limit = _MAX_TOKENS_VISION if "pathologist" in agent_name.lower() else _MAX_TOKENS_TEXT
    if max_tokens > limit:
        result = GuardrailResult(
            passed=False, category="ToolExecution", check="ParamValidation",
            message=f"{agent_name}: max_tokens {max_tokens} exceeds limit {limit}.",
            severity="warning",
            remediation=f"max_tokens capped at {limit}.",
        )
        _audit("ToolExecution", "ParamValidation", result)
        return result
    if temperature > _MAX_TEMPERATURE:
        result = GuardrailResult(
            passed=False, category="ToolExecution", check="ParamValidation",
            message=f"{agent_name}: temperature {temperature} exceeds max {_MAX_TEMPERATURE}.",
            severity="warning",
            remediation=f"temperature capped at {_MAX_TEMPERATURE}.",
        )
        _audit("ToolExecution", "ParamValidation", result)
        return result
    result = GuardrailResult(passed=True, category="ToolExecution", check="ParamValidation",
                             message=f"{agent_name} params OK (tokens={max_tokens}, temp={temperature}).")
    _audit("ToolExecution", "ParamValidation", result)
    return result


def run_tool_guardrails(model_id: str, endpoint: str,
                        max_tokens: int, temperature: float,
                        agent_name: str) -> tuple[bool, list[GuardrailResult]]:
    results = [
        check_model_permission(model_id),
        check_endpoint_authorization(endpoint),
        check_generation_params(max_tokens, temperature, agent_name),
    ]
    return all(r.passed for r in results), results


# =============================================================================
# 4. DATA GUARDRAILS
# =============================================================================

def anonymize_filename(filename: str) -> str:
    """Replace original filename with a safe hash-based alias."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    h = hashlib.sha256(filename.encode()).hexdigest()[:8]
    return f"crop_image_{h}.{ext}"


def check_data_privacy(text: str) -> GuardrailResult:
    """Ensure no raw PII is passed to models."""
    pii_result = check_pii(text)
    if not pii_result.passed:
        result = GuardrailResult(
            passed=False, category="Data", check="DataPrivacy",
            message="PII found in data payload — masking applied.",
            severity="warning",
            remediation="Data has been anonymized before transmission.",
            details=pii_result.details,
        )
        _audit("Data", "DataPrivacy", result)
        return result
    result = GuardrailResult(passed=True, category="Data", check="DataPrivacy",
                             message="No PII in data payload.")
    _audit("Data", "DataPrivacy", result)
    return result


def check_image_not_stored(action: str) -> GuardrailResult:
    """Verify image bytes are never written to disk."""
    if action in ("write_disk", "store_db", "log_bytes"):
        result = GuardrailResult(
            passed=False, category="Data", check="DataStorage",
            message=f"Unauthorized data storage action: '{action}'.",
            severity="critical",
            remediation="Images are processed in-memory only and never persisted.",
        )
        _audit("Data", "DataStorage", result)
        return result
    result = GuardrailResult(passed=True, category="Data", check="DataStorage",
                             message="Data storage policy compliant.")
    _audit("Data", "DataStorage", result)
    return result


# =============================================================================
# 5. SECURITY GUARDRAILS
# =============================================================================

_SQL_INJECTION_PATTERNS = [
    r"(--|;|'|\")\s*(drop|delete|insert|update|select|union|exec|xp_)",
    r"\bor\b\s+\d+\s*=\s*\d+",
    r"\bsleep\s*\(\d+\)",
]

_JAILBREAK_PHRASES = [
    "do anything now", "no restrictions", "without limitations",
    "pretend you have no", "bypass your", "ignore your training",
    "you have no rules", "developer mode", "enable developer",
]


def check_sql_injection(text: str) -> GuardrailResult:
    """Prevent SQL injection in any text field."""
    text_lower = text.lower()
    for pattern in _SQL_INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            result = GuardrailResult(
                passed=False, category="Security", check="SQLInjection",
                message="SQL injection pattern detected.",
                severity="critical",
                remediation="Input rejected.",
            )
            _audit("Security", "SQLInjection", result)
            return result
    result = GuardrailResult(passed=True, category="Security", check="SQLInjection",
                             message="No SQL injection detected.")
    _audit("Security", "SQLInjection", result)
    return result


def check_jailbreak(text: str) -> GuardrailResult:
    """Detect jailbreak attempts."""
    text_lower = text.lower()
    hits = [p for p in _JAILBREAK_PHRASES if p in text_lower]
    if hits:
        result = GuardrailResult(
            passed=False, category="Security", check="JailbreakDetection",
            message=f"Jailbreak phrases detected: {hits}.",
            severity="critical",
            remediation="Input blocked.",
        )
        _audit("Security", "JailbreakDetection", result)
        return result
    result = GuardrailResult(passed=True, category="Security", check="JailbreakDetection",
                             message="No jailbreak detected.")
    _audit("Security", "JailbreakDetection", result)
    return result


def check_credential_exposure(text: str) -> GuardrailResult:
    """Ensure no API keys or secrets are present in any text going to the model."""
    # IBM API key pattern (~44 chars), generic long tokens
    patterns = [
        r"[A-Za-z0-9_\-]{40,}",   # long opaque tokens
        r"Bearer\s+[A-Za-z0-9_\-\.]+",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        # Ignore base64 image data (will be very long and contain +/=)
        real_matches = [m for m in matches if "+" not in m and "/" not in m and len(m) < 80]
        if real_matches:
            result = GuardrailResult(
                passed=False, category="Security", check="CredentialExposure",
                message="Possible credential/token in prompt text.",
                severity="critical",
                remediation="Remove credentials from text before submission.",
                details={"count": len(real_matches)},
            )
            _audit("Security", "CredentialExposure", result)
            return result
    result = GuardrailResult(passed=True, category="Security", check="CredentialExposure",
                             message="No credentials detected in text.")
    _audit("Security", "CredentialExposure", result)
    return result


def run_security_guardrails(text: str) -> tuple[bool, list[GuardrailResult]]:
    results = [
        check_prompt_injection(text),
        check_jailbreak(text),
        check_sql_injection(text),
    ]
    return all(r.passed for r in results), results


# =============================================================================
# 6. OPERATIONAL GUARDRAILS
# =============================================================================

_MAX_EXECUTION_SECONDS = 120
_MAX_RETRIES           = 3
_MAX_AGENT_ITERATIONS  = 2   # two agents in this pipeline
_RETRY_DELAY_SECONDS   = 2


class OperationalGuardrail:
    """Context manager that enforces execution time and retry limits."""

    def __init__(self, agent_name: str, max_seconds: int = _MAX_EXECUTION_SECONDS):
        self.agent_name  = agent_name
        self.max_seconds = max_seconds
        self.start_time: float | None = None
        self.attempts    = 0

    def start(self) -> None:
        self.start_time = time.time()
        self.attempts += 1
        result = GuardrailResult(
            passed=True, category="Operational", check="ExecutionStart",
            message=f"{self.agent_name} started (attempt {self.attempts}).",
        )
        _audit("Operational", "ExecutionStart", result, self.agent_name)

    def check_timeout(self) -> GuardrailResult:
        if self.start_time is None:
            return GuardrailResult(passed=True, category="Operational", check="Timeout",
                                   message="Timer not started.")
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            result = GuardrailResult(
                passed=False, category="Operational", check="Timeout",
                message=f"{self.agent_name} exceeded {self.max_seconds}s limit ({elapsed:.1f}s elapsed).",
                severity="error",
                remediation="Request timed out. Please retry.",
            )
            _audit("Operational", "Timeout", result, self.agent_name)
            return result
        result = GuardrailResult(passed=True, category="Operational", check="Timeout",
                                 message=f"{self.agent_name} within time limit ({elapsed:.1f}s).")
        _audit("Operational", "Timeout", result, self.agent_name)
        return result

    def check_retries(self) -> GuardrailResult:
        if self.attempts > _MAX_RETRIES:
            result = GuardrailResult(
                passed=False, category="Operational", check="RetryLimit",
                message=f"{self.agent_name} exceeded max retries ({_MAX_RETRIES}).",
                severity="error",
                remediation="Max retries reached. Please check credentials and model settings.",
            )
            _audit("Operational", "RetryLimit", result, self.agent_name)
            return result
        result = GuardrailResult(passed=True, category="Operational", check="RetryLimit",
                                 message=f"{self.agent_name} retry count OK ({self.attempts}/{_MAX_RETRIES}).")
        _audit("Operational", "RetryLimit", result, self.agent_name)
        return result

    def finish(self) -> float:
        elapsed = time.time() - (self.start_time or time.time())
        result = GuardrailResult(
            passed=True, category="Operational", check="ExecutionFinish",
            message=f"{self.agent_name} completed in {elapsed:.2f}s.",
        )
        _audit("Operational", "ExecutionFinish", result, self.agent_name)
        return elapsed


# =============================================================================
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# =============================================================================

# Trigger keywords that warrant human expert escalation
_ESCALATION_TRIGGERS = [
    "unknown disease", "cannot identify", "inconclusive",
    "severe", "widespread", "epidemic", "mass infection",
    "unclear image", "insufficient detail", "blurry",
    "toxic", "poisonous", "systemic",
]

_HIGH_CONFIDENCE_CHEMICAL_KEYWORDS = [
    "apply pesticide", "spray insecticide", "use fungicide",
    "chemical treatment", "herbicide application",
]


def check_escalation_needed(diagnosis_text: str) -> GuardrailResult:
    """Determine if human expert escalation is recommended."""
    text_lower = diagnosis_text.lower()
    triggers = [t for t in _ESCALATION_TRIGGERS if t in text_lower]
    if triggers:
        result = GuardrailResult(
            passed=False, category="HumanInLoop", check="EscalationCheck",
            message=f"Human expert review recommended. Triggers: {triggers}.",
            severity="warning",
            remediation="Please consult a certified plant pathologist or local agricultural extension officer.",
            details={"triggers": triggers},
        )
        _audit("HumanInLoop", "EscalationCheck", result)
        return result
    result = GuardrailResult(passed=True, category="HumanInLoop", check="EscalationCheck",
                             message="No escalation triggers detected.")
    _audit("HumanInLoop", "EscalationCheck", result)
    return result


def check_chemical_recommendation(advisory_text: str) -> GuardrailResult:
    """Flag chemical treatment recommendations for user awareness."""
    text_lower = advisory_text.lower()
    hits = [k for k in _HIGH_CONFIDENCE_CHEMICAL_KEYWORDS if k in text_lower]
    if hits:
        result = GuardrailResult(
            passed=False, category="HumanInLoop", check="ChemicalRecommendation",
            message="Advisory contains chemical treatment recommendations.",
            severity="warning",
            remediation=(
                "Always verify chemical dosages with your local agricultural extension officer "
                "or product label before application. Wear appropriate PPE."
            ),
            details={"keywords_found": hits},
        )
        _audit("HumanInLoop", "ChemicalRecommendation", result)
        return result
    result = GuardrailResult(passed=True, category="HumanInLoop", check="ChemicalRecommendation",
                             message="No high-risk chemical recommendation keywords found.")
    _audit("HumanInLoop", "ChemicalRecommendation", result)
    return result


# =============================================================================
# 8. MULTI-AGENT GUARDRAILS
# =============================================================================

_agent_registry: dict[str, dict] = {}
_MAX_AGENT_LOOP_ITERATIONS = 5


def register_agent(agent_name: str, permitted_inputs: list[str], permitted_outputs: list[str]) -> None:
    """Register an agent with its allowed I/O types."""
    _agent_registry[agent_name] = {
        "permitted_inputs": permitted_inputs,
        "permitted_outputs": permitted_outputs,
        "call_count": 0,
    }


def check_agent_handoff(from_agent: str, to_agent: str, payload: str) -> GuardrailResult:
    """Validate that inter-agent data transfer is safe and non-empty."""
    if not payload or len(payload.strip()) < 20:
        result = GuardrailResult(
            passed=False, category="MultiAgent", check="AgentHandoff",
            message=f"Handoff from '{from_agent}' to '{to_agent}' has empty/minimal payload.",
            severity="error",
            remediation=f"'{from_agent}' must produce a valid diagnosis before passing to '{to_agent}'.",
        )
        _audit("MultiAgent", "AgentHandoff", result)
        return result
    # Run security check on the handoff payload
    sec_ok, _ = run_security_guardrails(payload)
    if not sec_ok:
        result = GuardrailResult(
            passed=False, category="MultiAgent", check="AgentHandoff",
            message=f"Security violation in handoff payload from '{from_agent}'.",
            severity="critical",
            remediation="Payload blocked at agent boundary.",
        )
        _audit("MultiAgent", "AgentHandoff", result)
        return result
    result = GuardrailResult(
        passed=True, category="MultiAgent", check="AgentHandoff",
        message=f"Handoff from '{from_agent}' to '{to_agent}' is valid ({len(payload)} chars).",
    )
    _audit("MultiAgent", "AgentHandoff", result)
    return result


def check_loop_prevention(agent_name: str) -> GuardrailResult:
    """Prevent infinite agent execution loops."""
    if agent_name in _agent_registry:
        _agent_registry[agent_name]["call_count"] += 1
        count = _agent_registry[agent_name]["call_count"]
        if count > _MAX_AGENT_LOOP_ITERATIONS:
            result = GuardrailResult(
                passed=False, category="MultiAgent", check="LoopPrevention",
                message=f"Agent '{agent_name}' called {count} times — possible loop.",
                severity="critical",
                remediation="Agent execution halted to prevent infinite loop.",
            )
            _audit("MultiAgent", "LoopPrevention", result)
            return result
    result = GuardrailResult(passed=True, category="MultiAgent", check="LoopPrevention",
                             message=f"Agent '{agent_name}' call count within limits.")
    _audit("MultiAgent", "LoopPrevention", result)
    return result


def reset_agent_counters() -> None:
    """Reset call counters at the start of each new diagnosis run."""
    for name in _agent_registry:
        _agent_registry[name]["call_count"] = 0


# =============================================================================
# 9. ETHICAL & COMPLIANCE GUARDRAILS
# =============================================================================

_BIAS_INDICATORS = [
    r"\b(always|never)\s+(use|apply|recommend)\b",
    r"\b(all|every)\s+(farmers?|crops?)\s+(should|must)\b",
    r"\btraditional\s+farming\s+is\s+(better|worse|inferior|superior)\b",
]

_FABRICATED_DOSAGE_PATTERNS = [
    r"\d+\s*(ml|g|kg|L|litre|liter|gram|kilogram)\s*(per|/)\s*(acre|hectare|plant|ha)\b",
    r"apply\s+\d+",
    r"mix\s+\d+\s*(ml|g|L)\s+in\s+\d+\s*(L|litre|gallon)",
]


def check_bias(text: str) -> GuardrailResult:
    """Detect potentially biased language in AI output."""
    hits = []
    for pattern in _BIAS_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    if hits:
        result = GuardrailResult(
            passed=False, category="Ethics", check="BiasDetection",
            message=f"Potentially biased language detected ({len(hits)} pattern(s)).",
            severity="warning",
            remediation="Output may contain generalizations. Verify with local agronomist.",
            details={"patterns": hits},
        )
        _audit("Ethics", "BiasDetection", result)
        return result
    result = GuardrailResult(passed=True, category="Ethics", check="BiasDetection",
                             message="No bias indicators detected.")
    _audit("Ethics", "BiasDetection", result)
    return result


def check_fabricated_dosage(text: str) -> GuardrailResult:
    """Flag specific chemical dosages that may be fabricated by the model."""
    hits = []
    for pattern in _FABRICATED_DOSAGE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            hits.extend(matches)
    if hits:
        result = GuardrailResult(
            passed=False, category="Ethics", check="FabricatedDosage",
            message=f"Specific dosage figures detected ({len(hits)} instance(s)).",
            severity="warning",
            remediation=(
                "AI-generated dosages are not verified. Always check the product label "
                "and consult your local agricultural extension officer before applying any chemical."
            ),
            details={"dosage_instances": hits[:5]},
        )
        _audit("Ethics", "FabricatedDosage", result)
        return result
    result = GuardrailResult(passed=True, category="Ethics", check="FabricatedDosage",
                             message="No specific dosage figures detected.")
    _audit("Ethics", "FabricatedDosage", result)
    return result


def check_explainability(text: str, agent_name: str) -> GuardrailResult:
    """Ensure agent output includes reasoning/explanation, not just conclusions."""
    explanation_markers = ["because", "due to", "caused by", "as a result", "indicates",
                           "suggests", "observed", "symptom", "pattern", "evidence"]
    text_lower = text.lower()
    found = [m for m in explanation_markers if m in text_lower]
    if len(found) < 3:
        result = GuardrailResult(
            passed=False, category="Ethics", check="Explainability",
            message=f"{agent_name} output lacks sufficient explanatory reasoning.",
            severity="warning",
            remediation="Response should explain the reasoning behind its conclusions.",
        )
        _audit("Ethics", "Explainability", result)
        return result
    result = GuardrailResult(passed=True, category="Ethics", check="Explainability",
                             message=f"{agent_name} output contains adequate reasoning.")
    _audit("Ethics", "Explainability", result)
    return result


def run_ethics_guardrails(text: str, agent_name: str) -> tuple[bool, list[GuardrailResult]]:
    results = [
        check_bias(text),
        check_fabricated_dosage(text),
        check_explainability(text, agent_name),
    ]
    # Ethics warnings don't hard-block — they append warnings to the UI
    all_passed = all(r.passed for r in results)
    return all_passed, results


# =============================================================================
# CONVENIENCE: run_all_output_guardrails  (used by app.py)
# =============================================================================

def run_all_output_guardrails(text: str, agent_name: str) -> tuple[bool, list[GuardrailResult]]:
    """Run Output + Ethics + Human-in-the-Loop guardrails on agent output."""
    _, out_results   = run_output_guardrails(text, agent_name)
    _, eth_results   = run_ethics_guardrails(text, agent_name)
    hil_escalation   = check_escalation_needed(text)
    hil_chemical     = check_chemical_recommendation(text)

    all_results = out_results + eth_results + [hil_escalation, hil_chemical]
    # Hard-fail only on critical severity
    hard_fail = any(not r.passed and r.severity == "critical" for r in all_results)
    return not hard_fail, all_results
