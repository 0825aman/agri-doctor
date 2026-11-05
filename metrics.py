# -*- coding: utf-8 -*-
"""
metrics.py — Agri Doctor LLM & Agent Evaluation Engine
=======================================================
Collects, computes, and stores all evaluation metrics for both agents.

Categories
----------
1. Accuracy & Quality   (ROUGE, BLEU, semantic similarity, F1)
2. RAG Evaluation       (faithfulness, answer relevancy, groundedness)
3. Agentic AI Metrics   (task completion, tool accuracy, handoff, loop detection)
4. Safety & Guardrails  (hallucination rate, toxicity, bias, PII leakage)
5. Performance & Cost   (latency, tokens, cost, error rate)
6. Multi-Agent System   (end-to-end, coordination, consistency)
"""

from __future__ import annotations

import re
import time
import math
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Session store  (one RunMetrics object per diagnosis run)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    """Per-agent performance snapshot."""
    agent_name:           str   = ""
    # Timing
    start_ts:             float = 0.0
    end_ts:               float = 0.0
    latency_s:            float = 0.0
    # Tokens
    input_tokens:         int   = 0
    output_tokens:        int   = 0
    total_tokens:         int   = 0
    # Quality (computed post-hoc)
    response_length:      int   = 0
    avg_sentence_length:  float = 0.0
    flesch_score:         float = 0.0    # readability 0-100
    keyword_hit_rate:     float = 0.0    # domain keywords present
    disclaimer_present:   bool  = False
    # Agentic
    task_completed:       bool  = False
    tool_call_valid:      bool  = True
    handoff_valid:        bool  = True
    # Safety
    hallucination_flag:   bool  = False
    toxicity_flag:        bool  = False
    bias_flag:            bool  = False
    pii_flag:             bool  = False
    guardrail_triggers:   int   = 0
    # Cost estimate (IBM watsonx pricing tiers — approximate)
    estimated_cost_usd:   float = 0.0
    # Error
    errored:              bool  = False
    error_message:        str   = ""


@dataclass
class RunMetrics:
    """Full metrics for one diagnosis run (both agents)."""
    run_id:                   str             = ""
    timestamp:                str             = ""
    image_filename:           str             = ""
    vision_model:             str             = ""
    text_model:               str             = ""
    # Per-agent
    agent1:                   AgentMetrics    = field(default_factory=AgentMetrics)
    agent2:                   AgentMetrics    = field(default_factory=AgentMetrics)
    # Multi-agent / system
    total_latency_s:          float           = 0.0
    total_tokens:             int             = 0
    total_cost_usd:           float           = 0.0
    workflow_completed:       bool            = False
    handoff_success:          bool            = False
    agent_coordination_score: float           = 0.0   # 0-1
    inter_agent_consistency:  float           = 0.0   # overlap of key terms 0-1
    # Aggregate safety
    overall_hallucination_rate: float         = 0.0
    overall_guardrail_rate:     float         = 0.0
    # Guardrail details (passed in from guardrails.py results)
    guardrail_summary:        dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# In-memory history (persisted to Streamlit session_state by dashboard.py)
_run_history: list[RunMetrics] = []


def record_run(run: RunMetrics) -> None:
    _run_history.append(run)


def get_run_history() -> list[RunMetrics]:
    return list(_run_history)


def clear_history() -> None:
    _run_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. ACCURACY & QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Agricultural domain keywords expected in a good diagnosis
_DOMAIN_KEYWORDS_DIAGNOSIS = [
    "disease", "symptom", "infection", "fungal", "bacterial", "viral",
    "lesion", "blight", "mildew", "rust", "wilt", "necrosis", "chlorosis",
    "pathogen", "leaf", "stem", "root", "crop", "plant", "diagnosis",
    "confidence", "assessment", "preliminary",
]
_DOMAIN_KEYWORDS_ADVISORY = [
    "treatment", "action", "recommendation", "prevention", "apply",
    "fungicide", "organic", "spray", "remove", "isolate", "monitor",
    "farmer", "crop", "safety", "consult", "extension", "pesticide",
]


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L (longest common subsequence) F1 score."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    m, n = len(r_tokens), len(h_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r_tokens[i - 1] == h_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall    = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu_1(hypothesis: str, reference: str) -> float:
    """Compute unigram BLEU score (BLEU-1)."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens:
        return 0.0
    r_counts = Counter(r_tokens)
    clip_count = sum(min(cnt, r_counts[tok]) for tok, cnt in Counter(h_tokens).items())
    precision = clip_count / len(h_tokens)
    # Brevity penalty
    bp = 1.0 if len(h_tokens) >= len(r_tokens) else math.exp(1 - len(r_tokens) / len(h_tokens))
    return bp * precision


def compute_f1_token(hypothesis: str, reference: str) -> float:
    """Token-level F1 between hypothesis and reference."""
    h_set = set(hypothesis.lower().split())
    r_set = set(reference.lower().split())
    if not h_set or not r_set:
        return 0.0
    common = h_set & r_set
    precision = len(common) / len(h_set)
    recall    = len(common) / len(r_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Lightweight Jaccard-based semantic similarity (no external embedding needed).
    Returns 0.0–1.0.
    """
    a_tokens = set(re.findall(r"\b\w+\b", text_a.lower()))
    b_tokens = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def compute_keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of domain keywords present in text."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


def compute_flesch_readability(text: str) -> float:
    """
    Flesch Reading Ease approximation (0=hard, 100=easy).
    Uses sentence and syllable counts estimated from text.
    """
    sentences = max(len(re.findall(r"[.!?]", text)), 1)
    words     = text.split()
    if not words:
        return 0.0
    # Rough syllable count: vowel groups per word
    syllables = sum(max(len(re.findall(r"[aeiouAEIOU]+", w)), 1) for w in words)
    asl = len(words) / sentences          # avg sentence length
    asw = syllables / len(words)          # avg syllables per word
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0.0, min(100.0, score)), 1)


def compute_avg_sentence_length(text: str) -> float:
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 1)


def compute_quality_metrics(text: str, agent_name: str) -> dict:
    """Compute all quality metrics for an agent response."""
    keywords = (_DOMAIN_KEYWORDS_DIAGNOSIS if "pathologist" in agent_name.lower()
                else _DOMAIN_KEYWORDS_ADVISORY)
    disclaimer_markers = ["preliminary", "ai-generated", "not a confirmed",
                          "consult", "extension officer", "laboratory"]
    return {
        "response_length":      len(text),
        "avg_sentence_length":  compute_avg_sentence_length(text),
        "flesch_readability":   compute_flesch_readability(text),
        "keyword_hit_rate":     compute_keyword_hit_rate(text, keywords),
        "disclaimer_present":   any(m in text.lower() for m in disclaimer_markers),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_faithfulness(response: str, context: str) -> float:
    """Estimate faithfulness: fraction of response sentences traceable to context."""
    sentences = [s.strip() for s in re.split(r"[.!?]", response) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    faithful = 0
    for s in sentences:
        s_tokens = set(re.findall(r"\b\w+\b", s.lower()))
        if len(s_tokens & context_tokens) / max(len(s_tokens), 1) > 0.3:
            faithful += 1
    return round(faithful / len(sentences), 3)


def compute_answer_relevancy(response: str, query_keywords: list[str]) -> float:
    """Fraction of query keywords present in the response."""
    return compute_keyword_hit_rate(response, query_keywords)


def compute_groundedness(response: str, source: str) -> float:
    """
    Groundedness: how much of the response content can be traced back to the source.
    Uses token overlap between response and source context.
    """
    return compute_semantic_similarity(response, source)


def compute_rag_metrics(response: str, source_context: str, query_keywords: list[str]) -> dict:
    return {
        "faithfulness":      compute_faithfulness(response, source_context),
        "answer_relevancy":  compute_answer_relevancy(response, query_keywords),
        "groundedness":      compute_groundedness(response, source_context),
        "context_precision": compute_semantic_similarity(response, source_context),
        "context_recall":    compute_keyword_hit_rate(source_context, query_keywords),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENTIC AI METRICS
# ─────────────────────────────────────────────────────────────────────────────

_TASK_COMPLETION_SECTIONS_AGENT1 = [
    "image quality", "suspected disease", "observed symptoms",
    "possible causes", "confidence", "alternative",
]
_TASK_COMPLETION_SECTIONS_AGENT2 = [
    "immediate action", "treatment", "prevention",
    "crop care", "safety", "summary",
]


def compute_task_completion_rate(response: str, agent_name: str) -> float:
    """
    Measures what fraction of expected output sections the agent produced.
    """
    sections = (_TASK_COMPLETION_SECTIONS_AGENT1
                if "pathologist" in agent_name.lower()
                else _TASK_COMPLETION_SECTIONS_AGENT2)
    text_lower = response.lower()
    found = sum(1 for s in sections if s in text_lower)
    return round(found / len(sections), 3)


def compute_reasoning_quality(response: str) -> float:
    """
    Proxy for reasoning quality: presence of causal/evidential language.
    """
    reasoning_markers = [
        "because", "therefore", "due to", "as a result", "indicates",
        "suggests", "evidence", "caused by", "consistent with", "likely",
        "possible", "probable", "observed", "characteristic",
    ]
    text_lower = response.lower()
    hits = sum(1 for m in reasoning_markers if m in text_lower)
    return round(min(hits / 8.0, 1.0), 3)   # normalise to 0–1


def compute_planning_accuracy(response: str) -> float:
    """
    Measures whether the response follows a structured plan (numbered/bulleted steps).
    """
    numbered_steps = len(re.findall(r"^\s*\d+[\.\)]\s", response, re.MULTILINE))
    bullet_steps   = len(re.findall(r"^\s*[-•*]\s", response, re.MULTILINE))
    total_steps    = numbered_steps + bullet_steps
    # 5+ structured steps = full score
    return round(min(total_steps / 5.0, 1.0), 3)


def compute_handoff_accuracy(payload: str) -> float:
    """
    Evaluate the quality of the handoff payload from Agent 1 → Agent 2.
    Returns 0–1 based on completeness.
    """
    required_elements = [
        "disease", "symptom", "confidence", "cause", "assessment"
    ]
    text_lower = payload.lower()
    found = sum(1 for e in required_elements if e in text_lower)
    return round(found / len(required_elements), 3)


def compute_agentic_metrics(response: str, agent_name: str,
                             handoff_payload: str = "") -> dict:
    return {
        "task_completion_rate": compute_task_completion_rate(response, agent_name),
        "reasoning_quality":    compute_reasoning_quality(response),
        "planning_accuracy":    compute_planning_accuracy(response),
        "handoff_accuracy":     compute_handoff_accuracy(handoff_payload) if handoff_payload else None,
        "loop_detected":        False,   # set by guardrails.py LoopPrevention
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAFETY & GUARDRAIL METRICS
# ─────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_MARKERS = [
    r"as an ai", r"i cannot see", r"i don.t have access",
    r"no image (was )?provided", r"i cannot (view|analyze|process) image",
]
_TOXICITY_PATTERNS = [
    r"\b(kill|destroy)\s+(all|the)\s+(humans?|people)\b",
    r"\b(stupid|dumb|idiot)\b",
]
_BIAS_PATTERNS = [
    r"\b(always|never)\s+(use|apply|recommend)\b",
    r"\b(all|every)\s+(farmers?)\s+(should|must)\b",
]
_PII_CHECK = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"


def compute_safety_metrics(response: str, guardrail_results: list) -> dict:
    text_lower = response.lower()
    hallucination = any(re.search(p, text_lower) for p in _HALLUCINATION_MARKERS)
    toxicity      = any(re.search(p, text_lower) for p in _TOXICITY_PATTERNS)
    bias          = any(re.search(p, text_lower) for p in _BIAS_PATTERNS)
    pii_leak      = bool(re.search(_PII_CHECK, response))

    total_gr = len(guardrail_results)
    triggered = sum(1 for r in guardrail_results if not r.passed)
    gr_rate   = round(triggered / total_gr, 3) if total_gr else 0.0

    return {
        "hallucination_detected":   hallucination,
        "toxicity_detected":        toxicity,
        "bias_detected":            bias,
        "pii_leak_detected":        pii_leak,
        "guardrail_trigger_count":  triggered,
        "guardrail_total_checks":   total_gr,
        "guardrail_intervention_rate": gr_rate,
        "policy_violation_detected": hallucination or toxicity or pii_leak,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERFORMANCE & COST METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Approximate IBM watsonx pricing (USD per 1K tokens) — Class 9 / Class 18
_COST_PER_1K_INPUT  = 0.0001    # Class 9 input tier estimate
_COST_PER_1K_OUTPUT = 0.0003    # Class 9 output tier estimate


def estimate_token_count(text: str) -> int:
    """
    Rough token count: ~0.75 words per token (GPT-like tokenisation heuristic).
    """
    words = len(text.split())
    return max(1, int(words / 0.75))


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single API call."""
    input_cost  = (input_tokens  / 1000) * _COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * _COST_PER_1K_OUTPUT
    return round(input_cost + output_cost, 6)


def compute_performance_metrics(
    latency_s: float,
    input_text: str,
    output_text: str,
    errored: bool = False,
) -> dict:
    input_tokens  = estimate_token_count(input_text)
    output_tokens = estimate_token_count(output_text)
    total_tokens  = input_tokens + output_tokens
    tokens_per_s  = round(output_tokens / latency_s, 1) if latency_s > 0 else 0.0
    cost          = compute_cost(input_tokens, output_tokens)
    return {
        "latency_s":          round(latency_s, 3),
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "total_tokens":       total_tokens,
        "tokens_per_second":  tokens_per_s,
        "estimated_cost_usd": cost,
        "error_occurred":     errored,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-AGENT SYSTEM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_inter_agent_consistency(agent1_output: str, agent2_output: str) -> float:
    """
    Measures how consistently Agent 2 builds on Agent 1's key terms.
    High score = Agent 2 references the same disease/symptoms Agent 1 identified.
    """
    return compute_semantic_similarity(agent1_output, agent2_output)


def compute_agent_coordination_score(
    handoff_valid: bool,
    task1_complete: float,
    task2_complete: float,
    consistency: float,
) -> float:
    """
    Composite agent coordination score (0–1).
    Weights: handoff(30%) + task1(20%) + task2(20%) + consistency(30%)
    """
    h_score = 1.0 if handoff_valid else 0.0
    return round(0.30 * h_score + 0.20 * task1_complete +
                 0.20 * task2_complete + 0.30 * consistency, 3)


def compute_system_metrics(run: RunMetrics) -> dict:
    """Compute system-level multi-agent metrics from a completed RunMetrics."""
    a1 = run.agent1
    a2 = run.agent2
    consistency = compute_inter_agent_consistency(
        a2.response_length * " ",   # placeholder when full text not stored
        a2.response_length * " ",
    )
    coordination = compute_agent_coordination_score(
        run.handoff_success,
        a1.keyword_hit_rate,
        a2.keyword_hit_rate,
        run.inter_agent_consistency,
    )
    return {
        "total_latency_s":          round(a1.latency_s + a2.latency_s, 3),
        "total_tokens":             a1.total_tokens + a2.total_tokens,
        "total_cost_usd":           round(a1.estimated_cost_usd + a2.estimated_cost_usd, 6),
        "workflow_completed":       run.workflow_completed,
        "handoff_success":          run.handoff_success,
        "agent_coordination_score": coordination,
        "inter_agent_consistency":  run.inter_agent_consistency,
        "error_rate":               sum([a1.errored, a2.errored]) / 2,
        "task_distribution": {
            "agent1_share": round(a1.latency_s / max(a1.latency_s + a2.latency_s, 0.001), 2),
            "agent2_share": round(a2.latency_s / max(a1.latency_s + a2.latency_s, 0.001), 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: build_run_metrics — called by app.py after both agents complete
# ─────────────────────────────────────────────────────────────────────────────

def build_run_metrics(
    vision_model: str,
    text_model:   str,
    image_filename: str,
    # Agent 1
    a1_prompt:    str,
    a1_response:  str,
    a1_latency:   float,
    a1_guardrail_results: list,
    a1_errored:   bool = False,
    # Agent 2
    a2_prompt:    str  = "",
    a2_response:  str  = "",
    a2_latency:   float = 0.0,
    a2_guardrail_results: list = None,
    a2_errored:   bool = False,
    # System
    handoff_valid: bool = True,
    workflow_completed: bool = True,
) -> RunMetrics:
    """Build a complete RunMetrics from raw agent data."""
    if a2_guardrail_results is None:
        a2_guardrail_results = []

    run = RunMetrics(
        run_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20],
        timestamp = datetime.now(timezone.utc).isoformat(),
        image_filename = image_filename,
        vision_model   = vision_model,
        text_model     = text_model,
    )

    # ── Agent 1 metrics ───────────────────────────────────────────────────
    a1_perf  = compute_performance_metrics(a1_latency, a1_prompt, a1_response, a1_errored)
    a1_qual  = compute_quality_metrics(a1_response, "PathologistAgent")
    a1_agent = compute_agentic_metrics(a1_response, "PathologistAgent", a1_response)
    a1_safe  = compute_safety_metrics(a1_response, a1_guardrail_results)

    run.agent1 = AgentMetrics(
        agent_name          = "PathologistAgent",
        latency_s           = a1_perf["latency_s"],
        input_tokens        = a1_perf["input_tokens"],
        output_tokens       = a1_perf["output_tokens"],
        total_tokens        = a1_perf["total_tokens"],
        response_length     = a1_qual["response_length"],
        avg_sentence_length = a1_qual["avg_sentence_length"],
        flesch_score        = a1_qual["flesch_readability"],
        keyword_hit_rate    = a1_qual["keyword_hit_rate"],
        disclaimer_present  = a1_qual["disclaimer_present"],
        task_completed      = a1_agent["task_completion_rate"] >= 0.5,
        guardrail_triggers  = a1_safe["guardrail_trigger_count"],
        hallucination_flag  = a1_safe["hallucination_detected"],
        toxicity_flag       = a1_safe["toxicity_detected"],
        bias_flag           = a1_safe["bias_detected"],
        pii_flag            = a1_safe["pii_leak_detected"],
        estimated_cost_usd  = a1_perf["estimated_cost_usd"],
        errored             = a1_errored,
    )

    # ── Agent 2 metrics ───────────────────────────────────────────────────
    if a2_response:
        a2_perf  = compute_performance_metrics(a2_latency, a2_prompt, a2_response, a2_errored)
        a2_qual  = compute_quality_metrics(a2_response, "ExtensionOfficerAgent")
        a2_agent = compute_agentic_metrics(a2_response, "ExtensionOfficerAgent")
        a2_safe  = compute_safety_metrics(a2_response, a2_guardrail_results)

        run.agent2 = AgentMetrics(
            agent_name          = "ExtensionOfficerAgent",
            latency_s           = a2_perf["latency_s"],
            input_tokens        = a2_perf["input_tokens"],
            output_tokens       = a2_perf["output_tokens"],
            total_tokens        = a2_perf["total_tokens"],
            response_length     = a2_qual["response_length"],
            avg_sentence_length = a2_qual["avg_sentence_length"],
            flesch_score        = a2_qual["flesch_readability"],
            keyword_hit_rate    = a2_qual["keyword_hit_rate"],
            disclaimer_present  = a2_qual["disclaimer_present"],
            task_completed      = a2_agent["task_completion_rate"] >= 0.5,
            guardrail_triggers  = a2_safe["guardrail_trigger_count"],
            hallucination_flag  = a2_safe["hallucination_detected"],
            toxicity_flag       = a2_safe["toxicity_detected"],
            bias_flag           = a2_safe["bias_detected"],
            pii_flag            = a2_safe["pii_leak_detected"],
            estimated_cost_usd  = a2_perf["estimated_cost_usd"],
            errored             = a2_errored,
        )

    # ── System metrics ────────────────────────────────────────────────────
    run.handoff_success     = handoff_valid
    run.workflow_completed  = workflow_completed
    run.total_latency_s     = round(run.agent1.latency_s + run.agent2.latency_s, 3)
    run.total_tokens        = run.agent1.total_tokens + run.agent2.total_tokens
    run.total_cost_usd      = round(run.agent1.estimated_cost_usd + run.agent2.estimated_cost_usd, 6)
    run.inter_agent_consistency = compute_inter_agent_consistency(a1_response, a2_response)
    run.agent_coordination_score = compute_agent_coordination_score(
        handoff_valid,
        a1_agent["task_completion_rate"],
        a2_agent["task_completion_rate"] if a2_response else 0.0,
        run.inter_agent_consistency,
    )
    total_gr = (len(a1_guardrail_results) + len(a2_guardrail_results))
    triggered = ((run.agent1.guardrail_triggers or 0) + (run.agent2.guardrail_triggers or 0))
    run.overall_guardrail_rate    = round(triggered / total_gr, 3) if total_gr else 0.0
    run.overall_hallucination_rate = round(
        sum([run.agent1.hallucination_flag, run.agent2.hallucination_flag]) / 2, 3
    )

    record_run(run)
    return run
