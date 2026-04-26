"""Safety guardrails for triage inputs and outputs.

This module applies deterministic post-processing rules to LLM output before
anything reaches the user-facing layer.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List

from models.safety import InputSafetyFlags
from models.triage import CareRecommendation, SEVERITY_4, SEVERITY_5, TriageResponse

logger = logging.getLogger(__name__)

CANONICAL_DISCLAIMER = (
    "⚠️ This is AI-generated information for educational purposes only. It is not "
    "a substitute for professional medical advice, diagnosis, or treatment. If you "
    "believe you are experiencing a medical emergency, call 911 immediately. Always "
    "consult a qualified healthcare provider for medical concerns."
)
GENERIC_ESCALATION_WARNING = (
    "If symptoms worsen, persist beyond 72 hours, or you develop fever above "
    "103°F / 39.4°C, chest pain, difficulty breathing, or confusion, seek medical "
    "attention immediately."
)
GENERIC_SELF_CARE_WARNING_SIGNS = [
    "Seek immediate emergency care if you develop: chest pain",
    "Seek immediate emergency care if you develop: difficulty breathing",
    "Seek immediate emergency care if you develop: confusion, fainting, or rapid worsening",
]
MENTAL_HEALTH_RESOURCE_MESSAGE = (
    "If you're having thoughts of self-harm, please contact the 988 Suicide & "
    "Crisis Lifeline by calling or texting 988. Help is available 24/7."
)

PATHWAY_DEFAULTS = {
    "911": ("Call 911", "Immediately"),
    "er_now": ("Emergency room now", "Immediately"),
    "urgent_care_today": ("Urgent care today", "Within 24 hours"),
    "doctor_soon": ("Doctor soon", "Within a week"),
    "self_care": ("Self-care", "Monitor at home"),
    "monitor": ("Monitor symptoms", "Monitor at home"),
}
EMERGENCY_PREFIXES = ("⚠️ SEEK EMERGENCY CARE IMMEDIATELY", "🚨 CALL 911 NOW")
DIAGNOSTIC_LANGUAGE_PATTERNS = (
    (
        re.compile(
            r"\b(?:you are|you're|you were)\s+diagnosed with\s+([^.,;\n]+)",
            re.IGNORECASE,
        ),
        r"your symptoms may be consistent with \1",
    ),
    (re.compile(r"\byou have\s+([^.,;\n]+)", re.IGNORECASE), r"your symptoms may be consistent with \1"),
    (re.compile(r"\bthis is\s+([^.,;\n]+)", re.IGNORECASE), r"this could indicate \1"),
    (
        re.compile(r"\byou are having\s+(?:a\s+|an\s+)?([^.,;\n]+)", re.IGNORECASE),
        r"your symptoms may suggest \1",
    ),
    (
        re.compile(r"\bdiagnosed with\s+([^.,;\n]+)", re.IGNORECASE),
        r"symptoms consistent with \1",
    ),
)


def _replace_care_recommendation(
    recommendation: CareRecommendation, **updates: object
) -> CareRecommendation:
    """Rebuild a care recommendation with validation."""

    data = recommendation.model_dump()
    data.update(updates)
    return CareRecommendation.model_validate(data)


def _replace_response(response: TriageResponse, **updates: object) -> TriageResponse:
    """Rebuild a response with validation."""

    data = response.model_dump()
    if "care_recommendation" in updates and isinstance(
        updates["care_recommendation"], CareRecommendation
    ):
        updates["care_recommendation"] = updates["care_recommendation"].model_dump()
    data.update(updates)
    return TriageResponse.model_validate(data)


def _unique_preserve_order(values: List[str]) -> list[str]:
    """Deduplicate while keeping first-seen order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _ensure_prefix(text: str, prefix: str) -> str:
    """Ensure a message starts with the required emergency prefix."""

    if any(text.startswith(existing) for existing in EMERGENCY_PREFIXES):
        return text
    return f"{prefix}: {text}" if text else prefix


def _sanitize_text(text: str) -> str:
    """Hedge overly definitive medical language in free text."""

    sanitized = text
    for pattern, replacement in DIAGNOSTIC_LANGUAGE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _is_fast_path_emergency_message(message: str) -> bool:
    """Detect the hardcoded emergency detector format and leave it untouched."""

    return message.startswith("**CALL 911 NOW**") or message.startswith(
        "**GET IMMEDIATE HELP NOW**"
    )


def _strip_emergency_prefix(text: str) -> str:
    """Remove a leading emergency prefix so it can be rebuilt cleanly."""

    for prefix in EMERGENCY_PREFIXES:
        if text.startswith(f"{prefix}: "):
            return text[len(prefix) + 2 :]
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text


def _enforce_minimum_emergency_pathway(response: TriageResponse) -> TriageResponse:
    """Raise the recommendation to an emergency pathway when required."""

    target_pathway = "911" if response.urgency_level >= 5 else "er_now"
    current_pathway = response.care_recommendation.pathway
    if current_pathway in {"911", "er_now"}:
        target_pathway = "911" if current_pathway == "911" else target_pathway

    if current_pathway not in {"911", "er_now"}:
        logger.warning(
            "Overriding non-emergency pathway '%s' for urgency level %s",
            current_pathway,
            response.urgency_level,
        )

    pathway_label, urgency_window = PATHWAY_DEFAULTS[target_pathway]
    care_recommendation = _replace_care_recommendation(
        response.care_recommendation,
        pathway=target_pathway,
        pathway_label=pathway_label,
        urgency_window=urgency_window,
        self_care_steps=None,
    )
    prefix = "🚨 CALL 911 NOW" if target_pathway == "911" else "⚠️ SEEK EMERGENCY CARE IMMEDIATELY"
    immediate_message = _ensure_prefix(response.immediate_message, prefix)

    return _replace_response(
        response,
        care_recommendation=care_recommendation,
        immediate_message=immediate_message,
        is_emergency=True,
        urgency_level=max(response.urgency_level, 4),
        urgency_label=SEVERITY_5 if max(response.urgency_level, 4) == 5 else SEVERITY_4,
    )


def enforce_disclaimer(response: TriageResponse) -> TriageResponse:
    """Restore the canonical disclaimer exactly."""

    if response.disclaimer == CANONICAL_DISCLAIMER:
        return response

    logger.info("Restoring canonical disclaimer.")
    return _replace_response(response, disclaimer=CANONICAL_DISCLAIMER)


def enforce_emergency_escalation(response: TriageResponse) -> TriageResponse:
    """Ensure emergency cases are routed and messaged appropriately."""

    if not response.is_emergency and response.urgency_level < 4:
        return response

    return _enforce_minimum_emergency_pathway(response)


def sanitize_diagnostic_language(response: TriageResponse) -> TriageResponse:
    """Hedge definitive diagnostic wording in user-facing text."""

    immediate_message = response.immediate_message
    if not _is_fast_path_emergency_message(immediate_message):
        immediate_message = _sanitize_text(immediate_message)

    full_explanation = _sanitize_text(response.full_explanation)
    care_recommendation = _replace_care_recommendation(
        response.care_recommendation,
        reasoning=_sanitize_text(response.care_recommendation.reasoning),
    )

    if (
        immediate_message == response.immediate_message
        and full_explanation == response.full_explanation
        and care_recommendation.reasoning == response.care_recommendation.reasoning
    ):
        return response

    return _replace_response(
        response,
        immediate_message=immediate_message,
        full_explanation=full_explanation,
        care_recommendation=care_recommendation,
    )


def validate_self_care_safety(response: TriageResponse) -> TriageResponse:
    """Ensure low-acuity pathways include clear escalation warnings."""

    if response.care_recommendation.pathway not in {"self_care", "monitor"}:
        return response

    if response.care_recommendation.warning_signs:
        return response

    logger.info("Adding generic escalation warning to self-care guidance.")
    care_recommendation = _replace_care_recommendation(
        response.care_recommendation,
        warning_signs=GENERIC_SELF_CARE_WARNING_SIGNS,
    )
    return _replace_response(response, care_recommendation=care_recommendation)


def add_mental_health_resources(response: TriageResponse) -> TriageResponse:
    """Inject crisis-line messaging and elevate care for self-harm concerns."""

    flags = set(response.safety_flags)
    relevant = {
        InputSafetyFlags.CONTAINS_SUICIDAL_IDEATION,
        InputSafetyFlags.CONTAINS_SELF_HARM,
    }
    if not flags.intersection(relevant):
        return response

    base_message = _strip_emergency_prefix(response.immediate_message)
    if base_message.startswith(MENTAL_HEALTH_RESOURCE_MESSAGE):
        base_message = base_message[len(MENTAL_HEALTH_RESOURCE_MESSAGE) :].lstrip()

    emergency_prefix = (
        "🚨 CALL 911 NOW"
        if response.care_recommendation.pathway == "911"
        else "⚠️ SEEK EMERGENCY CARE IMMEDIATELY"
    )
    immediate_message = (
        f"{emergency_prefix}\n\n{MENTAL_HEALTH_RESOURCE_MESSAGE}\n\n{base_message}"
        if base_message
        else f"{emergency_prefix}\n\n{MENTAL_HEALTH_RESOURCE_MESSAGE}"
    )

    updated = _replace_response(response, immediate_message=immediate_message)
    if updated.care_recommendation.pathway in {"911", "er_now"}:
        return updated

    logger.warning("Escalating mental health response to at least er_now.")
    care_recommendation = _replace_care_recommendation(
        updated.care_recommendation,
        pathway="er_now",
        pathway_label=PATHWAY_DEFAULTS["er_now"][0],
        urgency_window=PATHWAY_DEFAULTS["er_now"][1],
        self_care_steps=None,
    )
    return _replace_response(
        updated,
        care_recommendation=care_recommendation,
        is_emergency=True,
        urgency_level=max(updated.urgency_level, 4),
        urgency_label=SEVERITY_5 if max(updated.urgency_level, 4) == 5 else SEVERITY_4,
    )


def apply_guardrails(response: TriageResponse, safety_flags: List[str]) -> TriageResponse:
    """Run guardrail rules in sequence and never raise to callers."""

    try:
        current = _replace_response(
            response,
            safety_flags=_unique_preserve_order([*response.safety_flags, *safety_flags]),
        )
    except Exception:
        logger.exception("Failed to initialize guardrail pipeline; returning original response.")
        return response

    rules: list[Callable[[TriageResponse], TriageResponse]] = [
        enforce_disclaimer,
        enforce_emergency_escalation,
        sanitize_diagnostic_language,
        validate_self_care_safety,
        add_mental_health_resources,
    ]

    for rule in rules:
        try:
            updated = rule(current)
            if updated != current:
                logger.info("Guardrail applied: %s", rule.__name__)
            current = updated
        except Exception:
            logger.exception("Guardrail rule failed: %s", rule.__name__)

    return current
