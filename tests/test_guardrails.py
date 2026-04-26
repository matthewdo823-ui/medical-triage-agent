"""Tests for deterministic output guardrails."""

from __future__ import annotations

from uuid import uuid4

from models.safety import InputSafetyFlags
from models.triage import CareRecommendation, TriageResponse
from safety.guardrails import CANONICAL_DISCLAIMER, apply_guardrails


def build_response(
    *,
    urgency_level: int = 2,
    urgency_label: str = "Non-urgent",
    pathway: str = "doctor_soon",
    pathway_label: str = "Doctor soon",
    urgency_window: str = "Within a week",
    immediate_message: str = "You have a mild illness.",
    full_explanation: str = "This is a mild illness.",
    disclaimer: str = "modified disclaimer",
    is_emergency: bool = False,
    warning_signs: list[str] | None = None,
) -> TriageResponse:
    """Create a valid baseline response for guardrail tests."""

    return TriageResponse(
        session_id=str(uuid4()),
        urgency_level=urgency_level,
        urgency_label=urgency_label,
        care_recommendation=CareRecommendation(
            pathway=pathway,
            pathway_label=pathway_label,
            urgency_window=urgency_window,
            reasoning="You have a mild illness.",
            immediate_actions=["Rest"],
            warning_signs=warning_signs or [],
            self_care_steps=["Hydrate"] if pathway in {"self_care", "monitor"} else None,
        ),
        top_differentials=[],
        immediate_message=immediate_message,
        full_explanation=full_explanation,
        disclaimer=disclaimer,
        is_emergency=is_emergency,
        processing_time_ms=25,
        safety_flags=[],
    )


def test_disclaimer_always_present() -> None:
    """Guardrails should restore the canonical disclaimer exactly."""

    response = build_response(disclaimer="LLM changed the disclaimer")
    guarded = apply_guardrails(response, [])
    assert guarded.disclaimer == CANONICAL_DISCLAIMER


def test_emergency_escalation() -> None:
    """Severity 4-5 responses must route to er_now or 911."""

    response = build_response(
        urgency_level=5,
        urgency_label="Life-threatening",
        pathway="doctor_soon",
        pathway_label="Doctor soon",
        urgency_window="Within a week",
        immediate_message="You are having a heart attack.",
        full_explanation="This is a heart attack.",
        is_emergency=True,
    )
    guarded = apply_guardrails(response, [])
    assert guarded.care_recommendation.pathway in {"er_now", "911"}
    assert guarded.immediate_message.startswith(("⚠️ SEEK EMERGENCY CARE IMMEDIATELY", "🚨 CALL 911 NOW"))


def test_diagnostic_language_sanitized() -> None:
    """Definitive diagnostic phrasing should be hedged before display."""

    response = build_response(
        immediate_message="You have diabetes.",
        full_explanation="This is diabetes and you are diagnosed with diabetes.",
    )
    guarded = apply_guardrails(response, [])
    assert "You have diabetes" not in guarded.immediate_message
    assert "your symptoms may be consistent with diabetes" in guarded.immediate_message.lower()
    assert "this could indicate diabetes" in guarded.full_explanation.lower()


def test_self_care_always_has_warning_signs() -> None:
    """Self-care guidance must include clear escalation warnings."""

    response = build_response(
        urgency_level=1,
        urgency_label="Self-care",
        pathway="self_care",
        pathway_label="Self-care",
        urgency_window="No rush",
        warning_signs=[],
    )
    guarded = apply_guardrails(response, [])
    assert len(guarded.care_recommendation.warning_signs) >= 3
    assert all(
        sign.startswith("Seek immediate emergency care if you develop:")
        for sign in guarded.care_recommendation.warning_signs
    )


def test_mental_health_resources() -> None:
    """Suicidal ideation should add 988 resources and escalate care."""

    response = build_response(
        urgency_level=2,
        urgency_label="Non-urgent",
        pathway="monitor",
        pathway_label="Monitor symptoms",
        urgency_window="No rush",
        immediate_message="Please monitor symptoms.",
        full_explanation="You have stress.",
    )
    guarded = apply_guardrails(response, [InputSafetyFlags.CONTAINS_SUICIDAL_IDEATION])
    assert "988 Suicide & Crisis Lifeline" in guarded.immediate_message
    assert guarded.care_recommendation.pathway in {"er_now", "911"}
    assert guarded.urgency_level >= 4
