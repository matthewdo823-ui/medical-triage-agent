"""End-to-end triage pipeline tests with mocked LLM clients."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest

from agents import orchestrator
from models.triage import SymptomInput
from safety.emergency_detector import detect_emergency
from safety.guardrails import apply_guardrails

INTEGRATION_TEST_CASES = [
    {
        "input": "I have a mild runny nose and slight sore throat for 2 days",
        "expected_severity_range": (1, 2),
        "expected_pathway_options": ["self_care", "monitor"],
        "must_have_warning_signs": True,
    },
    {
        "input": "severe chest pain radiating to jaw, shortness of breath, sweating",
        "expected_fast_path": True,
        "response_must_contain": "911",
    },
    {
        "input": "high fever 103 for 2 days, headache, stiff neck, sensitivity to light",
        "expected_severity_range": (4, 5),
        "response_must_contain": "emergency",
    },
]


class FakeGemmaClient:
    """Scenario-aware fake Gemma client for deterministic tests."""

    async def complete_json(self, system_prompt: str, user_message: str, **_: object) -> dict:
        prompt = f"{system_prompt}\n{user_message}".lower()

        if "medical symptom classifier" in system_prompt.lower():
            if "runny nose" in prompt:
                return {
                    "severity_score": 1,
                    "red_flags": [],
                    "symptom_clusters": ["upper respiratory"],
                    "affected_systems": ["respiratory"],
                    "is_emergency": False,
                    "confidence": 0.92,
                }
            if "stiff neck" in prompt:
                return {
                    "severity_score": 4,
                    "red_flags": ["meningitis concern"],
                    "symptom_clusters": ["neurological", "infectious"],
                    "affected_systems": ["neurological"],
                    "is_emergency": True,
                    "confidence": 0.88,
                }

        if "medical knowledge retrieval assistant" in system_prompt.lower():
            if "runny nose" in prompt:
                return {
                    "differentials": [
                        {
                            "condition": "Common cold",
                            "likelihood": "high",
                            "key_matching_symptoms": ["runny nose", "sore throat"],
                            "red_flag_if_present": "difficulty breathing",
                            "is_life_threatening": False,
                        }
                    ],
                    "relevant_conditions": ["viral upper respiratory infection"],
                    "search_sources": ["Mayo Clinic"],
                    "knowledge_confidence": 0.85,
                }
            if "stiff neck" in prompt:
                return {
                    "differentials": [
                        {
                            "condition": "Meningitis",
                            "likelihood": "moderate",
                            "key_matching_symptoms": ["high fever", "stiff neck", "light sensitivity"],
                            "red_flag_if_present": "confusion",
                            "is_life_threatening": True,
                        }
                    ],
                    "relevant_conditions": ["central nervous system infection"],
                    "search_sources": ["MedlinePlus"],
                    "knowledge_confidence": 0.81,
                }

        if "care pathway router" in system_prompt.lower():
            if '"severity_score": 1' in prompt or "severity score: 1" in prompt:
                return {
                    "pathway": "self_care",
                    "pathway_label": "Self-care",
                    "urgency_window": "No rush",
                    "reasoning": "You have a mild cold-like illness.",
                    "immediate_actions": ["Rest", "Hydrate"],
                    "warning_signs": [],
                    "self_care_steps": ["Drink fluids", "Use a humidifier"],
                }
            if '"severity_score": 4' in prompt or "severity score: 4" in prompt:
                return {
                    "pathway": "urgent_care_today",
                    "pathway_label": "Urgent care today",
                    "urgency_window": "Within 24 hours",
                    "reasoning": "You are having a serious infection.",
                    "immediate_actions": ["Have someone stay with you", "Do not drive yourself"],
                    "warning_signs": [
                        "Seek immediate emergency care if you develop: confusion",
                        "Seek immediate emergency care if you develop: vomiting",
                        "Seek immediate emergency care if you develop: worsening pain",
                    ],
                    "self_care_steps": None,
                }

        raise AssertionError(f"Unhandled fake Gemma prompt: {system_prompt[:60]!r}")


class FakeClaudeClient:
    """Simple fake Claude client returning deterministic markdown."""

    async def complete_text(self, system_prompt: str, user_message: str, **_: object) -> str:
        if "runny nose" in user_message.lower():
            return (
                "### What we found\n"
                "You have a cold.\n\n"
                "### What this might mean\n"
                "This is a mild viral illness.\n\n"
                "### What you should do\n"
                "Rest and drink fluids.\n\n"
                "### Watch for these warning signs\n"
                "- Trouble breathing\n\n"
                "### Important reminder\n"
                "placeholder"
            )
        return (
            "### What we found\n"
            "You are having a serious infection.\n\n"
            "### What this might mean\n"
            "This is meningitis.\n\n"
            "### What you should do\n"
            "**Go now.**\n\n"
            "### Watch for these warning signs\n"
            "- Confusion\n\n"
            "### Important reminder\n"
            "placeholder"
        )


async def _run_non_fast_path_pipeline(text: str) -> orchestrator.TriageResponse:
    """Execute the orchestrator helper pipeline for non-fast-path cases."""

    session_id = str(uuid4())
    start_time = datetime.utcnow().timestamp()
    symptom_input = SymptomInput(
        raw_text=text,
        session_id=session_id,
        timestamp=datetime.utcnow(),
    )
    classification, knowledge = await asyncio.gather(
        orchestrator.call_classifier_agent(symptom_input),
        orchestrator.call_knowledge_agent(symptom_input),
    )
    recommendation = await orchestrator.call_router_agent(classification, knowledge)
    triage_response = await orchestrator.synthesize_response(
        symptom_input=symptom_input,
        classification=classification,
        knowledge=knowledge,
        recommendation=recommendation,
        session_id=session_id,
        start_time=start_time,
    )
    safety_result = detect_emergency(text)
    return apply_guardrails(triage_response, safety_result.triggered_rules)


def test_fast_path_emergency_contains_911() -> None:
    """Severe chest pain should bypass the LLM pipeline entirely."""

    case = INTEGRATION_TEST_CASES[1]
    result = detect_emergency(case["input"])
    assert result.requires_911 is True
    assert result.override_response is not None
    assert case["response_must_contain"] in result.override_response


def test_mild_symptom_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mild URI symptoms should stay in the self-care or monitor range."""

    monkeypatch.setattr(orchestrator, "GemmaClient", FakeGemmaClient)
    monkeypatch.setattr(orchestrator, "ClaudeClient", FakeClaudeClient)

    case = INTEGRATION_TEST_CASES[0]
    response = asyncio.run(_run_non_fast_path_pipeline(case["input"]))
    assert case["expected_severity_range"][0] <= response.urgency_level <= case["expected_severity_range"][1]
    assert response.care_recommendation.pathway in case["expected_pathway_options"]
    if case["must_have_warning_signs"]:
        assert len(response.care_recommendation.warning_signs) >= 3


def test_meningitis_like_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """High-risk infectious symptoms should escalate to emergency care."""

    monkeypatch.setattr(orchestrator, "GemmaClient", FakeGemmaClient)
    monkeypatch.setattr(orchestrator, "ClaudeClient", FakeClaudeClient)

    case = INTEGRATION_TEST_CASES[2]
    response = asyncio.run(_run_non_fast_path_pipeline(case["input"]))
    assert case["expected_severity_range"][0] <= response.urgency_level <= case["expected_severity_range"][1]
    formatted = orchestrator.format_chat_response(response).lower()
    assert case["response_must_contain"] in formatted
    assert response.care_recommendation.pathway in {"er_now", "911"}
