"""Tests for fast-path emergency detection."""

from __future__ import annotations

import pytest

from safety.emergency_detector import detect_emergency

EMERGENCY_TEST_CASES = [
    ("chest pain radiating to left arm", True, "cardiac"),
    ("my chest hurts and my left arm feels numb", True, "cardiac"),
    ("crushing chest pressure", True, "cardiac"),
    ("chest pain shortness of breath", True, "cardiac"),
    ("I have heartburn after eating", False, None),
    ("chest cold for 3 days", False, None),
    ("my face is drooping on one side and I can't speak clearly", True, "stroke"),
    ("sudden severe headache worst of my life", True, "stroke"),
    ("sudden arm weakness and facial droop", True, "stroke"),
    ("headache for 2 days", False, None),
    ("my arm is sore from the gym", False, None),
    ("my throat is swelling and I can't breathe after eating peanuts", True, "anaphylaxis"),
    ("allergic reaction can't breathe", True, "anaphylaxis"),
    ("mild allergic reaction to pollen, runny nose", False, None),
    ("I took too many pills", True, "overdose"),
    ("overdose on medication", True, "overdose"),
    ("infant not breathing", True, "pediatric"),
    ("pregnant with heavy bleeding", True, "obstetric"),
    ("I want to die tonight", True, "suicidal"),
]

EXPECTED_RESPONSE_SNIPPETS = {
    "cardiac": "chest pain symptoms",
    "stroke": "exact time symptoms started",
    "anaphylaxis": "epinephrine auto-injector",
    "overdose": "naloxone",
    "pediatric": "child or infant",
    "obstetric": "labor and delivery",
    "suicidal": "988",
}


@pytest.mark.parametrize(
    ("text", "should_trigger", "expected_type"),
    EMERGENCY_TEST_CASES,
)
def test_emergency_detector(text: str, should_trigger: bool, expected_type: str | None) -> None:
    """Verify emergency pattern detection across all major groups."""

    result = detect_emergency(text)
    assert result.emergency_detected == should_trigger, f"Failed for: {text}"

    if should_trigger:
        assert result.override_response is not None
        assert result.passed is False
        snippet = EXPECTED_RESPONSE_SNIPPETS[expected_type]
        assert snippet in result.override_response
    else:
        assert result.override_response is None
        assert result.triggered_rules == []


def test_empty_input_does_not_trigger_emergency() -> None:
    """Blank input should fail open and allow downstream handling."""

    result = detect_emergency("   ")
    assert result.passed is True
    assert result.emergency_detected is False
    assert result.override_response is None
