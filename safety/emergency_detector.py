"""Fast-path emergency detection before any LLM invocation.

This module performs low-latency regex matching for critical symptom patterns.
It is designed to run before any model call so urgent cases can be escalated
immediately with a pre-reviewed response.
"""

from __future__ import annotations

import logging
import re
from typing import Final

from models.safety import InputSafetyFlags, SafetyCheckResult

logger = logging.getLogger(__name__)

CARDIAC_EMERGENCY: Final[str] = "cardiac_emergency"
STROKE: Final[str] = "stroke"
ANAPHYLAXIS: Final[str] = "anaphylaxis"
OVERDOSE: Final[str] = "overdose"
PEDIATRIC_CRITICAL: Final[str] = "pediatric_critical"
OBSTETRIC: Final[str] = "obstetric"
SUICIDAL_CRISIS: Final[str] = "suicidal_crisis"


def _compile(pattern: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex once at import time."""

    return re.compile(pattern, re.IGNORECASE)


EMERGENCY_PATTERNS: Final[tuple[tuple[str, str, bool, tuple[re.Pattern[str], ...]], ...]] = (
    (
        CARDIAC_EMERGENCY,
        InputSafetyFlags.CONTAINS_CHEST_PAIN_PLUS_SHORTNESS,
        True,
        (
            _compile(
                r"(?=.*\b(chest\s+pain|chest\s+hurts|chest\s+pressure|chest\s+tightness)\b)"
                r"(?=.*\b(left\s+arm|jaw|shoulder|back)\b).+"
            ),
            _compile(r"\bheart\s+attack\b"),
            _compile(r"\bcrushing\s+chest(?:\s+pressure)?\b"),
            _compile(
                r"(?:(?=.*\b(chest\s+pain|chest\s+hurts|chest\s+pressure|chest\s+tightness)\b)(?=.*\b(shortness\s+of\s+breath|can'?t\s+breathe|cannot\s+breathe|trouble\s+breathing)\b))"
            ),
            _compile(r"\bcardiac\s+arrest\b"),
        ),
    ),
    (
        STROKE,
        InputSafetyFlags.CONTAINS_STROKE_SYMPTOMS,
        True,
        (
            _compile(r"\b(face(?:\s+is)?\s+drooping|facial\s+droop)\b"),
            _compile(r"(?=.*\barm\s+weakness\b)(?=.*\bsudden\b)"),
            _compile(
                r"(?:(?=.*\b(slurred\s+speech|can(?:not|'?t)\s+speak|can(?:not|'?t)\s+speak\s+clearly)\b)(?=.*\bsudden(?:ly)?\b))"
            ),
            _compile(
                r"\b(sudden\s+severe\s+headache|worst\s+headache\s+of\s+my\s+life)\b"
            ),
            _compile(
                r"(?:(?=.*\bface(?:\s+is)?\s+drooping\b)(?=.*\bcan(?:not|'?t)\s+speak\b))"
            ),
            _compile(r"\bstroke\b"),
        ),
    ),
    (
        ANAPHYLAXIS,
        InputSafetyFlags.CONTAINS_ANAPHYLAXIS,
        True,
        (
            _compile(
                r"(?:(?=.*\bthroat(?:\s+is)?\s+swelling\b)(?=.*\b(difficulty\s+breathing|can'?t\s+breathe|cannot\s+breathe|trouble\s+breathing)\b))"
            ),
            _compile(
                r"(?:(?=.*\ballergic\s+reaction\b)(?=.*\b(can'?t\s+breathe|cannot\s+breathe)\b))"
            ),
            _compile(r"\bepipen\b"),
            _compile(r"\banaphylaxis\b"),
        ),
    ),
    (
        OVERDOSE,
        InputSafetyFlags.CONTAINS_OVERDOSE,
        True,
        (
            _compile(
                r"\btook\s+too\s+many\b\s+(?:of\s+)?(?:my\s+)?[a-z][a-z0-9-]{2,}(?:\s+[a-z][a-z0-9-]{2,}){0,3}"
            ),
            _compile(r"\boverdose\b"),
            _compile(r"(?:(?=.*\bnot\s+breathing\b)(?=.*\bunconscious\b))"),
        ),
    ),
    (
        PEDIATRIC_CRITICAL,
        InputSafetyFlags.CONTAINS_PEDIATRIC_EMERGENCY,
        True,
        (
            _compile(r"(?:(?=.*\binfant\b)(?=.*\bnot\s+breathing\b))"),
            _compile(r"(?:(?=.*\bchild\b)(?=.*\bseizure\b)(?=.*\bfirst\s+time\b))"),
            _compile(r"(?:(?=.*\bbaby\b)(?=.*\b(blue|cyanotic)\b))"),
        ),
    ),
    (
        OBSTETRIC,
        InputSafetyFlags.CONTAINS_OBSTETRIC_EMERGENCY,
        True,
        (
            _compile(r"(?:(?=.*\bpregnant\b)(?=.*\bheavy\s+bleeding\b))"),
            _compile(
                r"(?:(?=.*\bwater\s+broke\b)(?=.*\b(cord|umbilical\s+cord|prolapse|something\s+coming\s+out)\b))"
            ),
        ),
    ),
    (
        SUICIDAL_CRISIS,
        InputSafetyFlags.CONTAINS_SUICIDAL_IDEATION,
        False,
        (
            _compile(
                r"(?:(?=.*\bwant\s+to\s+die\b)(?=.*\b(now|tonight|today|immediately|right\s+now)\b))"
            ),
            _compile(r"\bend\s+my\s+life\b"),
            _compile(r"\bsuicide\b"),
        ),
    ),
)


def build_emergency_response(emergency_type: str) -> str:
    """Return a pre-written escalation response for a detected emergency."""

    responses = {
        CARDIAC_EMERGENCY: (
            "**CALL 911 NOW**\n\n"
            "1. Call 911 immediately or have someone call for you.\n"
            "2. Unlock the door if possible and sit or lie down somewhere safe.\n"
            "3. Do not drive yourself to the hospital.\n\n"
            "Tell responders: chest pain symptoms, when they started, where the pain is "
            "spreading, and whether you are short of breath.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        STROKE: (
            "**CALL 911 NOW**\n\n"
            "1. Call 911 immediately and note the exact time symptoms started.\n"
            "2. Stay seated or lying down somewhere safe.\n"
            "3. Do not eat, drink, or drive yourself.\n\n"
            "Tell responders: what symptoms appeared suddenly, which side is affected, "
            "and the last time you were known to be normal.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        ANAPHYLAXIS: (
            "**CALL 911 NOW**\n\n"
            "1. Use an epinephrine auto-injector immediately if one is available.\n"
            "2. Call 911 right away even if symptoms improve.\n"
            "3. Lie down with legs elevated unless breathing is easier sitting up.\n\n"
            "Tell responders: what may have triggered the reaction, whether epinephrine "
            "was used, and if throat swelling or breathing trouble is worsening.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        OVERDOSE: (
            "**CALL 911 NOW**\n\n"
            "1. Call 911 immediately.\n"
            "2. If the person is not breathing, begin rescue breathing if you know how.\n"
            "3. If naloxone is available and an opioid may be involved, give it now.\n\n"
            "Tell responders: what substance may have been taken, how much, when, and "
            "whether the person is awake or breathing.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        PEDIATRIC_CRITICAL: (
            "**CALL 911 NOW**\n\n"
            "1. Call 911 immediately for the child or infant.\n"
            "2. If the child is not breathing, begin age-appropriate CPR if trained.\n"
            "3. Keep the child warm and monitor breathing until help arrives.\n\n"
            "Tell responders: the child's approximate age, exact symptoms, when they "
            "started, and whether this is a first-time event.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        OBSTETRIC: (
            "**CALL 911 NOW**\n\n"
            "1. Call 911 or go to the nearest labor and delivery emergency department "
            "immediately.\n"
            "2. Lie down on your left side if you can do so safely.\n"
            "3. Do not insert anything vaginally or pull on anything protruding.\n\n"
            "Tell responders: how far along the pregnancy is, the amount of bleeding or "
            "fluid loss, and whether you see or feel cord tissue.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
        SUICIDAL_CRISIS: (
            "**GET IMMEDIATE HELP NOW**\n\n"
            "1. Call or text 988 right now for the Suicide & Crisis Lifeline.\n"
            "2. If you might act on these thoughts, call 911 now or go to the nearest "
            "emergency room.\n"
            "3. Move closer to another person, and put distance between yourself and any "
            "medications, weapons, or other means of self-harm.\n\n"
            "Tell responders or the crisis counselor: whether you are in immediate danger, "
            "if you have a plan, and whether you are alone right now.\n\n"
            "This is an emergency safety response and not a substitute for professional care."
        ),
    }
    return responses.get(
        emergency_type,
        (
            "**SEEK IMMEDIATE EMERGENCY HELP**\n\n"
            "1. Call 911 now.\n"
            "2. Stay with another person if possible.\n"
            "3. Share your symptoms and when they began.\n\n"
            "Tell responders what is happening and whether symptoms are worsening.\n\n"
            "This is an emergency safety response and not a diagnosis."
        ),
    )


def detect_emergency(text: str) -> SafetyCheckResult:
    """Run fast-path emergency matching and return the first applicable response."""

    try:
        normalized = text.strip()
        if not normalized:
            return SafetyCheckResult(
                passed=True,
                triggered_rules=[],
                override_response=None,
                emergency_detected=False,
                requires_911=False,
            )

        for emergency_type, rule_flag, requires_911, patterns in EMERGENCY_PATTERNS:
            for pattern in patterns:
                if pattern.search(normalized):
                    return SafetyCheckResult(
                        passed=False,
                        triggered_rules=[rule_flag],
                        override_response=build_emergency_response(emergency_type),
                        emergency_detected=True,
                        requires_911=requires_911,
                    )

        return SafetyCheckResult(
            passed=True,
            triggered_rules=[],
            override_response=None,
            emergency_detected=False,
            requires_911=False,
        )
    except Exception:
        logger.exception("Emergency detector failed; falling open to downstream handling.")
        return SafetyCheckResult(
            passed=True,
            triggered_rules=[],
            override_response=None,
            emergency_detected=False,
            requires_911=False,
        )
