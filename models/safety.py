"""Pydantic models and constants for safety validation and enforcement."""

from __future__ import annotations

from typing import Final, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SafetyBaseModel(BaseModel):
    """Common base model with strict field handling for safety schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputSafetyFlags:
    """String constants used by emergency detection and input guardrails."""

    CONTAINS_CHEST_PAIN_PLUS_SHORTNESS: Final[str] = "chest_pain_dyspnea"
    CONTAINS_RESPIRATORY_DISTRESS: Final[str] = "respiratory_distress"
    CONTAINS_STROKE_SYMPTOMS: Final[str] = "stroke_symptoms"
    CONTAINS_ANAPHYLAXIS: Final[str] = "anaphylaxis"
    CONTAINS_OVERDOSE: Final[str] = "overdose"
    CONTAINS_SUICIDAL_IDEATION: Final[str] = "suicidal_ideation"
    CONTAINS_PEDIATRIC_EMERGENCY: Final[str] = "pediatric_emergency"
    CONTAINS_OBSTETRIC_EMERGENCY: Final[str] = "obstetric_emergency"
    CONTAINS_SELF_HARM: Final[str] = "self_harm"
    AMBIGUOUS_SEVERITY: Final[str] = "ambiguous"

    ALL: Final[tuple[str, ...]] = (
        CONTAINS_CHEST_PAIN_PLUS_SHORTNESS,
        CONTAINS_RESPIRATORY_DISTRESS,
        CONTAINS_STROKE_SYMPTOMS,
        CONTAINS_ANAPHYLAXIS,
        CONTAINS_OVERDOSE,
        CONTAINS_SUICIDAL_IDEATION,
        CONTAINS_PEDIATRIC_EMERGENCY,
        CONTAINS_OBSTETRIC_EMERGENCY,
        CONTAINS_SELF_HARM,
        AMBIGUOUS_SEVERITY,
    )


class SafetyCheckResult(SafetyBaseModel):
    """Result of pre-LLM or post-LLM safety validation."""

    passed: bool
    triggered_rules: list[str] = Field(default_factory=list)
    override_response: Optional[str] = Field(default=None)
    emergency_detected: bool
    requires_911: bool

    @model_validator(mode="after")
    def align_emergency_flags(self) -> "SafetyCheckResult":
        """Keep emergency flags internally consistent."""

        if self.requires_911:
            object.__setattr__(self, "emergency_detected", True)
        return self
