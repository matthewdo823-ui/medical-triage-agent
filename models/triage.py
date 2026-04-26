"""Pydantic models for cross-agent triage data contracts.

These schemas define the structured payloads passed between the intake layer,
specialist agents, and the final orchestrator response returned to ASI:One.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Severity scale labels used across classification and final response synthesis.
SEVERITY_1 = "Self-care"
SEVERITY_2 = "Non-urgent"
SEVERITY_3 = "Semi-urgent"
SEVERITY_4 = "Urgent"
SEVERITY_5 = "Life-threatening"

SEVERITY_LABELS = {
    1: SEVERITY_1,
    2: SEVERITY_2,
    3: SEVERITY_3,
    4: SEVERITY_4,
    5: SEVERITY_5,
}

LikelihoodLevel = Literal["high", "moderate", "low"]
CarePathway = Literal[
    "911",
    "er_now",
    "urgent_care_today",
    "doctor_soon",
    "self_care",
    "monitor",
]
UrgencyLabel = Literal[
    "Life-threatening",
    "Urgent",
    "Semi-urgent",
    "Non-urgent",
    "Self-care",
]


class TriageBaseModel(BaseModel):
    """Common base model with strict field handling for triage schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SymptomInput(TriageBaseModel):
    """User-provided symptom description and optional demographic context."""

    raw_text: str = Field(..., min_length=1, description="Original user message.")
    session_id: str = Field(
        ...,
        min_length=1,
        description="UUID string identifying the triage session.",
    )
    timestamp: datetime = Field(..., description="Timestamp for the symptom report.")
    user_age_range: Optional[str] = Field(
        default=None,
        description='Optional age range such as "20-30" or "60+".',
    )
    user_biological_sex: Optional[str] = Field(
        default=None,
        description="Optional biological sex context supplied by the user.",
    )
    known_conditions: Optional[list[str]] = Field(
        default=None,
        description="Self-reported medical conditions relevant to triage.",
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        """Ensure the session identifier is a valid UUID string."""

        UUID(value)
        return value


class DifferentialDiagnosis(TriageBaseModel):
    """Structured representation of a candidate differential diagnosis."""

    condition: str = Field(..., min_length=1)
    likelihood: LikelihoodLevel
    key_matching_symptoms: list[str] = Field(default_factory=list)
    red_flag_if_present: Optional[str] = Field(default=None)
    is_life_threatening: bool


class ClassificationResult(TriageBaseModel):
    """Output produced by the symptom-classifier specialist agent."""

    severity_score: int = Field(..., ge=1, le=5)
    red_flags: list[str] = Field(default_factory=list)
    symptom_clusters: list[str] = Field(default_factory=list)
    affected_systems: list[str] = Field(default_factory=list)
    is_emergency: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_classifier_output: str = Field(..., min_length=1)
    error_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_emergency_status(self) -> "ClassificationResult":
        """Force emergency status to reflect severity and red-flag presence."""

        object.__setattr__(
            self,
            "is_emergency",
            self.severity_score >= 4 or bool(self.red_flags),
        )
        return self


class KnowledgeResult(TriageBaseModel):
    """Knowledge retrieval output used to support differentials and routing."""

    differentials: list[DifferentialDiagnosis] = Field(default_factory=list)
    relevant_conditions: list[str] = Field(default_factory=list)
    search_sources: list[str] = Field(default_factory=list)
    knowledge_confidence: float = Field(..., ge=0.0, le=1.0)
    error_flags: list[str] = Field(default_factory=list)


class CareRecommendation(TriageBaseModel):
    """Recommended next step for the user based on triage findings."""

    pathway: CarePathway
    pathway_label: str = Field(..., min_length=1)
    urgency_window: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)
    immediate_actions: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)
    self_care_steps: Optional[list[str]] = Field(default=None)

    @model_validator(mode="after")
    def validate_self_care_steps(self) -> "CareRecommendation":
        """Restrict self-care guidance to low-acuity pathways."""

        if self.pathway in {"self_care", "monitor"}:
            return self

        if self.self_care_steps:
            raise ValueError(
                "self_care_steps may only be populated for self_care or monitor pathways"
            )

        object.__setattr__(self, "self_care_steps", None)
        return self


class RouterInput(TriageBaseModel):
    """Combined routing payload sent from the orchestrator to the care router."""

    classification: ClassificationResult
    knowledge: KnowledgeResult
    known_conditions: Optional[list[str]] = Field(default=None)


class TriageResponse(TriageBaseModel):
    """Final structured response returned to the user-facing interface."""

    session_id: str = Field(..., min_length=1)
    urgency_level: int = Field(..., ge=1, le=5)
    urgency_label: UrgencyLabel
    care_recommendation: CareRecommendation
    top_differentials: list[DifferentialDiagnosis] = Field(default_factory=list, max_length=3)
    immediate_message: str = Field(..., min_length=1)
    full_explanation: str = Field(..., min_length=1)
    disclaimer: str = Field(..., min_length=1)
    is_emergency: bool
    processing_time_ms: int = Field(..., ge=0)
    safety_flags: list[str] = Field(default_factory=list)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        """Ensure the response references a valid UUID session identifier."""

        UUID(value)
        return value

    @model_validator(mode="after")
    def validate_urgency_consistency(self) -> "TriageResponse":
        """Align urgency labels and emergency status with severity semantics."""

        expected_label = SEVERITY_LABELS[self.urgency_level]
        if self.urgency_label != expected_label:
            raise ValueError(
                f"urgency_label must match severity level {self.urgency_level}: {expected_label}"
            )

        derived_emergency = (
            self.urgency_level >= 4
            or self.care_recommendation.pathway in {"911", "er_now"}
            or any(item.is_life_threatening for item in self.top_differentials)
        )
        if self.is_emergency != derived_emergency:
            raise ValueError(
                "is_emergency must align with urgency level, pathway, and differentials"
            )

        return self
