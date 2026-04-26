"""Gemma-powered symptom classifier uAgent."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from uagents import Agent, Context

from models.triage import ClassificationResult, SymptomInput
from prompts.classifier import SYMPTOM_CLASSIFIER_SYSTEM, build_classifier_user_prompt
from utils.llm_clients import GemmaClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

AGENT_NAME = "symptom-classifier-agent"
AGENT_DESCRIPTION = "Classifies symptom severity, red flags, and affected systems."
AGENT_PORT = int(os.getenv("SYMPTOM_CLASSIFIER_PORT", "8001"))
AGENT_SEED = os.getenv(
    "CLASSIFIER_SEED",
    os.getenv(
        "SYMPTOM_CLASSIFIER_AGENT_SEED",
        "medical triage symptom classifier local development seed",
    ),
)
README_PATH = str(Path(__file__).resolve().parent.parent / "README.md")
CLASSIFIER_ERROR_FLAG = "classifier_error"
SAFE_DEFAULT_SEVERITY = 3
SAFE_DEFAULT_CONFIDENCE = 0.1

agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://127.0.0.1:{AGENT_PORT}/submit"],
    readme_path=README_PATH,
    publish_agent_details=True,
    metadata={"description": AGENT_DESCRIPTION},
)


def build_classifier_error_result(error: Exception | str) -> ClassificationResult:
    """Return a safe default classification when the LLM fails."""

    error_message = str(error)
    return ClassificationResult.model_validate(
        {
            "severity_score": SAFE_DEFAULT_SEVERITY,
            "red_flags": [],
            "symptom_clusters": ["unclassified"],
            "affected_systems": [],
            "is_emergency": False,
            "confidence": SAFE_DEFAULT_CONFIDENCE,
            "raw_classifier_output": error_message or "classifier_error",
            "error_flags": [CLASSIFIER_ERROR_FLAG],
        }
    )


async def classify_symptoms(symptom_input: SymptomInput) -> ClassificationResult:
    """Run the Gemma-backed symptom classifier."""

    try:
        client = GemmaClient()
        raw_payload = await client.complete_json(
            system_prompt=SYMPTOM_CLASSIFIER_SYSTEM,
            user_message=build_classifier_user_prompt(symptom_input),
        )
        return ClassificationResult.model_validate(
            {
                "severity_score": raw_payload.get("severity_score", SAFE_DEFAULT_SEVERITY),
                "red_flags": raw_payload.get("red_flags", []),
                "symptom_clusters": raw_payload.get("symptom_clusters", []),
                "affected_systems": raw_payload.get("affected_systems", []),
                "is_emergency": raw_payload.get("is_emergency", False),
                "confidence": raw_payload.get("confidence", SAFE_DEFAULT_CONFIDENCE),
                "raw_classifier_output": json.dumps(raw_payload),
                "error_flags": [],
            }
        )
    except Exception as exc:
        logger.exception("Symptom classifier failed for session %s", symptom_input.session_id)
        return build_classifier_error_result(exc)


@agent.on_message(model=SymptomInput, replies=ClassificationResult)
async def on_classify(ctx: Context, sender: str, msg: SymptomInput) -> None:
    """Handle symptom classification requests from the orchestrator."""

    result = await classify_symptoms(msg)
    if result.is_emergency:
        ctx.logger.warning(
            "Emergency classification detected for session %s with severity=%s and red_flags=%s",
            msg.session_id,
            result.severity_score,
            result.red_flags,
        )
    await ctx.send(sender, result)


@agent.on_event("startup")
async def on_startup(ctx: Context) -> None:
    """Log agent readiness."""

    ctx.logger.info("Symptom classifier ready at %s", agent.address)


if __name__ == "__main__":
    agent.run()
