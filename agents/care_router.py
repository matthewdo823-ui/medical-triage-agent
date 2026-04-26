"""Gemma-powered care routing uAgent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from uagents import Agent, Context

from models.triage import CareRecommendation, RouterInput
from prompts.router import CARE_ROUTER_SYSTEM
from utils.llm_clients import GemmaClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

AGENT_NAME = "care-router-agent"
AGENT_DESCRIPTION = "Maps severity and differential data to a concrete care pathway."
AGENT_PORT = int(os.getenv("CARE_ROUTER_PORT", "8003"))
AGENT_SEED = os.getenv(
    "ROUTER_SEED",
    os.getenv(
        "CARE_ROUTER_AGENT_SEED",
        "medical triage care router local development seed",
    ),
)
README_PATH = str(Path(__file__).resolve().parent.parent / "README.md")

SAFE_DEFAULT_PATHWAY = "urgent_care_today"
SAFE_DEFAULT_WINDOW = "Within 4 hours"
SAFE_DEFAULT_REASONING = (
    "We encountered an issue analyzing your symptoms. Out of caution, we "
    "recommend being evaluated by a medical professional today."
)

agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://127.0.0.1:{AGENT_PORT}/submit"],
    readme_path=README_PATH,
    publish_agent_details=True,
    metadata={"description": AGENT_DESCRIPTION},
)


def build_router_user_prompt(router_input: RouterInput) -> str:
    """Build the user prompt for the care router."""

    top_differentials = [
        item.model_dump()
        for item in router_input.knowledge.differentials[:3]
    ]
    known_conditions = ", ".join(router_input.known_conditions or []) or "None reported"

    return (
        "Route this triage case to the most appropriate care pathway.\n\n"
        f"Severity score: {router_input.classification.severity_score}\n"
        f"Red flags: {router_input.classification.red_flags}\n"
        f"Symptom clusters: {router_input.classification.symptom_clusters}\n"
        f"Known conditions: {known_conditions}\n"
        f"Top differentials: {top_differentials}\n"
    )


def build_router_error_recommendation() -> CareRecommendation:
    """Return the safe care recommendation if routing fails."""

    return CareRecommendation.model_validate(
        {
            "pathway": SAFE_DEFAULT_PATHWAY,
            "pathway_label": "Urgent care today",
            "urgency_window": SAFE_DEFAULT_WINDOW,
            "reasoning": SAFE_DEFAULT_REASONING,
            "immediate_actions": [
                "Arrange prompt medical evaluation today.",
                "Avoid strenuous activity until you are assessed.",
            ],
            "warning_signs": [
                "Seek immediate emergency care if you develop: chest pain",
                "Seek immediate emergency care if you develop: difficulty breathing",
                "Seek immediate emergency care if you develop: confusion or sudden worsening",
            ],
            "self_care_steps": None,
        }
    )


def enforce_high_severity_override(
    recommendation: CareRecommendation,
    router_input: RouterInput,
) -> CareRecommendation:
    """Override low-acuity pathways for high-severity classifications."""

    severity = router_input.classification.severity_score
    if severity < 4:
        return recommendation

    if recommendation.pathway in {"911", "er_now"}:
        return recommendation

    if severity >= 5 or router_input.classification.red_flags:
        return CareRecommendation.model_validate(
            {
                **recommendation.model_dump(),
                "pathway": "911",
                "pathway_label": "Call 911",
                "urgency_window": "Immediately",
                "self_care_steps": None,
            }
        )

    return CareRecommendation.model_validate(
        {
            **recommendation.model_dump(),
            "pathway": "er_now",
            "pathway_label": "Emergency room now",
            "urgency_window": "Immediately",
            "self_care_steps": None,
        }
    )


async def route_care(router_input: RouterInput) -> CareRecommendation:
    """Run the Gemma-backed care routing flow."""

    try:
        client = GemmaClient()
        raw_payload = await client.complete_json(
            system_prompt=CARE_ROUTER_SYSTEM,
            user_message=build_router_user_prompt(router_input),
        )
        recommendation = CareRecommendation.model_validate(
            {
                "pathway": raw_payload.get("pathway", SAFE_DEFAULT_PATHWAY),
                "pathway_label": raw_payload.get("pathway_label", "Urgent care today"),
                "urgency_window": raw_payload.get("urgency_window", SAFE_DEFAULT_WINDOW),
                "reasoning": raw_payload.get("reasoning", SAFE_DEFAULT_REASONING),
                "immediate_actions": raw_payload.get("immediate_actions", []),
                "warning_signs": raw_payload.get("warning_signs", []),
                "self_care_steps": raw_payload.get("self_care_steps"),
            }
        )
        return enforce_high_severity_override(recommendation, router_input)
    except Exception:
        logger.exception("Care routing failed")
        return build_router_error_recommendation()


@agent.on_message(model=RouterInput, replies=CareRecommendation)
async def on_route(ctx: Context, sender: str, msg: RouterInput) -> None:
    """Handle routing requests from the orchestrator."""

    recommendation = await route_care(msg)
    await ctx.send(sender, recommendation)


@agent.on_event("startup")
async def on_startup(ctx: Context) -> None:
    """Log agent readiness."""

    ctx.logger.info("Care router ready at %s", agent.address)


if __name__ == "__main__":
    agent.run()
