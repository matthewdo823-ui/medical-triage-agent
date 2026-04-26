"""Gemma-powered knowledge retrieval uAgent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv
from uagents import Agent, Context

from models.triage import DifferentialDiagnosis, KnowledgeResult, SymptomInput
from prompts.retrieval import KNOWLEDGE_RETRIEVAL_SYSTEM
from utils.llm_clients import GemmaClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

AGENT_NAME = "knowledge-retrieval-agent"
AGENT_DESCRIPTION = "Builds differential diagnoses and supporting medical context."
AGENT_PORT = int(os.getenv("KNOWLEDGE_RETRIEVAL_PORT", "8002"))
AGENT_SEED = os.getenv(
    "KNOWLEDGE_SEED",
    os.getenv(
        "KNOWLEDGE_RETRIEVAL_AGENT_SEED",
        "medical triage knowledge retrieval local development seed",
    ),
)
README_PATH = str(Path(__file__).resolve().parent.parent / "README.md")
ENABLE_MEDICAL_WEB_SEARCH = os.getenv("ENABLE_MEDICAL_WEB_SEARCH", "false").lower() == "true"
WEB_SEARCH_TIMEOUT_SECONDS: Final[float] = 3.0
KNOWLEDGE_ERROR_FLAG = "knowledge_retrieval_error"
ALLOWED_SEARCH_DOMAINS: Final[tuple[str, ...]] = ("mayoclinic.org", "medlineplus.gov")

agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://127.0.0.1:{AGENT_PORT}/submit"],
    readme_path=README_PATH,
    publish_agent_details=True,
    metadata={"description": AGENT_DESCRIPTION},
)


def build_retrieval_user_prompt(symptom_input: SymptomInput) -> str:
    """Construct the knowledge retrieval prompt from intake context."""

    age_range = symptom_input.user_age_range or "Not provided"
    biological_sex = symptom_input.user_biological_sex or "Not provided"
    known_conditions = (
        ", ".join(symptom_input.known_conditions)
        if symptom_input.known_conditions
        else "None reported"
    )
    return (
        "Generate a structured differential diagnosis for this symptom report.\n\n"
        f"Session ID: {symptom_input.session_id}\n"
        f"Age range: {age_range}\n"
        f"Biological sex: {biological_sex}\n"
        f"Known conditions: {known_conditions}\n\n"
        "Symptom description:\n"
        f"{symptom_input.raw_text}\n"
    )


def build_knowledge_error_result(error: Exception | str) -> KnowledgeResult:
    """Return the minimal safe retrieval result on failure."""

    return KnowledgeResult.model_validate(
        {
            "differentials": [],
            "relevant_conditions": [],
            "search_sources": [],
            "knowledge_confidence": 0.1,
            "error_flags": [KNOWLEDGE_ERROR_FLAG, str(error)],
        }
    )


def _extract_search_query(symptom_input: SymptomInput) -> str:
    """Create a compact search query from the symptom description."""

    normalized = " ".join(symptom_input.raw_text.split())
    snippet = " ".join(normalized.split()[:8]) or "symptoms"
    return (
        f"{snippet} differential diagnosis site:mayoclinic.org OR site:medlineplus.gov"
    )


async def maybe_search_medical_sources(symptom_input: SymptomInput) -> list[str]:
    """Optionally augment retrieval with a quick best-effort web search."""

    if not ENABLE_MEDICAL_WEB_SEARCH:
        return []

    query = _extract_search_query(symptom_input)
    try:
        async with asyncio.timeout(WEB_SEARCH_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                )
                response.raise_for_status()

            matches = re.findall(r"uddg=([^&\"]+)", response.text)
            results: list[str] = []
            for match in matches:
                url = unquote(match)
                if any(domain in url for domain in ALLOWED_SEARCH_DOMAINS) and url not in results:
                    results.append(url)
                if len(results) >= 5:
                    break
            return results
    except Exception:
        logger.info("Medical web search skipped or timed out for session %s", symptom_input.session_id)
        return []


async def retrieve_knowledge(symptom_input: SymptomInput) -> KnowledgeResult:
    """Run the Gemma-backed knowledge retrieval flow."""

    try:
        client = GemmaClient()
        raw_payload = await client.complete_json(
            system_prompt=KNOWLEDGE_RETRIEVAL_SYSTEM,
            user_message=build_retrieval_user_prompt(symptom_input),
        )
        differentials = [
            DifferentialDiagnosis.model_validate(item)
            for item in raw_payload.get("differentials", [])
            if isinstance(item, dict)
        ]
        search_sources = raw_payload.get("search_sources", [])
        supplemental_sources = await maybe_search_medical_sources(symptom_input)
        merged_sources = []
        for source in [*search_sources, *supplemental_sources]:
            if isinstance(source, str) and source and source not in merged_sources:
                merged_sources.append(source)

        return KnowledgeResult.model_validate(
            {
                "differentials": differentials,
                "relevant_conditions": raw_payload.get("relevant_conditions", []),
                "search_sources": merged_sources,
                "knowledge_confidence": raw_payload.get("knowledge_confidence", 0.3),
                "error_flags": [],
            }
        )
    except Exception as exc:
        logger.exception("Knowledge retrieval failed for session %s", symptom_input.session_id)
        return build_knowledge_error_result(exc)


@agent.on_message(model=SymptomInput, replies=KnowledgeResult)
async def on_retrieve(ctx: Context, sender: str, msg: SymptomInput) -> None:
    """Handle knowledge retrieval requests from the orchestrator."""

    result = await retrieve_knowledge(msg)
    await ctx.send(sender, result)


@agent.on_event("startup")
async def on_startup(ctx: Context) -> None:
    """Log agent readiness."""

    ctx.logger.info("Knowledge retrieval agent ready at %s", agent.address)


if __name__ == "__main__":
    agent.run()
