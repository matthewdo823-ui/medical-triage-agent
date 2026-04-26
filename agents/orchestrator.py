"""Main Fetch.ai orchestration agent for the medical triage system."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiosqlite
import httpx
from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    MetadataContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

try:
    from fetchai.registration import register_with_agentverse
    from uagents_core.identity import Identity

    FETCHAI_REGISTRATION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    FETCHAI_REGISTRATION_AVAILABLE = False
    Identity = None
    register_with_agentverse = None

try:
    from uagents_core.contrib.protocols.payment import (
        CommitPayment,
        CompletePayment,
        Funds,
        RejectPayment,
        RequestPayment,
        payment_protocol_spec,
    )

    PAYMENT_PROTOCOL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    PAYMENT_PROTOCOL_AVAILABLE = False
    CommitPayment = CompletePayment = Funds = RejectPayment = RequestPayment = None
    payment_protocol_spec = None

from models.triage import (
    CareRecommendation,
    ClassificationResult,
    DifferentialDiagnosis,
    KnowledgeResult,
    SEVERITY_LABELS,
    SymptomInput,
    TriageResponse,
)
from prompts.classifier import (
    SYMPTOM_CLASSIFIER_SYSTEM,
    build_classifier_user_prompt,
)
from prompts.retrieval import KNOWLEDGE_RETRIEVAL_SYSTEM
from prompts.router import CARE_ROUTER_SYSTEM
from safety.emergency_detector import detect_emergency
from safety.guardrails import CANONICAL_DISCLAIMER, apply_guardrails
from utils.llm_clients import ClaudeClient, ClaudeClientError, GemmaClient, RESPONSE_SYNTHESIS_SYSTEM

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

AGENT_NAME = "medical-triage-agent"
AGENT_DESCRIPTION = (
    "I help analyze your medical symptoms and recommend the appropriate level of "
    "care - from self-care at home to calling 911 immediately. Describe your "
    "symptoms and I'll assess urgency, explain possible causes, and tell you "
    "exactly what to do next. NOTE: This is for informational triage only and is "
    "not a substitute for professional medical advice."
)
AGENT_TAGS = ["medical", "health", "triage", "symptoms", "healthcare", "emergency"]
AGENT_PORT = 8000
DB_PATH = Path(__file__).resolve().parent.parent / "triage_sessions.db"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"
PUBLIC_AGENT_ENDPOINT = os.getenv(
    "PUBLIC_AGENT_ENDPOINT",
    f"http://127.0.0.1:{AGENT_PORT}/submit",
)
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY", "")
AGENTVERSE_BASE_URL = os.getenv("AGENTVERSE_BASE_URL", "https://agentverse.ai").rstrip("/")
ENABLE_PAYMENT_GATE = (
    os.getenv("ENABLE_PAYMENT_PROTOCOL", os.getenv("ENABLE_PAYMENT_GATE", "false")).lower()
    == "true"
)
PAYMENT_AMOUNT = os.getenv("PAYMENT_AMOUNT_ATESTFET", os.getenv("TRIAGE_PAYMENT_AMOUNT", "10"))
PAYMENT_CURRENCY = os.getenv("TRIAGE_PAYMENT_CURRENCY", "atestfet")
PAYMENT_METHOD = os.getenv("TRIAGE_PAYMENT_METHOD", "skyfire")
PAYMENT_DEADLINE_SECONDS = int(os.getenv("TRIAGE_PAYMENT_DEADLINE_SECONDS", "900"))
ORCHESTRATOR_AGENT_SEED = os.getenv(
    "ORCHESTRATOR_SEED",
    os.getenv(
        "ORCHESTRATOR_AGENT_SEED",
        "medical triage orchestrator local development seed",
    ),
)

CHAT_PROTOCOL_NAME = "AgentChatProtocol"
CHAT_PROTOCOL_VERSION = "0.3.0"
REGISTRATION_METADATA = {
    "description": AGENT_DESCRIPTION,
    "tags": AGENT_TAGS,
    "keywords": AGENT_TAGS,
    "category": "healthcare",
}

agent = Agent(
    name=AGENT_NAME,
    seed=ORCHESTRATOR_AGENT_SEED,
    port=AGENT_PORT,
    mailbox=True,
    readme_path=str(README_PATH),
    publish_agent_details=True,
    metadata=REGISTRATION_METADATA,
)
chat_protocol = Protocol(
    name=CHAT_PROTOCOL_NAME,
    version=CHAT_PROTOCOL_VERSION,
    spec=chat_protocol_spec,
)
payment_protocol = (
    Protocol(spec=payment_protocol_spec, role="seller")
    if PAYMENT_PROTOCOL_AVAILABLE
    else None
)

_pending_paid_reports: dict[str, dict[str, Any]] = {}


def _compute_processing_time_ms(start_time: float) -> int:
    """Compute non-negative processing duration in milliseconds."""

    elapsed_ms = int((time.time() - start_time) * 1000)
    return max(0, elapsed_ms)


def _build_text_message(content: str) -> ChatMessage:
    """Create a chat-protocol response message with a text payload."""

    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=content),
            EndSessionContent(type="end-session"),
        ],
    )


def _read_agent_readme() -> str:
    """Return the README text used for Agentverse registration."""

    try:
        return README_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("README not found for Agentverse registration at %s", README_PATH)
        return f"# {AGENT_NAME}\n\n{AGENT_DESCRIPTION}\n"


async def _register_with_fetchai_helper(ctx: Context) -> bool:
    """Register the agent through the optional Fetch.ai helper SDK."""

    if not FETCHAI_REGISTRATION_AVAILABLE or not register_with_agentverse or not Identity:
        return False

    try:
        await asyncio.to_thread(
            register_with_agentverse,
            identity=Identity.from_seed(ORCHESTRATOR_AGENT_SEED, 0),
            url=PUBLIC_AGENT_ENDPOINT,
            agentverse_token=AGENTVERSE_API_KEY,
            agent_title=AGENT_NAME,
            readme=_read_agent_readme(),
        )
        ctx.logger.info("Registered orchestrator with Agentverse via fetchai helper.")
        return True
    except Exception:
        ctx.logger.exception("Fetch.ai Agentverse helper registration failed.")
        return False


async def _register_with_agentverse_api(ctx: Context) -> None:
    """Register or update the agent listing and discoverability keywords."""

    payload = {
        "address": agent.address,
        "name": AGENT_NAME,
        "url": PUBLIC_AGENT_ENDPOINT,
        "agent_type": "uagent",
        "profile": {"description": AGENT_DESCRIPTION},
        "endpoints": [{"url": PUBLIC_AGENT_ENDPOINT, "weight": 1}],
        "metadata": REGISTRATION_METADATA,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        register_response = await client.post(
            f"{AGENTVERSE_BASE_URL}/v2/agents",
            headers={"Authorization": f"Bearer {AGENTVERSE_API_KEY}"},
            json=payload,
        )
        register_response.raise_for_status()

        keyword_response = await client.put(
            f"{AGENTVERSE_BASE_URL}/v1/search/agents/{agent.address}/target-keywords",
            headers={"Authorization": f"Bearer {AGENTVERSE_API_KEY}"},
            json={"keywords": AGENT_TAGS},
        )
        keyword_response.raise_for_status()

    ctx.logger.info("Registered orchestrator listing and keywords with Agentverse APIs.")


async def register_agentverse_listing(ctx: Context) -> None:
    """Best-effort Agentverse registration for discoverability."""

    if not AGENTVERSE_API_KEY:
        ctx.logger.info("AGENTVERSE_API_KEY not set; skipping explicit Agentverse registration.")
        return

    if await _register_with_fetchai_helper(ctx):
        try:
            await _register_with_agentverse_api(ctx)
        except Exception:
            ctx.logger.exception("Keyword sync failed after helper registration.")
        return

    try:
        await _register_with_agentverse_api(ctx)
    except Exception:
        ctx.logger.exception("Agentverse API registration failed.")


def _extract_chat_text(msg: ChatMessage) -> str:
    """Flatten text content from a chat protocol message."""

    text_parts: list[str] = []
    for item in msg.content if isinstance(msg.content, list) else []:
        if isinstance(item, TextContent):
            text_parts.append(item.text)
        elif hasattr(item, "text") and getattr(item, "text", None):
            text_parts.append(str(item.text))

    return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()


def _is_session_start_message(msg: ChatMessage) -> bool:
    """Detect chat envelopes that only initialize a session."""

    return any(isinstance(item, StartSessionContent) for item in msg.content)


def _build_session_ready_message() -> ChatMessage:
    """Return a lightweight chat response for session initialization."""

    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            MetadataContent(
                type="metadata",
                metadata={"status": "ready", "agent": AGENT_NAME},
            ),
            TextContent(
                type="text",
                text=(
                    "Medical triage session started. Describe your symptoms in a sentence "
                    "or two, including how severe they feel and when they began."
                ),
            ),
        ],
    )


def _hash_input(raw_input: str) -> str:
    """Hash user input before persistence to avoid storing symptom text."""

    return hashlib.sha256(raw_input.encode("utf-8")).hexdigest()


async def initialize_database() -> None:
    """Create the session log table if it does not already exist."""

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS triage_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                severity_score INTEGER,
                pathway TEXT,
                processing_time_ms INTEGER,
                safety_flags_triggered TEXT NOT NULL,
                fast_path_used INTEGER NOT NULL
            )
            """
        )
        await db.commit()


async def log_session(
    session_id: str,
    status: str,
    raw_input: str,
    response: Optional[TriageResponse] = None,
    fast_path_used: bool = False,
    safety_flags_triggered: Optional[list[str]] = None,
) -> None:
    """Persist privacy-preserving session metadata to local SQLite."""

    try:
        flags = safety_flags_triggered or (response.safety_flags if response else [])
        severity = response.urgency_level if response else None
        pathway = response.care_recommendation.pathway if response else None
        processing_time_ms = response.processing_time_ms if response else None

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO triage_sessions (
                    session_id,
                    timestamp,
                    input_hash,
                    severity_score,
                    pathway,
                    processing_time_ms,
                    safety_flags_triggered,
                    fast_path_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    _hash_input(raw_input),
                    severity,
                    pathway,
                    processing_time_ms,
                    json.dumps(flags),
                    int(fast_path_used),
                ),
            )
            await db.commit()
        logger.info("Logged triage session %s with status=%s", session_id, status)
    except Exception:
        logger.exception("Failed to log triage session %s", session_id)


def _sanitize_differentials_for_response(
    differentials: list[DifferentialDiagnosis], is_emergency: bool
) -> list[DifferentialDiagnosis]:
    """Select up to three user-facing differentials without over-escalating benign cases."""

    if is_emergency:
        return differentials[:3]

    non_life_threatening = [item for item in differentials if not item.is_life_threatening]
    if non_life_threatening:
        return non_life_threatening[:3]
    return differentials[:3]


def _build_knowledge_user_prompt(symptom_input: SymptomInput) -> str:
    """Construct the user prompt for the knowledge retrieval specialist."""

    age_range = symptom_input.user_age_range or "Not provided"
    biological_sex = symptom_input.user_biological_sex or "Not provided"
    known_conditions = (
        ", ".join(symptom_input.known_conditions)
        if symptom_input.known_conditions
        else "None reported"
    )

    return (
        "Provide structured medical knowledge support for the following symptom report.\n\n"
        f"Session ID: {symptom_input.session_id}\n"
        f"Age range: {age_range}\n"
        f"Biological sex: {biological_sex}\n"
        f"Known conditions: {known_conditions}\n\n"
        "Symptom description:\n"
        f"{symptom_input.raw_text}\n"
    )


def _build_router_user_prompt(
    classification: ClassificationResult,
    knowledge: KnowledgeResult,
) -> str:
    """Construct the user prompt for the care routing specialist."""

    payload = {
        "severity_score": classification.severity_score,
        "red_flags": classification.red_flags,
        "symptom_clusters": classification.symptom_clusters,
        "affected_systems": classification.affected_systems,
        "is_emergency": classification.is_emergency,
        "differentials": [item.model_dump() for item in knowledge.differentials],
        "relevant_conditions": knowledge.relevant_conditions,
    }
    return (
        "Determine the appropriate care pathway for this triage case.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _build_synthesis_user_message(
    symptom_input: SymptomInput,
    classification: ClassificationResult,
    knowledge: KnowledgeResult,
    recommendation: CareRecommendation,
) -> str:
    """Create the structured synthesis payload for Claude."""

    payload = {
        "user_report": symptom_input.raw_text,
        "age_range": symptom_input.user_age_range,
        "biological_sex": symptom_input.user_biological_sex,
        "known_conditions": symptom_input.known_conditions or [],
        "classification": classification.model_dump(),
        "knowledge": knowledge.model_dump(),
        "recommendation": recommendation.model_dump(),
        "canonical_disclaimer": CANONICAL_DISCLAIMER,
    }
    return (
        "Write the patient-facing markdown explanation for this triage case.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _build_immediate_message(
    classification: ClassificationResult,
    recommendation: CareRecommendation,
) -> str:
    """Generate a short high-priority message for the top of the response."""

    if classification.severity_score >= 5 or recommendation.pathway == "911":
        return (
            "🚨 CALL 911 NOW: Your symptoms may indicate a medical emergency. "
            "Do not drive yourself."
        )
    if classification.severity_score >= 4 or recommendation.pathway == "er_now":
        return (
            "⚠️ SEEK EMERGENCY CARE IMMEDIATELY: Your symptoms may need urgent evaluation today."
        )
    return recommendation.reasoning


async def call_classifier_agent(symptom_input: SymptomInput) -> ClassificationResult:
    """Invoke the classifier specialist through Gemma."""

    client = GemmaClient()
    raw_payload = await client.complete_json(
        system_prompt=SYMPTOM_CLASSIFIER_SYSTEM,
        user_message=build_classifier_user_prompt(symptom_input),
    )
    classification_data = {
        "severity_score": raw_payload.get("severity_score", 3),
        "red_flags": raw_payload.get("red_flags", []),
        "symptom_clusters": raw_payload.get("symptom_clusters", []),
        "affected_systems": raw_payload.get("affected_systems", []),
        "is_emergency": raw_payload.get("is_emergency", False),
        "confidence": raw_payload.get("confidence", 0.3),
        "raw_classifier_output": json.dumps(raw_payload),
    }
    return ClassificationResult.model_validate(classification_data)


async def call_knowledge_agent(symptom_input: SymptomInput) -> KnowledgeResult:
    """Invoke the retrieval specialist through Gemma."""

    client = GemmaClient()
    raw_payload = await client.complete_json(
        system_prompt=KNOWLEDGE_RETRIEVAL_SYSTEM,
        user_message=_build_knowledge_user_prompt(symptom_input),
    )

    differential_payloads = raw_payload.get("differentials", [])
    differentials = [
        DifferentialDiagnosis.model_validate(item)
        for item in differential_payloads
        if isinstance(item, dict)
    ]

    knowledge_data = {
        "differentials": differentials,
        "relevant_conditions": raw_payload.get("relevant_conditions", []),
        "search_sources": raw_payload.get("search_sources", []),
        "knowledge_confidence": raw_payload.get("knowledge_confidence", 0.3),
    }
    return KnowledgeResult.model_validate(knowledge_data)


async def call_router_agent(
    classification: ClassificationResult,
    knowledge: KnowledgeResult,
) -> CareRecommendation:
    """Invoke the routing specialist through Gemma."""

    client = GemmaClient()
    raw_payload = await client.complete_json(
        system_prompt=CARE_ROUTER_SYSTEM,
        user_message=_build_router_user_prompt(classification, knowledge),
    )
    recommendation_data = {
        "pathway": raw_payload.get("pathway", "doctor_soon"),
        "pathway_label": raw_payload.get("pathway_label", "Doctor soon"),
        "urgency_window": raw_payload.get("urgency_window", "Within 1 week"),
        "reasoning": raw_payload.get("reasoning", "Prompt medical follow-up is recommended."),
        "immediate_actions": raw_payload.get("immediate_actions", []),
        "warning_signs": raw_payload.get("warning_signs", []),
        "self_care_steps": raw_payload.get("self_care_steps"),
    }
    return CareRecommendation.model_validate(recommendation_data)


async def synthesize_response(
    symptom_input: SymptomInput,
    classification: ClassificationResult,
    knowledge: KnowledgeResult,
    recommendation: CareRecommendation,
    session_id: str,
    start_time: float,
) -> TriageResponse:
    """Build the final structured triage response."""

    client = ClaudeClient()
    try:
        full_explanation = await client.complete_text(
            system_prompt=RESPONSE_SYNTHESIS_SYSTEM,
            user_message=_build_synthesis_user_message(
                symptom_input,
                classification,
                knowledge,
                recommendation,
            ),
        )
    except ClaudeClientError:
        logger.exception("Claude synthesis failed; falling back to deterministic explanation.")
        full_explanation = (
            "### What we found\n"
            f"Your symptoms may be consistent with {', '.join(classification.symptom_clusters or ['a medical issue that needs review'])}.\n\n"
            "### What this might mean\n"
            "The most likely causes depend on how your symptoms evolve and whether warning signs appear.\n\n"
            "### What you should do\n"
            f"{recommendation.reasoning}\n\n"
            "### Watch for these warning signs\n"
            + "\n".join(f"- {item}" for item in recommendation.warning_signs)
            + "\n\n### Important reminder\n"
            + CANONICAL_DISCLAIMER
        )

    top_differentials = _sanitize_differentials_for_response(
        knowledge.differentials,
        classification.is_emergency,
    )

    return TriageResponse.model_validate(
        {
            "session_id": session_id,
            "urgency_level": classification.severity_score,
            "urgency_label": SEVERITY_LABELS[classification.severity_score],
            "care_recommendation": recommendation,
            "top_differentials": top_differentials,
            "immediate_message": _build_immediate_message(classification, recommendation),
            "full_explanation": full_explanation,
            "disclaimer": CANONICAL_DISCLAIMER,
            "is_emergency": classification.is_emergency,
            "processing_time_ms": _compute_processing_time_ms(start_time),
            "safety_flags": [],
        }
    )


def format_chat_response(response: TriageResponse) -> str:
    """Convert a structured response into readable markdown for Chat Protocol."""

    sections: list[str] = []
    if response.urgency_level >= 4:
        sections.append(
            f"🚨🚨🚨 EMERGENCY — {response.care_recommendation.pathway_label} 🚨🚨🚨"
        )

    sections.append(
        f"Urgency Level: {response.urgency_label} | {response.care_recommendation.urgency_window}"
    )
    sections.append(response.immediate_message)
    sections.append(response.full_explanation)

    disclaimer_text = response.disclaimer
    if disclaimer_text.startswith("⚠️ "):
        sections.append(disclaimer_text)
    else:
        sections.append(f"⚠️ {disclaimer_text}")

    return "\n\n".join(section.strip() for section in sections if section and section.strip())


async def _maybe_request_payment(
    ctx: Context,
    sender: str,
    response: TriageResponse,
    formatted_report: str,
) -> bool:
    """Optionally gate the full report behind the payment protocol."""

    if not ENABLE_PAYMENT_GATE or not PAYMENT_PROTOCOL_AVAILABLE or payment_protocol is None:
        return False

    if not sender.startswith("agent"):
        ctx.logger.warning("Payment gating enabled but sender is not an agent address.")
        return False

    reference = str(uuid4())
    _pending_paid_reports[reference] = {
        "sender": sender,
        "report": formatted_report,
        "response": response.model_dump(),
    }

    await ctx.send(
        sender,
        RequestPayment(
            accepted_funds=[
                Funds(
                    amount=PAYMENT_AMOUNT,
                    currency=PAYMENT_CURRENCY,
                    payment_method=PAYMENT_METHOD,
                )
            ],
            recipient=ctx.agent.address,
            deadline_seconds=PAYMENT_DEADLINE_SECONDS,
            reference=reference,
            description="Medical triage full report",
            metadata={"session_id": response.session_id},
        ),
    )
    await ctx.send(
        sender,
        _build_text_message(
            "A payment request has been sent for the full triage report. Once payment is "
            "committed, the detailed markdown response will be delivered automatically."
        ),
    )
    return True


async def process_triage_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """Run the triage workflow after chat acknowledgement has been sent."""

    session_id = str(uuid4())
    start_time = time.time()

    raw_text = _extract_chat_text(msg)
    if not raw_text:
        await ctx.send(
            sender,
            _build_text_message(
                "Please describe your symptoms in a sentence or two so I can assess urgency and next steps."
            ),
        )
        await log_session(
            session_id,
            "EMPTY_INPUT",
            raw_input="",
            fast_path_used=False,
            safety_flags_triggered=[],
        )
        return

    try:
        safety_result = detect_emergency(raw_text)
        if safety_result.requires_911 and safety_result.override_response:
            await ctx.send(sender, _build_text_message(safety_result.override_response))
            await log_session(
                session_id,
                "FAST_PATH_EMERGENCY",
                raw_text,
                fast_path_used=True,
                safety_flags_triggered=safety_result.triggered_rules,
            )
            return

        symptom_input = SymptomInput(
            raw_text=raw_text,
            session_id=session_id,
            timestamp=datetime.utcnow(),
        )

        classification, knowledge = await asyncio.gather(
            call_classifier_agent(symptom_input),
            call_knowledge_agent(symptom_input),
        )
        recommendation = await call_router_agent(classification, knowledge)

        triage_response = await synthesize_response(
            symptom_input=symptom_input,
            classification=classification,
            knowledge=knowledge,
            recommendation=recommendation,
            session_id=session_id,
            start_time=start_time,
        )
        safe_response = apply_guardrails(triage_response, safety_result.triggered_rules)

        formatted = format_chat_response(safe_response)
        payment_requested = await _maybe_request_payment(
            ctx=ctx,
            sender=sender,
            response=safe_response,
            formatted_report=formatted,
        )
        if not payment_requested:
            await ctx.send(sender, _build_text_message(formatted))

        await log_session(
            session_id,
            "COMPLETED",
            raw_text,
            response=safe_response,
            fast_path_used=False,
            safety_flags_triggered=safe_response.safety_flags,
        )
    except Exception:
        ctx.logger.exception("Unhandled error while processing triage request")
        fallback = (
            "I hit a problem while processing your symptoms. If you have chest pain, trouble "
            "breathing, stroke-like symptoms, severe bleeding, or you feel unsafe right now, "
            "call 911 immediately. Otherwise, please try again with a brief description of your symptoms."
        )
        await ctx.send(sender, _build_text_message(fallback))
        await log_session(
            session_id,
            "ERROR",
            raw_text,
            fast_path_used=False,
            safety_flags_triggered=[],
        )


@chat_protocol.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """Primary Chat Protocol entrypoint for ASI:One and compatible agents."""

    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )

    if _is_session_start_message(msg):
        await ctx.send(sender, _build_session_ready_message())
        return

    await process_triage_message(ctx, sender, msg)


@chat_protocol.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
    """Accept chat acknowledgements for protocol completeness."""

    ctx.logger.debug("Received chat acknowledgement from %s for %s", sender, msg.acknowledged_msg_id)


if PAYMENT_PROTOCOL_AVAILABLE and payment_protocol is not None:

    @payment_protocol.on_message(CommitPayment)
    async def handle_payment_commit(ctx: Context, sender: str, msg: CommitPayment) -> None:
        """Deliver the paid report once the buyer commits payment."""

        reference = getattr(msg, "reference", None)
        pending = _pending_paid_reports.get(reference or "")
        if not pending:
            ctx.logger.warning("Received CommitPayment with unknown reference: %s", reference)
            return

        await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
        await ctx.send(sender, _build_text_message(pending["report"]))
        _pending_paid_reports.pop(reference or "", None)

    @payment_protocol.on_message(RejectPayment)
    async def handle_payment_reject(ctx: Context, sender: str, msg: RejectPayment) -> None:
        """Handle rejected payment requests gracefully."""

        reason = getattr(msg, "reason", None) or "payment request declined"
        for reference, pending in list(_pending_paid_reports.items()):
            if pending.get("sender") == sender:
                _pending_paid_reports.pop(reference, None)
        ctx.logger.info("Payment rejected by %s: %s", sender, reason)
        await ctx.send(
            sender,
            _build_text_message(
                "The full triage report was not delivered because the payment request was declined."
            ),
        )


@agent.on_event("startup")
async def on_startup(ctx: Context) -> None:
    """Initialize local resources required by the orchestrator."""

    await initialize_database()
    try:
        await asyncio.to_thread(fund_agent_if_low, agent.wallet.address())
    except Exception:
        ctx.logger.exception("Unable to fund orchestrator wallet automatically.")
    await register_agentverse_listing(ctx)
    ctx.logger.info("Medical triage orchestrator ready at %s", agent.address)


agent.include(chat_protocol, publish_manifest=True)
if payment_protocol is not None:
    agent.include(payment_protocol, publish_manifest=True)


if __name__ == "__main__":
    agent.run()
