"""Checks for Fetch.ai chat protocol compatibility on the orchestrator."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from agents import orchestrator
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)


def test_chat_protocol_uses_fetchai_spec() -> None:
    """The published protocol should match the current Fetch.ai chat spec."""

    assert orchestrator.protocol.name == chat_protocol_spec.name
    assert orchestrator.protocol.version == chat_protocol_spec.version

    manifest = orchestrator.protocol.manifest()
    interaction_map = {
        item["request"]: tuple(item["responses"])
        for item in manifest["interactions"]
    }

    chat_message_digest = next(
        model["digest"]
        for model in manifest["models"]
        if model["schema"]["title"] == "ChatMessage"
    )
    acknowledgement_digest = next(
        model["digest"]
        for model in manifest["models"]
        if model["schema"]["title"] == "ChatAcknowledgement"
    )

    assert interaction_map[chat_message_digest] == (acknowledgement_digest,)
    assert interaction_map[acknowledgement_digest] == ()


def test_start_session_messages_are_detected() -> None:
    """Session initialization envelopes should get the ready-message path."""

    message = ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[StartSessionContent(type="start-session")],
    )

    assert orchestrator._is_session_start_message(message) is True


def test_start_session_with_text_extracts_user_message() -> None:
    """Mixed envelopes should preserve symptom text instead of acting empty."""

    message = ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            StartSessionContent(type="start-session"),
            TextContent(type="text", text="i cant breathe"),
        ],
    )

    assert orchestrator._is_session_start_message(message) is True
    assert orchestrator._extract_text_content(message) == "i cant breathe"


def test_session_ready_message_contains_prompt_text() -> None:
    """The ready message should help ASI:One users start the conversation."""

    ready = orchestrator._build_session_ready_message()
    prompt_text = "\n".join(
        item.text
        for item in ready.content
        if hasattr(item, "text")
    )

    assert "describe your symptoms" in prompt_text.lower()
    assert all(not isinstance(item, ChatAcknowledgement) for item in ready.content)
