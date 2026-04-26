"""Shared LLM client wrappers for Anthropic and Google GenAI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types

from safety.guardrails import CANONICAL_DISCLAIMER

logger = logging.getLogger(__name__)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMMA_MODEL = "gemma-4-27b-it"
DEFAULT_MAX_TOKENS = 1500
JSON_RETRY_SUFFIX = "Output ONLY valid JSON. No preamble, no markdown."

RESPONSE_SYNTHESIS_SYSTEM = f"""
You are the final step in a medical triage system. You receive structured triage data and write a clear, compassionate, plain-English explanation for the patient.

## Tone and style:
- Warm, calm, non-alarmist (unless emergency — then clear and direct)
- Plain English, no medical jargon without explanation
- Empathetic but not dismissive
- Actionable — the patient should know exactly what to do next

## Format your response as markdown with these sections:
### What we found
[2-3 sentences summarizing the symptom pattern]

### What this might mean
[Brief explanation of top 2-3 differentials in plain language]

### What you should do
[Clear action steps based on care pathway]

### Watch for these warning signs
[Bulleted list of escalation symptoms]

### Important reminder
{CANONICAL_DISCLAIMER}

## Rules:
- Never use definitive diagnostic language ("you have X") — use hedged language ("your symptoms may be consistent with X")
- Never recommend specific prescription medications
- Never suggest delaying emergency care to "wait and see"
- If severity is 4-5, the "What you should do" section must lead with the emergency action in bold
""".strip()


class ClaudeClientError(Exception):
    """Raised when the Claude client cannot produce a response."""


class GemmaClientError(Exception):
    """Base exception for Gemma client failures."""


class GemmaJSONParseError(GemmaClientError):
    """Raised when Gemma returns invalid or non-dict JSON."""


class GemmaAPIError(GemmaClientError):
    """Raised when the Google GenAI API call fails."""


class ClaudeClient:
    """Thin async wrapper around the Anthropic Messages API."""

    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeClientError("ANTHROPIC_API_KEY is not set.")

        self.api_key = api_key
        self.model = DEFAULT_CLAUDE_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.client = anthropic.Anthropic(api_key=api_key)

    async def complete_text(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.4,
    ) -> str:
        """Generate a free-text markdown response with Claude Haiku."""

        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=self.max_tokens,
                temperature=temperature,
            )
            text = self._extract_text(response)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "Claude completion succeeded in %sms using model=%s",
                elapsed_ms,
                self.model,
            )
            return text
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "Claude completion failed in %sms using model=%s",
                elapsed_ms,
                self.model,
            )
            raise ClaudeClientError(f"Claude completion failed: {exc}") from exc

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Collect all text blocks from an Anthropic response."""

        content = getattr(response, "content", None) or []
        chunks: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                chunks.append(block.text)

        text = "".join(chunks).strip()
        if not text:
            raise ClaudeClientError("Claude returned an empty response.")
        return text


class GemmaClient:
    """Async wrapper around the Google GenAI SDK for JSON-first calls."""

    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise GemmaClientError("GOOGLE_API_KEY is not set.")

        self.api_key = api_key
        self.model = DEFAULT_GEMMA_MODEL
        self.max_output_tokens = DEFAULT_MAX_TOKENS
        self.client = genai.Client(api_key=api_key)

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Generate structured JSON with Gemma and retry on parse failures."""

        if max_retries < 1:
            raise GemmaClientError("max_retries must be at least 1.")

        retry_message = user_message
        last_error: Exception | None = None
        last_raw_response = ""

        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=retry_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                raw_text = (getattr(response, "text", "") or "").strip()
                last_raw_response = raw_text
                parsed = self._parse_json_response(raw_text)
                logger.info(
                    "Gemma JSON completion succeeded in %sms on attempt=%s using model=%s",
                    elapsed_ms,
                    attempt,
                    self.model,
                )
                return parsed
            except GemmaJSONParseError as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                last_error = exc
                logger.warning(
                    "Gemma JSON parse failure in %sms on attempt=%s using model=%s. Raw response=%r",
                    elapsed_ms,
                    attempt,
                    self.model,
                    last_raw_response,
                )
                if attempt >= max_retries:
                    raise GemmaJSONParseError(
                        f"Gemma returned invalid JSON after {max_retries} attempts."
                    ) from exc
                retry_message = f"{user_message}\n\n{JSON_RETRY_SUFFIX}"
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.exception(
                    "Gemma API failure in %sms on attempt=%s using model=%s. Raw response=%r",
                    elapsed_ms,
                    attempt,
                    self.model,
                    last_raw_response,
                )
                raise GemmaAPIError(f"Gemma API call failed: {exc}") from exc

        raise GemmaClientError(
            f"Gemma completion failed after {max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any]:
        """Parse a JSON response and enforce a top-level object."""

        if not raw_text:
            raise GemmaJSONParseError("Gemma returned an empty response.")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise GemmaJSONParseError(f"Invalid JSON response: {exc}") from exc

        if not isinstance(parsed, dict):
            raise GemmaJSONParseError("Gemma JSON response must be a top-level object.")
        return parsed


__all__ = [
    "CANONICAL_DISCLAIMER",
    "ClaudeClient",
    "ClaudeClientError",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_GEMMA_MODEL",
    "GemmaAPIError",
    "GemmaClient",
    "GemmaClientError",
    "GemmaJSONParseError",
    "RESPONSE_SYNTHESIS_SYSTEM",
]
