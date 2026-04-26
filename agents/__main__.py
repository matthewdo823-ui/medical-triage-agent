"""Helpful package entrypoint for the triage agent suite."""

from __future__ import annotations


def main() -> None:
    """Explain which agent module to run explicitly."""

    raise SystemExit(
        "Choose a concrete agent module to run.\n"
        "Examples:\n"
        "  python -m agents.orchestrator\n"
        "  python -m agents.symptom_classifier\n"
        "  python -m agents.knowledge_retrieval\n"
        "  python -m agents.care_router"
    )


if __name__ == "__main__":
    main()
