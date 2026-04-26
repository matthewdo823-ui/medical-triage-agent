"""Startup safety checks for the orchestrator entrypoint."""

from __future__ import annotations

import socket

import pytest

from agents import orchestrator


def test_port_is_available_for_unused_ephemeral_port() -> None:
    """A temporary free port should be reported as available."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    assert orchestrator._port_is_available(port) is True


def test_port_is_unavailable_when_already_bound() -> None:
    """A live listener should cause the preflight port check to fail."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]

        assert orchestrator._port_is_available(port) is False


def test_startup_port_check_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup should stop with a concise message before uagents spins up tasks."""

    monkeypatch.setattr(orchestrator, "AGENT_PORT", 8123)
    monkeypatch.setattr(orchestrator, "_port_is_available", lambda port: False)

    with pytest.raises(SystemExit) as exc_info:
        orchestrator._ensure_startup_port_available()

    assert exc_info.value.code is not None
    assert "port 8123 is already in use" in str(exc_info.value)
    assert "ORCHESTRATOR_PORT=<port>" in str(exc_info.value)
