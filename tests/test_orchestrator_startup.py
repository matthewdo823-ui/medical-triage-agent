"""Startup safety checks for the orchestrator entrypoint."""

from __future__ import annotations

import errno
from unittest.mock import MagicMock, Mock

import pytest

from agents import orchestrator


def test_port_is_available_for_unused_ephemeral_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful bind attempt should be reported as available."""

    fake_socket = Mock()
    socket_factory = MagicMock()
    socket_factory.return_value.__enter__.return_value = fake_socket
    socket_factory.return_value.__exit__.return_value = None
    monkeypatch.setattr(orchestrator.socket, "socket", socket_factory)

    assert orchestrator._port_is_available(8123) is True
    fake_socket.bind.assert_called_once_with(("0.0.0.0", 8123))


def test_port_is_unavailable_when_already_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """An address-in-use bind failure should be reported as unavailable."""

    fake_socket = Mock()
    fake_socket.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
    socket_factory = MagicMock()
    socket_factory.return_value.__enter__.return_value = fake_socket
    socket_factory.return_value.__exit__.return_value = None
    monkeypatch.setattr(orchestrator.socket, "socket", socket_factory)

    assert orchestrator._port_is_available(8123) is False


def test_startup_port_check_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup should stop with a concise message before uagents spins up tasks."""

    monkeypatch.setattr(orchestrator, "AGENT_PORT", 8123)
    monkeypatch.setattr(orchestrator, "_port_is_available", lambda port: False)

    with pytest.raises(SystemExit) as exc_info:
        orchestrator._ensure_startup_port_available()

    assert exc_info.value.code is not None
    assert "port 8123 is already in use" in str(exc_info.value)
    assert "ORCHESTRATOR_PORT=<port>" in str(exc_info.value)
