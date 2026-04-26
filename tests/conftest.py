"""Shared pytest test configuration for the medical triage project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ORCHESTRATOR_SEED", "test-orchestrator-seed")
os.environ.setdefault("CLASSIFIER_SEED", "test-classifier-seed")
os.environ.setdefault("KNOWLEDGE_SEED", "test-knowledge-seed")
os.environ.setdefault("ROUTER_SEED", "test-router-seed")
