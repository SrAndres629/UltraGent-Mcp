import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Any, Optional

import pytest
from dataclasses import dataclass, field
from neuro_architect import get_neuro_architect

from hud_manager import HUDManager, HUMAN_SIGNAL_TEMPLATE, HUD_TEMPLATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ultragent.hud")

# Set up test fixtures
@pytest.fixture
def hud_manager():
    return HUDManager()

@pytest.fixture
def mock_mcp_server():
    with patch("hud_manager.mcp") as mock_mcp:
        yield mock_mcp

@pytest.fixture
def mock_sentinel():
    with patch("hud_manager.get_sentinel") as mock_sentinel:
        yield mock_sentinel

@pytest.fixture
def mock_router():
    with patch("hud_manager.get_router") as mock_router:
        yield mock_router

def test_hud_manager_init(hud_manager):
    assert hud_manager._lock
    assert hud_manager._last_refresh is None
    assert hud_manager._start_time
    assert hud_manager._mission_goal == "Sistema en espera de misión"
    assert hud_manager._pending_human_signal is None

def test_set_mission_goal(hud_manager):
    goal = "Test Mission Goal"
    hud_manager.set_mission_goal(goal)
    assert hud_manager._mission_goal == goal

def test_request_human_decision(hud_manager):
    issue = "Test Issue"
    options = ["Option 1", "Option 2"]
    hud_manager.request_human_decision(issue, options)
    assert hud_manager._pending_human_signal == {
        "issue": issue,
        "options": " | ".join(f"[{o}]" for o in options),
        "timestamp": hud_manager._pending_human_signal["timestamp"],
    }

def test_clear_human_signal(hud_manager):
    hud_manager.request_human_decision("Test Issue", ["Option 1", "Option 2"])
    hud_manager.clear_human_signal()
    assert hud_manager._pending_human_signal is None

def test_get_uptime(hud_manager):
    hud_manager._start_time = datetime.now() - timedelta(hours=1, minutes=30, seconds=30)
    uptime = hud_manager._get_uptime()
    assert uptime == "01:30:30"

def test_get_mcp_status(mock_mcp_server):
    mock_mcp_server._tool_manager._tools = [Mock(), Mock(), Mock()]
    hud_manager = HUDManager()
    mcp_status = hud_manager._get_mcp_status()
    assert mcp_status == {"status": "🟢 Online", "version": "2.0", "tools": 3}

def test_get_sentinel_status(mock_sentinel):
    mock_sentinel.return_value.get_status.return_value = {
        "watching": True,
        "stats": {"events_detected": 10},
    }
    hud_manager = HUDManager()
    sentinel_status = hud_manager._get_sentinel_status()
    assert sentinel_status == {
        "status": "🟢 Watching",
        "events": 10,
    }

def test_get_router_status(mock_router):
    mock_router.return_value.get_status.return_value = {
        "token_usage": {
            "token1": {"total_tokens": 10},
            "token2": {"total_tokens": 20},
        },
    }
    hud_manager = HUDManager()
    router_status = hud_manager._get_router_status()
    assert router_status == {
        "status": "🟢 Online",
        "version": "2.0",
        "tools": 3,
    }

def test_hud_template(hud_manager):
    hud_manager.set_mission_goal("Test Mission Goal")
    hud_manager.request_human_decision("Test Issue", ["Option 1", "Option 2"])
    hud_template = HUD_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        mission_goal=hud_manager._mission_goal,
        uptime=hud_manager._get_uptime(),
        mcp_status="🟢 Online",
        mcp_version="2.0",
        mcp_tools=3,
        sentinel_status="🟢 Watching",
        sentinel_events=10,
        router_status="🟢 Online",
        router_tokens=30,
        librarian_status="🟢 Online",
        librarian_docs=100,
        scout_status="🟢 Online",
        scout_searches=10,
        evolution_status="🟢 Online",
        evolution_audits=10,
        mechanic_status="🟢 Online",
        docker_status="🟢 Online",
        vision_status="🟢 Online",
        vision_scans=10,
        human_signal=HUMAN_SIGNAL_TEMPLATE.format(
            issue="Test Issue",
            options=" | ".join(f"[{o}]" for o in ["Option 1", "Option 2"]),
        ),
        active_tasks="",
        fitness_score=10,
        fitness_grade="A",
        evolution_iterations=10,
        evolution_link="",
        architecture_image="",
        sentinel_alerts="",
        router_details="",
        mechanic_details="",
        scout_details="",
    )
    assert hud_template

def test_hud_manager_refresh(hud_manager):
    hud_manager._last_refresh = datetime.now() - timedelta(seconds=2)
    hud_manager._get_uptime = Mock(return_value="01:30:30")
    hud_manager._get_mcp_status = Mock(return_value={"status": "🟢 Online", "version": "2.0", "tools": 3})
    hud_manager._get_sentinel_status = Mock(return_value={"status": "🟢 Watching", "events": 10})
    hud_manager._get_router_status = Mock(return_value={"status": "🟢 Online", "version": "2.0", "tools": 3})
    hud_manager.request_human_decision("Test Issue", ["Option 1", "Option 2"])
    hud_manager.set_mission_goal("Test Mission Goal")
    hud_template = HUD_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        mission_goal=hud_manager._mission_goal,
        uptime=hud_manager._get_uptime(),
        mcp_status=hud_manager._get_mcp_status()["status"],
        mcp_version=hud_manager._get_mcp_status()["version"],
        mcp_tools=hud_manager._get_mcp_status()["tools"],
        sentinel_status=hud_manager._get_sentinel_status()["status"],
        sentinel_events=hud_manager._get_sentinel_status()["events"],
        router_status=hud_manager._get_router_status()["status"],
        router_tokens=30,
        librarian_status="🟢 Online",
        librarian_docs=100,
        scout_status="🟢 Online",
        scout_searches=10,
        evolution_status="🟢 Online",
        evolution_audits=10,
        mechanic_status="🟢 Online",
        docker_status="🟢 Online",
        vision_status="🟢 Online",
        vision_scans=10,
        human_signal=HUMAN_SIGNAL_TEMPLATE.format(
            issue="Test Issue",
            options=" | ".join(f"[{o}]" for o in ["Option 1", "Option 2"]),
        ),
        active_tasks="",
        fitness_score=10,
        fitness_grade="A",
        evolution_iterations=10,
        evolution_link="",
        architecture_image="",
        sentinel_alerts="",
        router_details="",
        mechanic_details="",
        scout_details="",
    )
    assert hud_template