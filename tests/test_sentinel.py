import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from sentinel import (
    SentinelEvent,
    UltragentEventHandler,
    DEBOUNCE_SECONDS,
    EXCLUDED_PATTERNS,
    MONITORED_EXTENSIONS,
)
from pathlib import Path

@pytest.fixture
def sentinel_event():
    return SentinelEvent("created", "/path/to/file.txt")

def test_sentinel_event_to_dict(sentinel_event):
    expected_dict = {
        "event_type": "created",
        "src_path": "/path/to/file.txt",
        "timestamp": sentinel_event.timestamp.isoformat(),
        "source": "unknown",
        "processed": False,
    }
    assert sentinel_event.to_dict() == expected_dict

def test_sentinel_event_to_hud_line(sentinel_event):
    expected_hud_line = "| 📄 CREATED | `file.txt` | {} | unknown |".format(
        sentinel_event.timestamp.strftime("%H:%M:%S")
    )
    assert sentinel_event.to_hud_line() == expected_hud_line

@pytest.fixture
def ultragent_event_handler():
    on_stable_event = Mock()
    return UltragentEventHandler(on_stable_event)

def test_ultragent_event_handler_init(ultragent_event_handler):
    assert ultragent_event_handler._on_stable_event is not None
    assert ultragent_event_handler._debounce_seconds == DEBOUNCE_SECONDS

def test_ultragent_event_handler_should_ignore(ultragent_event_handler):
    assert ultragent_event_handler._should_ignore("/path/to/.gitignore") is True
    assert ultragent_event_handler._should_ignore("/path/to/file.py") is False

def test_ultragent_event_handler_detect_source(ultragent_event_handler):
    with patch("sentinel.open", Mock(side_effect=IOError("Permission denied"))):
        assert ultragent_event_handler._detect_source("/path/to/file.txt") == "unknown"

    with patch("sentinel.open", return_value=Mock()) as mock_open:
        mock_file = mock_open.return_value.__enter__.return_value
        mock_file.read.return_value = "[ULTRAGENT:MODIFIED]"
        assert ultragent_event_handler._detect_source("/path/to/file.txt") == "ultragent"

def test_ultragent_event_handler_on_created(ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    ultragent_event_handler.on_created(event)
    assert ultragent_event_handler._pending_timers.get("/path/to/file.txt") is not None

def test_ultragent_event_handler_on_modified(ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    ultragent_event_handler.on_modified(event)
    assert ultragent_event_handler._pending_timers.get("/path/to/file.txt") is not None

def test_ultragent_event_handler_on_deleted(ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    ultragent_event_handler.on_deleted(event)
    assert ultragent_event_handler._on_stable_event.call_count == 1

def test_ultragent_event_handler_on_moved(ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    event.dest_path = "/path/to/new_file.txt"
    ultragent_event_handler.on_moved(event)
    assert ultragent_event_handler._on_stable_event.call_count == 1

@patch("sentinel.Timer")
def test_ultragent_event_handler_debounce(mock_timer, ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    ultragent_event_handler.on_created(event)
    assert mock_timer.call_count == 1
    assert mock_timer.return_value.start.call_count == 1
    assert ultragent_event_handler._pending_timers.get("/path/to/file.txt") is not None

@patch("sentinel.Timer")
def test_ultragent_event_handler_debounce_cancel(mock_timer, ultragent_event_handler):
    event = Mock()
    event.src_path = "/path/to/file.txt"
    ultragent_event_handler.on_created(event)
    assert mock_timer.call_count == 1
    assert mock_timer.return_value.start.call_count == 1
    assert ultragent_event_handler._pending_timers.get("/path/to/file.txt") is not None

    ultragent_event_handler.on_created(event)
    assert mock_timer.return_value.cancel.call_count == 1

def test_ultragent_event_handler_on_stable_event(ultragent_event_handler):
    event = SentinelEvent("created", "/path/to/file.txt")
    ultragent_event_handler._on_stable_event(event)
    assert ultragent_event_handler._on_stable_event.call_count == 1