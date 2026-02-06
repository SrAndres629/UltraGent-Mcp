# tests/test_main.py
import os
import sys
import unittest
from unittest.mock import Mock, patch
from pytest import fixture
from main import (
    VERSION,
    BANNER,
    PROJECT_ROOT,
    AI_DIR,
    hud_refresher_thread,
    sentinel_thread,
    setup_logging,
    signal_handler,
    print_startup_info,
    validate_environment,
    shutdown_event,
)

@fixture
def mock_hud_manager():
    with patch("main.get_hud_manager") as mock_get_hud_manager:
        yield mock_get_hud_manager

@fixture
def mock_sentinel():
    with patch("main.get_sentinel") as mock_get_sentinel:
        yield mock_get_sentinel

def test_hud_refresher_thread(mock_hud_manager):
    # Arrange
    mock_hud = mock_hud_manager.return_value
    mock_hud.refresh_dashboard.side_effect = None

    # Act
    hud_refresher_thread(interval=1.0)

    # Assert
    mock_hud_manager.assert_called_once()
    mock_hud.refresh_dashboard.assert_called_once()

def test_hud_refresher_thread_error(mock_hud_manager):
    # Arrange
    mock_hud = mock_hud_manager.return_value
    mock_hud.refresh_dashboard.side_effect = Exception("Test error")

    # Act and Assert
    with patch("main.logger") as mock_logger:
        hud_refresher_thread(interval=1.0)
        mock_logger.error.assert_called_once()

def test_sentinel_thread(mock_sentinel):
    # Arrange
    mock_sentinel_instance = mock_sentinel.return_value

    # Act
    sentinel_thread()

    # Assert
    mock_sentinel.assert_called_once()
    mock_sentinel_instance.start.assert_called_once()
    mock_sentinel_instance.stop.assert_called_once()

def test_sentinel_thread_error(mock_sentinel):
    # Arrange
    mock_sentinel_instance = mock_sentinel.return_value
    mock_sentinel_instance.start.side_effect = Exception("Test error")

    # Act and Assert
    with patch("main.logger") as mock_logger:
        sentinel_thread()
        mock_logger.error.assert_called_once()

def test_setup_logging():
    # Arrange
    log_level = "INFO"

    # Act
    setup_logging(log_level)

    # Assert
    assert logging.getLogger("ultragent.main").level == logging.INFO

def test_signal_handler():
    # Arrange
    signum = 15  # SIGTERM
    frame = None

    # Act
    signal_handler(signum, frame)

    # Assert
    assert shutdown_event.is_set()

def test_print_startup_info(capsys):
    # Act
    print_startup_info()

    # Assert
    captured = capsys.readouterr()
    assert BANNER.format(version=VERSION) in captured.out

def test_validate_environment():
    # Arrange
    os.environ["KIMI_API_KEY"] = ""
    os.environ["GITHUB_TOKEN"] = ""

    # Act
    warnings = validate_environment()

    # Assert
    assert len(warnings) == 3

def test_validate_environment_with_api_keys():
    # Arrange
    os.environ["KIMI_API_KEY"] = "test_key"
    os.environ["GITHUB_TOKEN"] = "test_token"

    # Act
    warnings = validate_environment()

    # Assert
    assert len(warnings) == 1