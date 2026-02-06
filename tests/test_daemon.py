# tests/test_daemon.py
import pytest
import tempfile
import time
from unittest.mock import patch, Mock
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from daemon import QualityGuardian, start_daemon, PROJECT_ROOT, IGNORE_DIRS, WATCH_EXTENSIONS
from metrics import MetricsEngine

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def temp_file(temp_dir):
    file_path = temp_dir / "test.py"
    file_path.write_text("print('Hello World')")
    yield file_path

@pytest.fixture
def quality_guardian():
    yield QualityGuardian()

def test_quality_guardian_init():
    quality_guardian = QualityGuardian()
    assert isinstance(quality_guardian, FileSystemEventHandler)

def test_quality_guardian_on_modified(quality_guardian, temp_file):
    event = Mock(src_path=str(temp_file))
    quality_guardian.on_modified(event)
    assert temp_file.name in quality_guardian._audit_file.call_args[0][0].name

def test_quality_guardian_on_modified_with_directory(quality_guardian, temp_dir):
    event = Mock(src_path=str(temp_dir), is_directory=True)
    quality_guardian.on_modified(event)
    assert quality_guardian._audit_file.call_count == 0

def test_quality_guardian_on_modified_with_ignored_dir(quality_guardian, temp_dir):
    event = Mock(src_path=str(temp_dir / ".git"), is_directory=False)
    quality_guardian.on_modified(event)
    assert quality_guardian._audit_file.call_count == 0

def test_quality_guardian_on_modified_with_ignored_extension(quality_guardian, temp_dir):
    event = Mock(src_path=str(temp_dir / "test.txt"), is_directory=False)
    quality_guardian.on_modified(event)
    assert quality_guardian._audit_file.call_count == 0

def test_quality_guardian_audit_file(quality_guardian, temp_file):
    quality_guardian._audit_file(temp_file)
    assert quality_guardian.logger.info.call_count == 1

def test_quality_guardian_audit_file_with_metrics_engine_error(quality_guardian, temp_file):
    with patch.object(MetricsEngine, "analyze_code", side_effect=Exception("Test Error")):
        quality_guardian._audit_file(temp_file)
        assert quality_guardian.logger.error.call_count == 1

def test_start_daemon():
    with patch.object(Observer, "start") as mock_start:
        start_daemon()
        mock_start.assert_called_once()

def test_start_daemon_with_keyboard_interrupt():
    with patch.object(Observer, "stop") as mock_stop:
        with patch.object(time, "sleep", side_effect=KeyboardInterrupt):
            start_daemon()
            mock_stop.assert_called_once()

def test_quality_guardian_with_metrics_engine_grade_d(quality_guardian, temp_file):
    with patch.object(MetricsEngine, "analyze_code", return_value=Mock(grade="D")):
        quality_guardian._audit_file(temp_file)
        assert quality_guardian.logger.warning.call_count == 2

def test_quality_guardian_with_metrics_engine_grade_f(quality_guardian, temp_file):
    with patch.object(MetricsEngine, "analyze_code", return_value=Mock(grade="F")):
        quality_guardian._audit_file(temp_file)
        assert quality_guardian.logger.warning.call_count == 2

def test_quality_guardian_with_metrics_engine_grade_other(quality_guardian, temp_file):
    with patch.object(MetricsEngine, "analyze_code", return_value=Mock(grade="A")):
        quality_guardian._audit_file(temp_file)
        assert quality_guardian.logger.info.call_count == 1