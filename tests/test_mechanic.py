import pytest
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
from typing import Dict, Any
from mechanic import LocalExecutor, LLMProvider, GeminiProvider

@pytest.fixture
def local_executor():
    return LocalExecutor()

@pytest.fixture
def gemini_provider():
    return GeminiProvider()

def test_local_executor_run_command_success(local_executor):
    command = "echo 'Hello World'"
    result = local_executor.run_command(command)
    assert result["exit_code"] == 0
    assert result["success"] == True
    assert result["stdout"] == "Hello World\n"

def test_local_executor_run_command_failure(local_executor):
    command = "invalid_command"
    result = local_executor.run_command(command)
    assert result["exit_code"] != 0
    assert result["success"] == False

def test_local_executor_run_command_timeout(local_executor):
    command = "sleep 10"
    result = local_executor.run_command(command, timeout=1)
    assert result["exit_code"] == -1
    assert result["success"] == False
    assert result["stderr"] == "Timeout expired"

def test_local_executor_write_file_success(local_executor):
    path = "test_file.txt"
    content = "Hello World"
    result = local_executor.write_file(path, content)
    assert result == True
    assert Path(path).exists()
    assert Path(path).read_text() == content

def test_local_executor_write_file_failure(local_executor):
    path = "/invalid/path/test_file.txt"
    content = "Hello World"
    result = local_executor.write_file(path, content)
    assert result == False

def test_local_executor_read_file_success(local_executor):
    path = "test_file.txt"
    content = "Hello World"
    Path(path).write_text(content)
    result = local_executor.read_file(path)
    assert result == content

def test_local_executor_read_file_failure(local_executor):
    path = "/invalid/path/test_file.txt"
    result = local_executor.read_file(path)
    assert result.startswith("Error: File")

def test_local_executor_edit_file_success(local_executor):
    path = "test_file.txt"
    content = "Hello World"
    Path(path).write_text(content)
    target = "World"
    replacement = "Universe"
    result = local_executor.edit_file(path, target, replacement)
    assert result["success"] == True
    assert Path(path).read_text() == "Hello Universe"

def test_local_executor_edit_file_failure(local_executor):
    path = "/invalid/path/test_file.txt"
    target = "World"
    replacement = "Universe"
    result = local_executor.edit_file(path, target, replacement)
    assert result["success"] == False

def test_local_executor_append_to_file_success(local_executor):
    path = "test_file.txt"
    content = "Hello World"
    Path(path).write_text(content)
    new_content = "This is new content"
    result = local_executor.append_to_file(path, new_content)
    assert result["success"] == True
    assert Path(path).read_text() == content + "\n" + new_content

def test_local_executor_append_to_file_failure(local_executor):
    path = "/invalid/path/test_file.txt"
    new_content = "This is new content"
    result = local_executor.append_to_file(path, new_content)
    assert result["success"] == False

@patch("mechanic.os")
def test_gemini_provider_init(mock_os):
    mock_os.getenv.return_value = "api_key"
    provider = GeminiProvider()
    assert provider.model is not None

@patch("mechanic.os")
def test_gemini_provider_init_failure(mock_os):
    mock_os.getenv.return_value = None
    with pytest.raises(ValueError):
        GeminiProvider()

@patch("mechanic.asyncio")
@patch("mechanic.google")
def test_gemini_provider_generate_success(mock_google, mock_asyncio):
    provider = GeminiProvider()
    mock_asyncio.to_thread.return_value = "response"
    result = provider.generate("prompt")
    assert result == "response"

@patch("mechanic.asyncio")
@patch("mechanic.google")
def test_gemini_provider_generate_failure(mock_google, mock_asyncio):
    provider = GeminiProvider()
    mock_asyncio.to_thread.side_effect = Exception("Error")
    with pytest.raises(Exception):
        provider.generate("prompt")

class TestLLMProvider(unittest.TestCase):
    def test_generate_not_implemented(self):
        provider = LLMProvider()
        with self.assertRaises(NotImplementedError):
            provider.generate("prompt")