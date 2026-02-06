import pytest
from unittest.mock import Mock, patch
from mcp_server import (
    with_timeout,
    get_db_connection,
    setup_logging,
    SecretFilter,
    FastMCP,
    DEFAULT_TOOL_TIMEOUT,
)

@pytest.fixture
def logger():
    return setup_logging()

@pytest.fixture
def db_connection():
    return get_db_connection()

@pytest.fixture
def mcp_server():
    return FastMCP(
        name="ultragent",
        instructions="Ultragent MCP Server - Sistema de Ingeniería Híbrida Autónoma.",
    )

def test_with_timeout_success():
    async def coro():
        return "Success"

    result = asyncio.run(with_timeout(coro()))
    assert result["success"]
    assert result["result"] == "Success"

def test_with_timeout_failure():
    async def coro():
        raise Exception("Test error")

    result = asyncio.run(with_timeout(coro()))
    assert not result["success"]
    assert result["error"] == "Test error"

def test_with_timeout_timeout():
    async def coro():
        await asyncio.sleep(DEFAULT_TOOL_TIMEOUT + 1)

    result = asyncio.run(with_timeout(coro()))
    assert not result["success"]
    assert result["error_type"] == "TIMEOUT"

def test_get_db_connection(db_connection):
    assert db_connection is not None
    assert db_connection.execute("SELECT 1") is not None

def test_setup_logging(logger):
    assert logger is not None
    assert logger.level is not None

def test_secret_filter():
    filter = SecretFilter()
    record = logging.LogRecord("test", logging.INFO, "test", 1, "sk-1234567890", None, None)
    filter.filter(record)
    assert record.msg == "[REDACTED]"

@patch("sqlite3.connect")
def test_get_db_connection_mock(mock_connect):
    get_db_connection()
    mock_connect.assert_called_once()

@patch("logging.Formatter")
def test_setup_logging_mock(mock_formatter):
    setup_logging()
    mock_formatter.assert_called_once()

def test_mcp_server(mcp_server):
    assert mcp_server.name == "ultragent"
    assert mcp_server.instructions is not None

@patch("fastmcp.FastMCP")
def test_mcp_server_mock(mock_fastmcp):
    FastMCP(
        name="ultragent",
        instructions="Ultragent MCP Server - Sistema de Ingeniería Híbrida Autónoma.",
    )
    mock_fastmcp.assert_called_once()