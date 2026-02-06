import asyncio
import json
import os
import pytest
from unittest.mock import Mock, patch
from probe import ProbeResult, probe_groq, probe_siliconflow, probe_github
from httpx import HTTPError

@pytest.fixture
def mock_env():
    with patch.dict('os.environ', {
        'GROQ_API_KEY': 'mock_api_key',
        'SILICONFLOW_API_KEY': 'mock_api_key',
        'GITHUB_TOKEN': 'mock_token'
    }):
        yield

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.AsyncClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_response():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={'choices': [{'message': {'content': 'mock_content'}}]})
    yield mock_response

def test_probe_result_init():
    result = ProbeResult('tier', 'provider', 'status')
    assert result.tier == 'tier'
    assert result.provider == 'provider'
    assert result.status == 'status'

@pytest.mark.asyncio
async def test_probe_groq(mock_env, mock_httpx_client, mock_response):
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response
    result = await probe_groq()
    assert result.tier == '⚡ Speed'
    assert result.provider == 'Groq'
    assert result.status == '✅ OK'

@pytest.mark.asyncio
async def test_probe_groq_no_api_key():
    with patch.dict('os.environ', {'GROQ_API_KEY': ''}):
        result = await probe_groq()
        assert result.tier == '⚡ Speed'
        assert result.provider == 'Groq'
        assert result.status == '⚠️ NO KEY'

@pytest.mark.asyncio
async def test_probe_groq_http_error(mock_env, mock_httpx_client):
    mock_httpx_client.return_value.__aenter__.return_value.post.side_effect = HTTPError('mock_error')
    result = await probe_groq()
    assert result.tier == '⚡ Speed'
    assert result.provider == 'Groq'
    assert result.status == '❌ FAIL'

@pytest.mark.asyncio
async def test_probe_siliconflow(mock_env, mock_httpx_client, mock_response):
    mock_httpx_client.return_value.__aenter__.return_value.post.return_value = mock_response
    result = await probe_siliconflow()
    assert result.tier == '🛠️ Coding'
    assert result.provider == 'SiliconFlow'
    assert result.status == '✅ OK'

@pytest.mark.asyncio
async def test_probe_siliconflow_no_api_key():
    with patch.dict('os.environ', {'SILICONFLOW_API_KEY': ''}):
        result = await probe_siliconflow()
        assert result.tier == '🛠️ Coding'
        assert result.provider == 'SiliconFlow'
        assert result.status == '⚠️ NO KEY'

@pytest.mark.asyncio
async def test_probe_siliconflow_http_error(mock_env, mock_httpx_client):
    mock_httpx_client.return_value.__aenter__.return_value.post.side_effect = HTTPError('mock_error')
    result = await probe_siliconflow()
    assert result.tier == '🛠️ Coding'
    assert result.provider == 'SiliconFlow'
    assert result.status == '❌ FAIL'

@pytest.mark.asyncio
async def test_probe_github(mock_env, mock_httpx_client, mock_response):
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={'items': []})
    mock_httpx_client.return_value.__aenter__.return_value.get.return_value = mock_response
    result = await probe_github()
    assert result.tier == '🕵️ Scout'
    assert result.provider == 'GitHub'
    assert result.status == '✅ OK'

@pytest.mark.asyncio
async def test_probe_github_no_token():
    with patch.dict('os.environ', {'GITHUB_TOKEN': ''}):
        result = await probe_github()
        assert result.tier == '🕵️ Scout'
        assert result.provider == 'GitHub'
        assert result.status == '⚠️ NO KEY'

@pytest.mark.asyncio
async def test_probe_github_http_error(mock_env, mock_httpx_client):
    mock_httpx_client.return_value.__aenter__.return_value.get.side_effect = HTTPError('mock_error')
    result = await probe_github()
    assert result.tier == '🕵️ Scout'
    assert result.provider == 'GitHub'
    assert result.status == '❌ FAIL'