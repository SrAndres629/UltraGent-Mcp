import pytest
from unittest.mock import Mock, patch
from router import (
    Tier,
    ProviderStatus,
    RouterResponse,
    ProviderState,
    BaseProvider,
    OpenAICompatibleProvider,
)

@pytest.fixture
def provider_state():
    return ProviderState(name="test_provider")

@pytest.fixture
def base_provider():
    return BaseProvider("test_provider", "api_key", "base_url", "model")

@pytest.fixture
def openai_compatible_provider():
    return OpenAICompatibleProvider("test_provider", "api_key", "base_url", "model")

def test_provider_state_init(provider_state):
    assert provider_state.name == "test_provider"
    assert provider_state.status == ProviderStatus.ACTIVE
    assert provider_state.failures == 0
    assert provider_state.last_failure is None
    assert provider_state.total_calls == 0
    assert provider_state.total_tokens == 0

def test_base_provider_init(base_provider):
    assert base_provider.name == "test_provider"
    assert base_provider.api_key == "api_key"
    assert base_provider.base_url == "base_url"
    assert base_provider.model == "model"
    assert isinstance(base_provider.state, ProviderState)

def test_base_provider_is_available(base_provider):
    assert base_provider.is_available() is True

def test_base_provider_is_available_inactive(base_provider):
    base_provider.state.status = ProviderStatus.INACTIVE
    assert base_provider.is_available() is False

def test_base_provider_mark_success(base_provider):
    base_provider.mark_success()
    assert base_provider.state.total_calls == 1

def test_base_provider_mark_failure(base_provider):
    base_provider.mark_failure()
    assert base_provider.state.failures == 1
    assert base_provider.state.last_failure is not None

def test_openai_compatible_provider_init(openai_compatible_provider):
    assert openai_compatible_provider.name == "test_provider"
    assert openai_compatible_provider.api_key == "api_key"
    assert openai_compatible_provider.base_url == "base_url"
    assert openai_compatible_provider.model == "model"
    assert isinstance(openai_compatible_provider.state, ProviderState)

@patch("httpx.post")
def test_openai_compatible_provider_complete(mock_post, openai_compatible_provider):
    mock_post.return_value = Mock(json=Mock(return_value={"choices": [{"text": "response"}]}))
    response = openai_compatible_provider.complete([{"role": "user", "content": "prompt"}])
    assert response[0] == "response"
    assert response[1] == 1

def test_router_response_init():
    response = RouterResponse(True, "content", "provider", Tier.SPEED, 100, 1000.0)
    assert response.success is True
    assert response.content == "content"
    assert response.provider == "provider"
    assert response.tier == Tier.SPEED
    assert response.tokens_used == 100
    assert response.latency_ms == 1000.0

def test_task_to_tier():
    assert TASK_TO_TIER["fix_syntax"] == Tier.SPEED
    assert TASK_TO_TIER["generate_code"] == Tier.CODING
    assert TASK_TO_TIER["analyze_image"] == Tier.VISUAL
    assert TASK_TO_TIER["architecture_review"] == Tier.STRATEGIC

def test_circuit_breaker_threshold():
    provider_state = ProviderState("test_provider")
    provider_state.failures = 3
    assert provider_state.status == ProviderStatus.INACTIVE

def test_circuit_breaker_reset():
    provider_state = ProviderState("test_provider")
    provider_state.status = ProviderStatus.INACTIVE
    provider_state.last_failure = datetime.now() - timedelta(seconds=61)
    assert provider_state.status == ProviderStatus.DEGRADED

def test_budget_guardian():
    # TO DO: implement budget guardian test
    pass