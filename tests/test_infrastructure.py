import asyncio
import logging
import os
import pytest
from unittest.mock import patch, MagicMock
from infrastructure import InfrastructureController, get_infrastructure

@pytest.fixture
def infra_controller():
    return InfrastructureController()

@pytest.fixture
def mock_env():
    with patch.dict('os.environ', {
        "CLOUDFLARE_API_TOKEN": "token",
        "VERCEL_TOKEN": "token",
        "SUPABASE_KEY": "token"
    }):
        yield

@pytest.fixture
def mock_env_empty():
    with patch.dict('os.environ', {}):
        yield

def test_infra_controller_init(infra_controller):
    assert infra_controller.cloudflare_enabled == False
    assert infra_controller.vercel_enabled == False
    assert infra_controller.supabase_enabled == False

def test_infra_controller_init_with_env(mock_env):
    infra_controller = InfrastructureController()
    assert infra_controller.cloudflare_enabled == True
    assert infra_controller.vercel_enabled == True
    assert infra_controller.supabase_enabled == True

def test_purge_global_cache_no_zone(infra_controller):
    result = asyncio.run(infra_controller.purge_global_cache())
    assert result == {"cloudflare": "Skipped", "vercel": "Skipped"}

def test_purge_global_cache_with_zone(mock_env):
    infra_controller = InfrastructureController()
    result = asyncio.run(infra_controller.purge_global_cache("zone_id"))
    assert result == {"cloudflare": "Purged (Simulated)", "vercel": "Purged (Simulated)"}

def test_purge_global_cache_with_zone_error(mock_env):
    infra_controller = InfrastructureController()
    with patch.object(infra_controller, 'purge_global_cache', side_effect=Exception("Error")):
        result = asyncio.run(infra_controller.purge_global_cache("zone_id"))
        assert result == {"cloudflare": "Error: Error", "vercel": "Skipped"}

def test_sync_secrets_no_env_file(infra_controller):
    with patch.object(infra_controller, 'env_file', new=None):
        result = asyncio.run(infra_controller.sync_secrets())
        assert result == {"error": ".env not found"}

def test_sync_secrets(mock_env):
    infra_controller = InfrastructureController()
    with patch.object(infra_controller, 'env_file', new="path/to/env/file"):
        with patch.object(infra_controller, 'logger', new=MagicMock()):
            result = asyncio.run(infra_controller.sync_secrets())
            assert result == {"synced_keys": [], "targets": ["Vercel", "Cloudflare Workers"]}

def test_get_status(mock_env):
    infra_controller = InfrastructureController()
    result = infra_controller.get_status()
    assert result == {"cloudflare": True, "vercel": True, "supabase": True}

def test_get_status_empty_env(mock_env_empty):
    infra_controller = InfrastructureController()
    result = infra_controller.get_status()
    assert result == {"cloudflare": False, "vercel": False, "supabase": False}

def test_get_infrastructure():
    infra_controller = get_infrastructure()
    assert isinstance(infra_controller, InfrastructureController)

def test_get_infrastructure_singleton():
    infra_controller1 = get_infrastructure()
    infra_controller2 = get_infrastructure()
    assert infra_controller1 is infra_controller2