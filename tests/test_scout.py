import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch, Mock
import pytest
from pytest import fixture
from pytest_asyncio import fixture as async_fixture

from scout import (
    StackProfile,
    RepositoryHealth,
    ScoutResult,
    SearchResult,
    GOLD_STANDARD_THRESHOLDS,
    GITHUB_API_BASE,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    REQUEST_DELAY_SECONDS,
    logger,
)

@fixture
def stack_profile():
    return StackProfile(
        project_name="Test Project",
        core_frameworks=["framework1", "framework2"],
        libraries=["lib1", "lib2"],
        constraints={"constraint1": "value1", "constraint2": "value2"},
    )

@fixture
def repository_health():
    return RepositoryHealth(
        name="Test Repository",
        full_name="Test Owner/Test Repository",
        url="https://github.com/Test Owner/Test Repository",
        stars=1000,
        forks=50,
        open_issues=10,
        language="Python",
        topics=["topic1", "topic2"],
        last_updated=datetime.now(),
        has_tests=True,
        has_typing=True,
        has_readme=True,
        commit_frequency=10.0,
    )

@fixture
def scout_result():
    return ScoutResult(
        success=True,
        data="Test Data",
        error=None,
        rate_limit_remaining=100,
        cached=False,
    )

@fixture
def search_result():
    return SearchResult(
        title="Test Title",
        url="https://test.url.com",
    )

def test_stack_profile_from_file():
    with patch("scout.json.loads") as mock_json_loads:
        mock_json_loads.return_value = {
            "project_name": "Test Project",
            "core_frameworks": ["framework1", "framework2"],
            "libraries": ["lib1", "lib2"],
            "constraints": {"constraint1": "value1", "constraint2": "value2"},
        }
        profile = StackProfile.from_file(Path("test.json"))
        assert profile.project_name == "Test Project"
        assert profile.core_frameworks == ["framework1", "framework2"]
        assert profile.libraries == ["lib1", "lib2"]
        assert profile.constraints == {"constraint1": "value1", "constraint2": "value2"}

def test_repository_health_health_score(repository_health):
    assert repository_health.health_score >= 0
    assert repository_health.health_score <= 100

def test_repository_health_is_gold_standard(repository_health):
    assert repository_health.is_gold_standard() == (repository_health.stars >= GOLD_STANDARD_THRESHOLDS["min_stars"] and
                                                    repository_health.forks >= GOLD_STANDARD_THRESHOLDS["min_forks"] and
                                                    (datetime.now() - repository_health.last_updated).days <= GOLD_STANDARD_THRESHOLDS["max_days_since_update"] and
                                                    repository_health.open_issues / repository_health.stars <= GOLD_STANDARD_THRESHOLDS["max_open_issues_ratio"])

def test_scout_result():
    result = ScoutResult(
        success=True,
        data="Test Data",
        error=None,
        rate_limit_remaining=100,
        cached=False,
    )
    assert result.success
    assert result.data == "Test Data"
    assert result.error is None
    assert result.rate_limit_remaining == 100
    assert not result.cached

def test_search_result():
    result = SearchResult(
        title="Test Title",
        url="https://test.url.com",
    )
    assert result.title == "Test Title"
    assert result.url == "https://test.url.com"

@async_fixture
async def async_scout_result():
    return ScoutResult(
        success=True,
        data="Test Data",
        error=None,
        rate_limit_remaining=100,
        cached=False,
    )

@async_fixture
async def async_search_result():
    return SearchResult(
        title="Test Title",
        url="https://test.url.com",
    )

@patch("scout.httpx.get")
async def test_async_scout_result(mock_get, async_scout_result):
    mock_get.return_value = Mock(status_code=200, json=lambda: {"data": "Test Data"})
    result = async_scout_result
    assert result.success
    assert result.data == "Test Data"
    assert result.error is None
    assert result.rate_limit_remaining == 100
    assert not result.cached

@patch("scout.httpx.get")
async def test_async_search_result(mock_get, async_search_result):
    mock_get.return_value = Mock(status_code=200, json=lambda: {"title": "Test Title", "url": "https://test.url.com"})
    result = async_search_result
    assert result.title == "Test Title"
    assert result.url == "https://test.url.com"

def test_stack_profile_invalid_file():
    with patch("scout.Path.exists") as mock_exists:
        mock_exists.return_value = False
        profile = StackProfile.from_file(Path("invalid.json"))
        assert profile.project_name == "Default"

def test_repository_health_invalid_input(repository_health):
    with patch("scout.datetime.now") as mock_now:
        mock_now.return_value = datetime(2022, 1, 1)
        repository_health.last_updated = datetime(2021, 1, 1)
        assert repository_health.is_gold_standard()

def test_scout_result_invalid_input():
    result = ScoutResult(
        success=False,
        data=None,
        error="Test Error",
        rate_limit_remaining=-1,
        cached=True,
    )
    assert not result.success
    assert result.data is None
    assert result.error == "Test Error"
    assert result.rate_limit_remaining == -1
    assert result.cached

def test_search_result_invalid_input():
    result = SearchResult(
        title=None,
        url=None,
    )
    assert result.title is None
    assert result.url is None