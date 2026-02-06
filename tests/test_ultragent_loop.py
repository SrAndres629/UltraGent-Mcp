import pytest
from unittest.mock import patch, MagicMock
from ultragent_loop import hunter_loop, logger
import asyncio
import logging

@pytest.fixture
def mock_get_neuro_architect():
    with patch('ultragent_loop.get_neuro_architect') as mock_get_neuro_architect:
        yield mock_get_neuro_architect

@pytest.fixture
def mock_get_librarian():
    with patch('ultragent_loop.get_librarian') as mock_get_librarian:
        yield mock_get_librarian

@pytest.fixture
def mock_get_mechanic():
    with patch('ultragent_loop.get_mechanic') as mock_get_mechanic:
        yield mock_get_mechanic

@pytest.fixture
def mock_get_cortex():
    with patch('ultragent_loop.get_cortex') as mock_get_cortex:
        yield mock_get_cortex

@pytest.fixture
def mock_get_sentinel():
    with patch('ultragent_loop.get_sentinel') as mock_get_sentinel:
        yield mock_get_sentinel

@pytest.fixture
def mock_asyncio_sleep():
    with patch('asyncio.sleep') as mock_asyncio_sleep:
        yield mock_asyncio_sleep

@pytest.mark.asyncio
async def test_hunter_loop(mock_get_neuro_architect, mock_get_librarian, mock_get_mechanic, mock_get_cortex, mock_get_sentinel):
    neuro_architect = mock_get_neuro_architect()
    librarian = mock_get_librarian()
    mechanic = mock_get_mechanic()
    cortex = mock_get_cortex()
    sentinel = mock_get_sentinel()

    neuro_architect.get_next_focus.return_value = {'target': None}
    mechanic.is_available = True

    await hunter_loop()

    neuro_architect.get_next_focus.assert_called()
    librarian.scan_debt.assert_not_called()
    mechanic.heal_node.assert_not_called()

@pytest.mark.asyncio
async def test_hunter_loop_with_target(mock_get_neuro_architect, mock_get_librarian, mock_get_mechanic, mock_get_cortex, mock_get_sentinel):
    neuro_architect = mock_get_neuro_architect()
    librarian = mock_get_librarian()
    mechanic = mock_get_mechanic()
    cortex = mock_get_cortex()
    sentinel = mock_get_sentinel()

    neuro_architect.get_next_focus.return_value = {'target': 'target_node', 'priority': 0.5}
    mechanic.is_available = True
    librarian.scan_debt.return_value = {'score': 50, 'issues': []}

    await hunter_loop()

    neuro_architect.get_next_focus.assert_called()
    librarian.scan_debt.assert_called_once_with('target_node')
    mechanic.heal_node.assert_not_called()

@pytest.mark.asyncio
async def test_hunter_loop_with_issue(mock_get_neuro_architect, mock_get_librarian, mock_get_mechanic, mock_get_cortex, mock_get_sentinel):
    neuro_architect = mock_get_neuro_architect()
    librarian = mock_get_librarian()
    mechanic = mock_get_mechanic()
    cortex = mock_get_cortex()
    sentinel = mock_get_sentinel()

    neuro_architect.get_next_focus.return_value = {'target': 'target_node', 'priority': 0.5}
    mechanic.is_available = True
    librarian.scan_debt.return_value = {'score': 50, 'issues': [{'category': 'SECURITY', 'type': 'issue_type', 'line': 1, 'content': 'issue_content'}]}

    await hunter_loop()

    neuro_architect.get_next_focus.assert_called()
    librarian.scan_debt.assert_called_once_with('target_node')
    mechanic.heal_node.assert_called_once_with('target_node', '[SECURITY] issue_type at line 1: issue_content')

@pytest.mark.asyncio
async def test_hunter_loop_with_mechanic_failure(mock_get_neuro_architect, mock_get_librarian, mock_get_mechanic, mock_get_cortex, mock_get_sentinel):
    neuro_architect = mock_get_neuro_architect()
    librarian = mock_get_librarian()
    mechanic = mock_get_mechanic()
    cortex = mock_get_cortex()
    sentinel = mock_get_sentinel()

    neuro_architect.get_next_focus.return_value = {'target': 'target_node', 'priority': 0.5}
    mechanic.is_available = True
    librarian.scan_debt.return_value = {'score': 50, 'issues': [{'category': 'SECURITY', 'type': 'issue_type', 'line': 1, 'content': 'issue_content'}]}
    mechanic.heal_node.return_value = {'success': False, 'error': 'heal_error'}

    await hunter_loop()

    neuro_architect.get_next_focus.assert_called()
    librarian.scan_debt.assert_called_once_with('target_node')
    mechanic.heal_node.assert_called_once_with('target_node', '[SECURITY] issue_type at line 1: issue_content')

@pytest.mark.asyncio
async def test_hunter_loop_with_mechanic_unavailable(mock_get_neuro_architect, mock_get_librarian, mock_get_mechanic, mock_get_cortex, mock_get_sentinel):
    neuro_architect = mock_get_neuro_architect()
    librarian = mock_get_librarian()
    mechanic = mock_get_mechanic()
    cortex = mock_get_cortex()
    sentinel = mock_get_sentinel()

    neuro_architect.get_next_focus.return_value = {'target': 'target_node', 'priority': 0.5}
    mechanic.is_available = False

    await hunter_loop()

    neuro_architect.get_next_focus.assert_called()
    librarian.scan_debt.assert_not_called()
    mechanic.heal_node.assert_not_called()

def test_logger():
    logger.info('test_logger')
    logger.warning('test_logger')
    logger.error('test_logger')
    logger.critical('test_logger')