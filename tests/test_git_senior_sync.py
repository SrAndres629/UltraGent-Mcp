import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from git_senior_sync import generate_commit_message, run_git, git_senior_sync
import asyncio

@pytest.fixture
def mock_router():
    with patch('router.get_router') as mock_get_router:
        mock_router = mock_get_router.return_value
        mock_router.route_task = AsyncMock(return_value={'response': 'feat: test commit message'})
        yield mock_router

@pytest.fixture
def mock_subprocess_run():
    with patch('subprocess.run') as mock_subprocess_run:
        mock_subprocess_run.return_value = MagicMock(stdout='test output', returncode=0)
        yield mock_subprocess_run

@pytest.fixture
def mock_logger():
    with patch('git_senior_sync.logger') as mock_logger:
        yield mock_logger

@pytest.mark.asyncio
async def test_generate_commit_message_empty_diff(mock_router):
    assert await generate_commit_message('') == 'chore: minor updates'

@pytest.mark.asyncio
async def test_generate_commit_message(mock_router):
    assert await generate_commit_message('test diff') == 'feat: test commit message'

@pytest.mark.asyncio
async def test_generate_commit_message_exception(mock_router):
    mock_router.route_task.side_effect = Exception('test exception')
    assert await generate_commit_message('test diff') == 'feat: automate senior git synchronization flow [emergency fallback]'

def test_run_git(mock_subprocess_run):
    assert run_git(['status', '--short']) == 'test output'

def test_run_git_called_process_error(mock_subprocess_run):
    mock_subprocess_run.return_value = MagicMock(stdout=None, returncode=1, stderr='test error')
    assert run_git(['status', '--short']) == ''

def test_run_git_unexpected_error(mock_subprocess_run):
    mock_subprocess_run.side_effect = Exception('test error')
    assert run_git(['status', '--short']) == ''

@pytest.mark.asyncio
async def test_git_senior_sync_no_changes(mock_subprocess_run, mock_logger):
    mock_subprocess_run.return_value = MagicMock(stdout=None, returncode=0)
    await git_senior_sync()
    mock_logger.info.assert_any_call('✅ No hay cambios pendientes.')

@pytest.mark.asyncio
async def test_git_senior_sync_changes(mock_subprocess_run, mock_logger, mock_router):
    mock_subprocess_run.return_value = MagicMock(stdout='test output', returncode=0)
    await git_senior_sync()
    mock_logger.info.assert_any_call('✨ Mensaje Sugerido: feat: test commit message')

@pytest.mark.asyncio
async def test_git_senior_sync_exception(mock_subprocess_run, mock_logger):
    mock_subprocess_run.side_effect = Exception('test error')
    await git_senior_sync()
    mock_logger.error.assert_any_call('Unexpected error in run_git: test error')