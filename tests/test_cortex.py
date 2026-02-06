import pytest
from unittest.mock import Mock, patch
from cortex import Cortex, Memory
from datetime import datetime

@pytest.fixture
def cortex():
    with patch('cortex.get_librarian') as mock_librarian:
        yield Cortex()

def test_memory_to_json():
    memory = Memory(content="Test", tags=["tag1", "tag2"], importance=0.5, metadata={"key": "value"})
    expected_json = '{"content": "Test", "tags": ["tag1", "tag2"], "importance": 0.5, "metadata": {"key": "value"}}'
    assert memory.to_json() == expected_json

def test_cortex_init(cortex):
    assert cortex._db_path.exists()
    assert cortex._lock is not None

def test_cortex_add_memory(cortex):
    with patch.object(cortex, '_librarian') as mock_librarian:
        memory_id = cortex.add_memory("Test", ["tag1", "tag2"], 0.5, {"key": "value"})
        assert memory_id > 0
        mock_librarian.index_memory.assert_called_once()

def test_cortex_search_memories(cortex):
    with patch.object(cortex._librarian, 'semantic_search') as mock_semantic_search:
        mock_semantic_search.return_value = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.2}]
        results = cortex.search_memories("Test", 5)
        assert len(results) == 2

def test_cortex_get_all_memories(cortex):
    with patch.object(cortex, '_init_db') as mock_init_db:
        cortex.add_memory("Test1", ["tag1", "tag2"], 0.5, {"key": "value"})
        cortex.add_memory("Test2", ["tag3", "tag4"], 0.8, {"key2": "value2"})
        memories = cortex.get_all_memories()
        assert len(memories) == 2

def test_cortex_get_related_memories(cortex):
    with patch.object(cortex._librarian, 'semantic_search') as mock_semantic_search:
        mock_semantic_search.return_value = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.2}]
        results = cortex.get_related_memories("Test")
        assert len(results) == 2

def test_cortex_init_db(cortex):
    with patch.object(cortex, '_lock') as mock_lock:
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.return_value = Mock()
            cortex._init_db()
            mock_connect.assert_called_once_with(str(cortex._db_path))

def test_cortex_add_memory_failure(cortex):
    with patch.object(cortex, '_librarian') as mock_librarian:
        mock_librarian.index_memory.side_effect = Exception("Test exception")
        memory_id = cortex.add_memory("Test", ["tag1", "tag2"], 0.5, {"key": "value"})
        assert memory_id > 0
        mock_librarian.index_memory.assert_called_once()

def test_cortex_search_memories_failure(cortex):
    with patch.object(cortex._librarian, 'semantic_search') as mock_semantic_search:
        mock_semantic_search.side_effect = Exception("Test exception")
        results = cortex.search_memories("Test", 5)
        assert results is not None

def test_cortex_get_all_memories_failure(cortex):
    with patch('sqlite3.connect') as mock_connect:
        mock_connect.side_effect = Exception("Test exception")
        memories = cortex.get_all_memories()
        assert memories == []

def test_cortex_get_related_memories_failure(cortex):
    with patch.object(cortex._librarian, 'semantic_search') as mock_semantic_search:
        mock_semantic_search.side_effect = Exception("Test exception")
        results = cortex.get_related_memories("Test")
        assert results == []