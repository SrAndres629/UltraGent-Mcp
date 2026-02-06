import pytest
from unittest.mock import Mock, patch
from threading import Lock
from librarian import SkeletonExtractor, CodeSkeleton, LANGUAGE_GRAMMARS, EXTRACTION_NODES, UnsupportedLanguageError, ParsingError

@pytest.fixture
def skeleton_extractor():
    return SkeletonExtractor()

def test_skeleton_extractor_init(skeleton_extractor):
    assert skeleton_extractor._parsers == {}
    assert isinstance(skeleton_extractor._lock, type(Lock()))

def test_get_parser_unsupported(skeleton_extractor):
    with pytest.raises(UnsupportedLanguageError):
        skeleton_extractor._get_parser('.unknown')

def test_get_node_name_python(skeleton_extractor):
    source = b"def test_func(): pass"
    class MockNode:
        def __init__(self, type, children):
            self.type = type
            self.children = children
    
    mock_id = MockNode("identifier", [])
    mock_id.start_byte = 4
    mock_id.end_byte = 13
    
    node = MockNode("function_definition", [mock_id])
    assert skeleton_extractor._get_node_name(node, source, "python") == "test_func"

def test_extract_signature_python(skeleton_extractor):
    source = b"def test_func(a: int):"
    class MockNode:
        def __init__(self, type, start_byte, end_byte):
            self.type = type
            self.start_byte = start_byte
            self.end_byte = end_byte
            self.children = []
            
    colon = MockNode(":", 21, 22)
    node = MockNode("function_definition", 0, 22)
    node.children = [colon]
    
    assert skeleton_extractor._extract_signature(node, source, "python") == "def test_func(a: int):"

def test_code_skeleton_metadata():
    skeleton = CodeSkeleton(
        name="test",
        node_type="function",
        signature="def test():",
        docstring="doc",
        file_path="test.py",
        start_line=1,
        end_line=5,
        language="python",
        file_hash="hash"
    )
    metadata = skeleton.to_metadata()
    assert metadata["name"] == "test"
    assert metadata["node_type"] == "function"
    assert "indexed_at" in metadata

class TestUnsupportedLanguageError:
    def test_init(self):
        error = UnsupportedLanguageError('Test language')
        assert str(error) == 'Test language'

class TestParsingError:
    def test_init(self):
        error = ParsingError('Test error')
        assert str(error) == 'Test error'