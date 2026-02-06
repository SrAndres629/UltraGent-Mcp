import pytest
from unittest.mock import Mock
from your_module import SkeletonExtractor, CodeSkeleton, LANGUAGE_GRAMMARS, EXTRACTION_NODES

@pytest.fixture
def skeleton_extractor():
    return SkeletonExtractor()

def test_skeleton_extractor_init(skeleton_extractor):
    assert skeleton_extractor._parsers == {}
    assert isinstance(skeleton_extractor._lock, type(Lock()))

def test_get_parser(skeleton_extractor):
    parser, language = skeleton_extractor._get_parser('.py')
    assert parser is not None
    assert language == 'python'

def test_get_parser_cached(skeleton_extractor):
    parser1, language1 = skeleton_extractor._get_parser('.py')
    parser2, language2 = skeleton_extractor._get_parser('.py')
    assert parser1 is parser2
    assert language1 == language2

def test_code_skeleton_init():
    code_skeleton = CodeSkeleton(
        name='test',
        node_type='function',
        signature='test()',
        docstring='Test function',
        file_path='test.py',
        start_line=1,
        end_line=2,
        language='python',
        file_hash='hash',
    )
    assert code_skeleton.name == 'test'
    assert code_skeleton.node_type == 'function'
    assert code_skeleton.signature == 'test()'
    assert code_skeleton.docstring == 'Test function'
    assert code_skeleton.file_path == 'test.py'
    assert code_skeleton.start_line == 1
    assert code_skeleton.end_line == 2
    assert code_skeleton.language == 'python'
    assert code_skeleton.file_hash == 'hash'

def test_code_skeleton_to_embedding_text():
    code_skeleton = CodeSkeleton(
        name='test',
        node_type='function',
        signature='test()',
        docstring='Test function',
        file_path='test.py',
        start_line=1,
        end_line=2,
        language='python',
        file_hash='hash',
    )
    embedding_text = code_skeleton.to_embedding_text()
    assert embedding_text.startswith('function: test')

def test_code_skeleton_to_metadata():
    code_skeleton = CodeSkeleton(
        name='test',
        node_type='function',
        signature='test()',
        docstring='Test function',
        file_path='test.py',
        start_line=1,
        end_line=2,
        language='python',
        file_hash='hash',
    )
    metadata = code_skeleton.to_metadata()
    assert metadata['name'] == 'test'
    assert metadata['node_type'] == 'function'
    assert metadata['file_path'] == 'test.py'
    assert metadata['start_line'] == 1
    assert metadata['end_line'] == 2
    assert metadata['language'] == 'python'
    assert metadata['file_hash'] == 'hash'

def test_language_grammars():
    assert LANGUAGE_GRAMMARS == {
        '.py': ('tree_sitter_python', 'python'),
        '.js': ('tree_sitter_javascript', 'javascript'),
        '.jsx': ('tree_sitter_javascript', 'javascript'),
        '.ts': ('tree_sitter_typescript', 'typescript'),
        '.tsx': ('tree_sitter_typescript', 'tsx'),
        '.html': ('tree_sitter_html', 'html'),
    }

def test_extraction_nodes():
    assert EXTRACTION_NODES == {
        'python': [
            'function_definition',
            'class_definition',
            'decorated_definition',
        ],
        'javascript': [
            'function_declaration',
            'class_declaration',
            'arrow_function',
            'method_definition',
        ],
        'typescript': [
            'function_declaration',
            'class_declaration',
            'arrow_function',
            'method_definition',
            'interface_declaration',
            'type_alias_declaration',
        ],
        'html': [
            'element',
            'script_element',
            'style_element',
        ],
    }

class TestUnsupportedLanguageError:
    def test_init(self):
        error = UnsupportedLanguageError('Test language')
        assert str(error) == 'Lenguaje no soportado: Test language'

class TestParsingError:
    def test_init(self):
        error = ParsingError('Test error')
        assert str(error) == 'Error al parsear el código: Test error'