import pytest
from unittest.mock import MagicMock
from metrics import ComplexityVisitor, MetricsEngine, FileMetric, FunctionMetric
import ast
import math

@pytest.fixture
def sample_code():
    return """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""

@pytest.fixture
def complex_code():
    return """
def add(a, b):
    if a > 0:
        return a + b
    else:
        return a - b

def multiply(a, b):
    for i in range(10):
        a += 1
    return a * b
"""

def test_ComplexityVisitor_init():
    visitor = ComplexityVisitor()
    assert visitor.complexity == 1

def test_ComplexityVisitor_visit_If():
    visitor = ComplexityVisitor()
    node = ast.If()
    visitor.visit_If(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_For():
    visitor = ComplexityVisitor()
    node = ast.For()
    visitor.visit_For(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_AsyncFor():
    visitor = ComplexityVisitor()
    node = ast.AsyncFor()
    visitor.visit_AsyncFor(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_While():
    visitor = ComplexityVisitor()
    node = ast.While()
    visitor.visit_While(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_Try():
    visitor = ComplexityVisitor()
    node = ast.Try()
    visitor.visit_Try(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_With():
    visitor = ComplexityVisitor()
    node = ast.With()
    visitor.visit_With(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_AsyncWith():
    visitor = ComplexityVisitor()
    node = ast.AsyncWith()
    visitor.visit_AsyncWith(node)
    assert visitor.complexity == 2

def test_ComplexityVisitor_visit_BoolOp():
    visitor = ComplexityVisitor()
    node = ast.BoolOp()
    node.values = [1, 2, 3]
    visitor.visit_BoolOp(node)
    assert visitor.complexity == 3

def test_MetricsEngine_calculate_grade():
    assert MetricsEngine._calculate_grade(3) == "A"
    assert MetricsEngine._calculate_grade(7) == "B"
    assert MetricsEngine._calculate_grade(15) == "C"
    assert MetricsEngine._calculate_grade(30) == "D"
    assert MetricsEngine._calculate_grade(50) == "F"

def test_MetricsEngine_analyze_code(sample_code):
    result = MetricsEngine.analyze_code(sample_code)
    assert isinstance(result, FileMetric)
    assert result.file_path == "unknown.py"
    assert result.maintainability_index > 0
    assert result.cyclomatic_complexity > 0
    assert result.loc > 0
    assert len(result.functions) > 0

def test_MetricsEngine_analyze_code_complex(complex_code):
    result = MetricsEngine.analyze_code(complex_code)
    assert isinstance(result, FileMetric)
    assert result.file_path == "unknown.py"
    assert result.maintainability_index > 0
    assert result.cyclomatic_complexity > 0
    assert result.loc > 0
    assert len(result.functions) > 0

def test_MetricsEngine_analyze_code_syntax_error():
    result = MetricsEngine.analyze_code("def add(a, b: return a + b")
    assert isinstance(result, FileMetric)
    assert result.file_path == "unknown.py"
    assert result.maintainability_index == 0
    assert result.cyclomatic_complexity == 0
    assert result.loc == 0
    assert result.functions == []
    assert result.grade == "F (Syntax Error)"

def test_MetricsEngine_analyze_code_empty():
    result = MetricsEngine.analyze_code("")
    assert isinstance(result, FileMetric)
    assert result.file_path == "unknown.py"
    assert result.maintainability_index == 0
    assert result.cyclomatic_complexity == 0
    assert result.loc == 0
    assert result.functions == []
    assert result.grade == "A"

def test_FileMetric_init():
    file_metric = FileMetric("test.py", 100, 10, 100, [])
    assert file_metric.file_path == "test.py"
    assert file_metric.maintainability_index == 100
    assert file_metric.cyclomatic_complexity == 10
    assert file_metric.loc == 100
    assert file_metric.functions == []
    assert file_metric.grade == "A"

def test_FunctionMetric_init():
    function_metric = FunctionMetric("add", 10, 10, 2, False)
    assert function_metric.name == "add"
    assert function_metric.complexity == 10
    assert function_metric.length == 10
    assert function_metric.arg_count == 2
    assert function_metric.is_async == False
    assert function_metric.grade == "A"