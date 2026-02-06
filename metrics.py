"""
ULTRAGENT METRICS v0.1 (Native Hardening)
=========================================
Módulo de análisis estático determinista.
Calcula métricas de complejidad usando solo la librería estándar (ast).

Implementa:
- Complejidad Ciclomática (McCabe)
- Halstead Metrics (simuladas para Maintainability Index)
- Análisis de Líneas Lógicas (LLOC)
- Detector de funciones "God Object"
"""

import ast
import math
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FunctionMetric:
    name: str
    complexity: int
    length: int
    arg_count: int
    is_async: bool
    grade: str = "A"

@dataclass
class FileMetric:
    file_path: str
    maintainability_index: float
    cyclomatic_complexity: int
    loc: int
    functions: List[FunctionMetric]
    grade: str = "A"

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1  # Base complexity

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_AsyncWith(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

class MetricsEngine:
    """Motor de análisis determinista nativo."""

    @staticmethod
    def _calculate_grade(complexity: int) -> str:
        if complexity <= 5: return "A"  # Simple
        if complexity <= 10: return "B" # Moderate
        if complexity <= 20: return "C" # Complex
        if complexity <= 40: return "D" # High Risk
        return "F" # Unmaintainable (God Function)

    @staticmethod
    def analyze_code(code: str, filename: str = "unknown.py") -> FileMetric:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FileMetric(filename, 0.0, 0, 0, [], "F (Syntax Error)")

        total_complexity = 0
        functions = []
        loc = len(code.splitlines())

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visitor = ComplexityVisitor()
                # Visit children of function only
                for child in ast.iter_child_nodes(node):
                    visitor.visit(child)
                
                comp = visitor.complexity
                func_len = node.end_lineno - node.lineno if hasattr(node, "end_lineno") else 0
                arg_count = len(node.args.args)
                
                functions.append(FunctionMetric(
                    name=node.name,
                    complexity=comp,
                    length=func_len,
                    arg_count=arg_count,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    grade=MetricsEngine._calculate_grade(comp)
                ))
                total_complexity += comp

        # Calculate simplistic Maintainability Index (MI)
        # MI = 171 - 5.2 * ln(Halstead Vol) - 0.23 * (Complexity) - 16.2 * ln(LOC)
        # Simplified approximation without full Halstead:
        # MI ~= 171 - 0.23 * AvgComplexity - 16.2 * ln(LOC) - 50 (Penalty factor)
        
        avg_comp = (total_complexity / len(functions)) if functions else 1
        try:
            mi = max(0, 100 - (avg_comp * 2) - (math.log(max(loc, 1)) * 5))
        except:
            mi = 50.0

        file_grade = "A"
        if mi < 65: file_grade = "B"
        if mi < 50: file_grade = "C"
        if mi < 30: file_grade = "D"
        if mi < 10: file_grade = "F"

        return FileMetric(
            file_path=filename,
            maintainability_index=round(mi, 2),
            cyclomatic_complexity=total_complexity,
            loc=loc,
            functions=functions,
            grade=file_grade
        )
