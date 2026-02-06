import pytest
from unittest.mock import Mock
from evolution import (
    AuditVerdict,
    AuditGrade,
    FitnessScorecard,
    CriticalIssue,
    AuditReport,
    MAX_AUDIT_ITERATIONS,
    IMPROVEMENT_THRESHOLD,
    FITNESS_WEIGHTS,
    logger,
)

@pytest.fixture
def mock_logger():
    return Mock(spec=logger)

@pytest.fixture
def mock_code_librarian():
    return Mock(spec=CodeLibrarian)

@pytest.fixture
def mock_omni_router():
    return Mock(spec=OmniRouter)

def test_fitness_scorecard_readability(mock_logger):
    scorecard = FitnessScorecard(readability=90, scalability=80, error_handling=70, coupling=60)
    assert scorecard.readability == 90
    assert scorecard.total == (90 * FITNESS_WEIGHTS["readability"] +
                               80 * FITNESS_WEIGHTS["scalability"] +
                               70 * FITNESS_WEIGHTS["error_handling"] +
                               60 * FITNESS_WEIGHTS["coupling"])

def test_fitness_scorecard_grade(mock_logger):
    scorecard = FitnessScorecard(readability=95, scalability=90, error_handling=85, coupling=80)
    assert scorecard.grade == AuditGrade.S

def test_critical_issue(mock_logger):
    issue = CriticalIssue(category="SOLID", severity="CRITICAL", description="description", solution="solution")
    assert issue.category == "SOLID"
    assert issue.severity == "CRITICAL"
    assert issue.description == "description"
    assert issue.solution == "solution"

def test_audit_report(mock_logger, mock_code_librarian, mock_omni_router):
    report = AuditReport(timestamp=datetime.now(), local_file="local_file", benchmark_repo="benchmark_repo")
    assert report.timestamp
    assert report.local_file == "local_file"
    assert report.benchmark_repo == "benchmark_repo"

def test_max_audit_iterations(mock_logger):
    assert MAX_AUDIT_ITERATIONS == 3

def test_improvement_threshold(mock_logger):
    assert IMPROVEMENT_THRESHOLD == 0.15

def test_fitness_weights(mock_logger):
    assert FITNESS_WEIGHTS == {
        "readability": 0.25,
        "scalability": 0.25,
        "error_handling": 0.25,
        "coupling": 0.25,
    }

def test_audit_verdict(mock_logger):
    assert AuditVerdict.APPROVED == "APPROVED"
    assert AuditVerdict.NEEDS_WORK == "NEEDS_WORK"
    assert AuditVerdict.REJECTED == "REJECTED"

def test_audit_grade(mock_logger):
    assert AuditGrade.S == "S"
    assert AuditGrade.A == "A"
    assert AuditGrade.B == "B"
    assert AuditGrade.C == "C"
    assert AuditGrade.D == "D"
    assert AuditGrade.F == "F"

class TestFitnessScorecard:
    @pytest.mark.parametrize("readability, scalability, error_handling, coupling, expected_total", [
        (90, 80, 70, 60, 75),
        (95, 90, 85, 80, 87.5),
        (80, 70, 60, 50, 65),
    ])
    def test_total(self, readability, scalability, error_handling, coupling, expected_total):
        scorecard = FitnessScorecard(readability, scalability, error_handling, coupling)
        assert scorecard.total == expected_total

    @pytest.mark.parametrize("readability, scalability, error_handling, coupling, expected_grade", [
        (95, 90, 85, 80, AuditGrade.S),
        (90, 80, 70, 60, AuditGrade.A),
        (80, 70, 60, 50, AuditGrade.B),
        (70, 60, 50, 40, AuditGrade.C),
        (60, 50, 40, 30, AuditGrade.D),
        (50, 40, 30, 20, AuditGrade.F),
    ])
    def test_grade(self, readability, scalability, error_handling, coupling, expected_grade):
        scorecard = FitnessScorecard(readability, scalability, error_handling, coupling)
        assert scorecard.grade == expected_grade

class TestCriticalIssue:
    def test_to_dict(self):
        issue = CriticalIssue(category="SOLID", severity="CRITICAL", description="description", solution="solution")
        expected_dict = {
            "category": "SOLID",
            "severity": "CRITICAL",
            "description": "description",
            "solution": "solution",
        }
        assert issue.to_dict() == expected_dict

class TestAuditReport:
    def test_to_dict(self):
        report = AuditReport(timestamp=datetime.now(), local_file="local_file", benchmark_repo="benchmark_repo")
        expected_dict = {
            "timestamp": report.timestamp,
            "local_file": "local_file",
            "benchmark_repo": "benchmark_repo",
        }
        assert report.__dict__ == expected_dict