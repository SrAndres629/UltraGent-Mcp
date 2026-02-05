"""
ULTRAGENT EVOLUTION v0.1
========================
Módulo de auditoría arquitectónica con crítica despiadada.

Implementa:
- Comparación genética de código contra Gold Standards
- Fitness Scorecard (Legibilidad, Escalabilidad, Error Handling, Acoplamiento)
- Integración con Omni-Router para análisis por Kimi K2.5
- Loop protection (max 3 iteraciones, 15% mejora mínima)
- Generación de reportes en .ai/reports/
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
REPORTS_DIR = AI_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Límites de auditoría
MAX_AUDIT_ITERATIONS = 3
IMPROVEMENT_THRESHOLD = 0.15  # 15% mejora mínima

# Pesos del Fitness Scorecard
FITNESS_WEIGHTS = {
    "readability": 0.25,
    "scalability": 0.25,
    "error_handling": 0.25,
    "coupling": 0.25,
}

# Logger
logger = logging.getLogger("ultragent.evolution")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS DE CRÍTICA SEVERA
# ═══════════════════════════════════════════════════════════════════════════════

SEVERE_CRITIC_SYSTEM = """You are EVOLUTION-CRITIC, an elite code reviewer with 30 years of experience.

Your role is to be BRUTALLY HONEST and UNFORGIVING in your code analysis.
You will compare the user's code against a Gold Standard benchmark.

## Your Personality:
- You are NEVER impressed by mediocre code
- You find flaws where others see "good enough"
- You demand EXCELLENCE in every line
- You cite specific patterns and anti-patterns by name
- You give ACTIONABLE feedback, not vague suggestions

## Focus Areas (IGNORE trivial issues):
✓ SOLID principles violations
✓ Clean Architecture patterns
✓ Error handling depth
✓ Interface contracts
✓ Dependency injection
✓ Separation of concerns
✓ Testability

✗ Variable naming style (unless egregiously bad)
✗ Comment formatting
✗ Import ordering
✗ Whitespace preferences

## Output Format:
You MUST respond with a valid JSON object (no markdown, no explanation outside JSON).
"""

AUDIT_PROMPT_TEMPLATE = """## Task: Comparative Code Audit

### LOCAL CODE (to evaluate):
```{language}
{local_code}
```

### BENCHMARK CODE (Gold Standard):
```{language}
{benchmark_code}
```

### Instructions:
1. Analyze ARCHITECTURAL patterns only (not trivial style)
2. Compare against the benchmark's superior patterns
3. Score each dimension 0-100
4. Provide SPECIFIC improvements with code examples

Respond with this JSON structure:
{{
    "scores": {{
        "readability": <0-100>,
        "scalability": <0-100>,
        "error_handling": <0-100>,
        "coupling": <0-100>
    }},
    "fitness": <0-100>,
    "grade": "<S|A|B|C|D|F>",
    "verdict": "<APPROVED|NEEDS_WORK|REJECTED>",
    "critical_issues": [
        {{
            "category": "<SOLID|ARCHITECTURE|ERROR_HANDLING|COUPLING>",
            "severity": "<CRITICAL|MAJOR|MINOR>",
            "description": "<specific problem>",
            "solution": "<specific fix with code>"
        }}
    ],
    "benchmark_lessons": [
        "<pattern from benchmark that local code should adopt>"
    ],
    "praise": [
        "<anything actually good about the local code>"
    ]
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

class AuditVerdict(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_WORK = "NEEDS_WORK"
    REJECTED = "REJECTED"


class AuditGrade(str, Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


@dataclass
class FitnessScorecard:
    """Scorecard de fitness del código."""
    
    readability: float
    scalability: float
    error_handling: float
    coupling: float
    
    @property
    def total(self) -> float:
        """Calcula fitness total ponderado."""
        return (
            self.readability * FITNESS_WEIGHTS["readability"] +
            self.scalability * FITNESS_WEIGHTS["scalability"] +
            self.error_handling * FITNESS_WEIGHTS["error_handling"] +
            self.coupling * FITNESS_WEIGHTS["coupling"]
        )
    
    @property
    def grade(self) -> AuditGrade:
        """Determina grade basado en fitness."""
        total = self.total
        if total >= 95:
            return AuditGrade.S
        elif total >= 85:
            return AuditGrade.A
        elif total >= 70:
            return AuditGrade.B
        elif total >= 55:
            return AuditGrade.C
        elif total >= 40:
            return AuditGrade.D
        else:
            return AuditGrade.F
    
    def to_dict(self) -> dict:
        return {
            "readability": self.readability,
            "scalability": self.scalability,
            "error_handling": self.error_handling,
            "coupling": self.coupling,
            "total": self.total,
            "grade": self.grade.value,
        }


@dataclass
class CriticalIssue:
    """Issue crítico encontrado en auditoría."""
    
    category: str
    severity: str
    description: str
    solution: str
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "solution": self.solution,
        }


@dataclass
class AuditReport:
    """Reporte completo de auditoría."""
    
    timestamp: datetime
    local_file: str
    benchmark_repo: str
    scorecard: FitnessScorecard
    verdict: AuditVerdict
    critical_issues: list[CriticalIssue]
    benchmark_lessons: list[str]
    praise: list[str]
    iteration: int
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "local_file": self.local_file,
            "benchmark_repo": self.benchmark_repo,
            "scorecard": self.scorecard.to_dict(),
            "verdict": self.verdict.value,
            "critical_issues": [i.to_dict() for i in self.critical_issues],
            "benchmark_lessons": self.benchmark_lessons,
            "praise": self.praise,
            "iteration": self.iteration,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Guarda el reporte en JSON."""
        if path is None:
            filename = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = REPORTS_DIR / filename
        
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        logger.info(f"Reporte guardado: {path}")
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION AUDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionAuditor:
    """
    Auditor de código con criterio de élite.
    
    Compara código local contra benchmarks Gold Standard
    usando análisis por LLM con prompts de crítica severa.
    """
    
    def __init__(self):
        self._lock = Lock()
        self._audit_history: list[float] = []
        
        # Estadísticas
        self._stats = {
            "audits_performed": 0,
            "iterations_total": 0,
            "approvals": 0,
            "rejections": 0,
        }
        
        logger.info("EvolutionAuditor inicializado")
    
    async def _call_router(
        self,
        prompt: str,
        system: str = SEVERE_CRITIC_SYSTEM,
    ) -> dict:
        """Llama al Omni-Router para análisis."""
        try:
            from router import get_router
            router = get_router()
            
            result = await router.route_request(
                prompt=prompt,
                task_type="strategic",  # Usar tier strategic (Kimi/Gemini)
                system_prompt=system,
            )
            
            if result.get("success"):
                # Parsear JSON de la respuesta
                response_text = result.get("response", "")
                
                # Intentar extraer JSON
                try:
                    # Buscar JSON en la respuesta
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
                
                return {"error": "Failed to parse response", "raw": response_text}
            else:
                return {"error": result.get("error", "Unknown error")}
                
        except ImportError:
            logger.warning("Router no disponible, usando fallback")
            return await self._fallback_analysis(prompt)
        except Exception as e:
            logger.error(f"Error calling router: {e}")
            return {"error": str(e)}
    
    async def _fallback_analysis(self, prompt: str) -> dict:
        """Análisis fallback sin LLM externo."""
        # Análisis básico sin LLM
        return {
            "scores": {
                "readability": 60,
                "scalability": 60,
                "error_handling": 60,
                "coupling": 60,
            },
            "fitness": 60,
            "grade": "C",
            "verdict": "NEEDS_WORK",
            "critical_issues": [
                {
                    "category": "ANALYSIS",
                    "severity": "MAJOR",
                    "description": "No LLM available for deep analysis",
                    "solution": "Configure Omni-Router with API keys",
                }
            ],
            "benchmark_lessons": [],
            "praise": [],
        }
    
    async def audit_code(
        self,
        local_code: str,
        benchmark_code: str,
        local_file: str = "unknown",
        benchmark_repo: str = "unknown",
        language: str = "python",
        iteration: int = 1,
    ) -> AuditReport:
        """
        Realiza auditoría comparativa de código.
        
        Args:
            local_code: Código local a evaluar
            benchmark_code: Código benchmark Gold Standard
            local_file: Nombre del archivo local
            benchmark_repo: Nombre del repo benchmark
            language: Lenguaje de programación
            iteration: Número de iteración
            
        Returns:
            AuditReport con resultados
        """
        with self._lock:
            self._stats["audits_performed"] += 1
            self._stats["iterations_total"] += 1
        
        # Construir prompt
        prompt = AUDIT_PROMPT_TEMPLATE.format(
            language=language,
            local_code=local_code[:8000],  # Limitar tamaño
            benchmark_code=benchmark_code[:8000],
        )
        
        # Llamar al router
        result = await self._call_router(prompt)
        
        if "error" in result and "scores" not in result:
            # Error en la llamada
            logger.error(f"Audit error: {result.get('error')}")
            result = await self._fallback_analysis(prompt)
        
        # Construir scorecard
        scores = result.get("scores", {})
        scorecard = FitnessScorecard(
            readability=scores.get("readability", 50),
            scalability=scores.get("scalability", 50),
            error_handling=scores.get("error_handling", 50),
            coupling=scores.get("coupling", 50),
        )
        
        # Construir issues
        critical_issues = []
        for issue in result.get("critical_issues", []):
            critical_issues.append(CriticalIssue(
                category=issue.get("category", "UNKNOWN"),
                severity=issue.get("severity", "MAJOR"),
                description=issue.get("description", ""),
                solution=issue.get("solution", ""),
            ))
        
        # Determinar verdict
        verdict_str = result.get("verdict", "NEEDS_WORK")
        try:
            verdict = AuditVerdict(verdict_str)
        except ValueError:
            verdict = AuditVerdict.NEEDS_WORK
        
        # Actualizar estadísticas
        with self._lock:
            if verdict == AuditVerdict.APPROVED:
                self._stats["approvals"] += 1
            elif verdict == AuditVerdict.REJECTED:
                self._stats["rejections"] += 1
            
            self._audit_history.append(scorecard.total)
        
        # Crear reporte
        report = AuditReport(
            timestamp=datetime.now(),
            local_file=local_file,
            benchmark_repo=benchmark_repo,
            scorecard=scorecard,
            verdict=verdict,
            critical_issues=critical_issues,
            benchmark_lessons=result.get("benchmark_lessons", []),
            praise=result.get("praise", []),
            iteration=iteration,
        )
        
        logger.info(
            f"Audit complete: {local_file} | "
            f"Fitness: {scorecard.total:.1f} | "
            f"Grade: {scorecard.grade.value} | "
            f"Verdict: {verdict.value}"
        )
        
        return report
    
    def should_continue_iteration(self) -> tuple[bool, str]:
        """
        Determina si debe continuar iterando.
        
        Returns:
            tuple[bool, reason]: (should_continue, reason)
        """
        if len(self._audit_history) >= MAX_AUDIT_ITERATIONS:
            return False, f"Max iterations ({MAX_AUDIT_ITERATIONS}) reached"
        
        if len(self._audit_history) >= 2:
            improvement = (
                self._audit_history[-1] - self._audit_history[-2]
            ) / max(1, self._audit_history[-2])
            
            if improvement < IMPROVEMENT_THRESHOLD:
                return False, f"Improvement ({improvement:.1%}) below threshold ({IMPROVEMENT_THRESHOLD:.0%})"
        
        return True, "Continue iteration"
    
    def reset_iteration_history(self) -> None:
        """Reinicia historial de iteraciones."""
        with self._lock:
            self._audit_history = []
    
    async def full_audit_cycle(
        self,
        local_code: str,
        benchmark_code: str,
        local_file: str = "unknown",
        benchmark_repo: str = "unknown",
        language: str = "python",
    ) -> list[AuditReport]:
        """
        Ejecuta ciclo completo de auditoría con iteraciones.
        
        Continúa iterando hasta aprobar, alcanzar max iteraciones,
        o estancarse.
        
        Returns:
            list[AuditReport]: Todos los reportes del ciclo
        """
        self.reset_iteration_history()
        reports = []
        iteration = 1
        
        while True:
            report = await self.audit_code(
                local_code=local_code,
                benchmark_code=benchmark_code,
                local_file=local_file,
                benchmark_repo=benchmark_repo,
                language=language,
                iteration=iteration,
            )
            
            reports.append(report)
            report.save()
            
            # Si aprobado, terminar
            if report.verdict == AuditVerdict.APPROVED:
                logger.info(f"Code APPROVED after {iteration} iteration(s)")
                break
            
            # Verificar si continuar
            should_continue, reason = self.should_continue_iteration()
            if not should_continue:
                logger.info(f"Stopping iterations: {reason}")
                break
            
            iteration += 1
            
            # En producción, aquí se aplicarían los fixes sugeridos
            # Por ahora solo informamos
            logger.info(f"Iteration {iteration}: Applying suggested fixes...")
        
        return reports
    
    def get_status(self) -> dict:
        """Retorna estado del auditor."""
        return {
            "stats": dict(self._stats),
            "current_iteration": len(self._audit_history),
            "max_iterations": MAX_AUDIT_ITERATIONS,
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "score_history": list(self._audit_history),
        }
    
    def get_latest_fitness(self) -> Optional[float]:
        """Retorna el último fitness score."""
        if self._audit_history:
            return self._audit_history[-1]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_evolution_instance: Optional[EvolutionAuditor] = None
_evolution_lock = Lock()


def get_evolution() -> EvolutionAuditor:
    """Obtiene la instancia singleton del Evolution."""
    global _evolution_instance
    with _evolution_lock:
        if _evolution_instance is None:
            _evolution_instance = EvolutionAuditor()
        return _evolution_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def _test_evolution():
    """Test básico del Evolution."""
    evolution = get_evolution()
    
    print("=" * 60)
    print("ULTRAGENT EVOLUTION v0.1 - Test")
    print("=" * 60)
    print(f"Status: {evolution.get_status()}")
    print("=" * 60)
    
    # Código de ejemplo
    local_code = '''
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''
    
    benchmark_code = '''
from typing import List, TypeVar

T = TypeVar('T', int, float)

def process_data(data: List[T]) -> List[T]:
    """Process positive values by doubling them.
    
    Args:
        data: Input list of numeric values
        
    Returns:
        List of doubled positive values
        
    Raises:
        TypeError: If data contains non-numeric values
    """
    if not isinstance(data, list):
        raise TypeError("Expected a list")
    
    return [item * 2 for item in data if item > 0]
'''
    
    print("\nEjecutando auditoría...")
    report = await evolution.audit_code(
        local_code=local_code,
        benchmark_code=benchmark_code,
        local_file="example.py",
        benchmark_repo="gold-standard/example",
    )
    
    print(f"\nResultado:")
    print(f"  Fitness: {report.scorecard.total:.1f}")
    print(f"  Grade: {report.scorecard.grade.value}")
    print(f"  Verdict: {report.verdict.value}")
    print(f"  Issues: {len(report.critical_issues)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    asyncio.run(_test_evolution())
