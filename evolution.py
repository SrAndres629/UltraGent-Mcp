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
import time

from librarian import CodeLibrarian
from router import OmniRouter
from scout import get_scout, SearchResult

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


@dataclass
class GapAnalysis:
    """Análisis de brechas entre implementación local y remota."""
    feature_name: str
    local_score: float
    remote_score: float
    missing_elements: list[str]
    structural_differences: list[str]
    adaptation_plan: str


@dataclass
class ResearchReport:
    """Reporte de investigación proactiva."""
    query: str
    references: list[SearchResult]
    recommendation: str
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "references": [r.to_dict() for r in self.references],
            "recommendation": self.recommendation
        }


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
        
        self.router = OmniRouter() # Instancia de router
        self.librarian = CodeLibrarian() # Integración con Librarian
        
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
            
            result = await router.route_task(
                task_type="strategic",  # Usar tier strategic (Kimi/Gemini)
                payload=prompt,
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

    async def analyze_gap(self, local_code: str, remote_code: str, feature_context: str) -> GapAnalysis:
        """
        Analiza la brecha entre una implementación local y una remota (Gold Standard).
        """
        prompt = f"""
        Actúa como un Arquitecto de Software Senior realizando un Gap Analysis.
        Compararás una implementación local con un Gold Standard remoto para la funcionalidad '{feature_context}'.
        
        LOCAL CODE:
        ```python
        {local_code[:2000]}
        ```
        
        REMOTE GOLD STANDARD:
        ```python
        {remote_code[:2000]}
        ```
        
        Analiza:
        1. Qué elementos estructurales faltan en local.
        2. Diferencias en manejo de errores, typing y patrones.
        3. Puntuación comparativa (0-10) para cada un.
        
        Responde estrictamente en JSON:
        {{
            "local_score": float,
            "remote_score": float,
            "missing_elements": ["elem1", "elem2"],
            "structural_differences": ["diff1", "diff2"],
            "adaptation_plan": "Pasos para adaptar lo mejor del remoto al local"
        }}
        """
        
        try:
            response = await self.router.route_task(
                task_type="coding", # Usar modelo de código fuerte
                payload=prompt,
                system_prompt="Eres un experto en análisis de diferencias de código."
            )
            
            content = response.get("content", "{}")
            # Limpieza básica de markdown json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            data = json.loads(content)
            
            return GapAnalysis(
                feature_name=feature_context,
                local_score=data.get("local_score", 0.0),
                remote_score=data.get("remote_score", 0.0),
                missing_elements=data.get("missing_elements", []),
                structural_differences=data.get("structural_differences", []),
                adaptation_plan=data.get("adaptation_plan", "")
            )
            
        except Exception as e:
            logger.error(f"Error en Gap Analysis: {e}")
            return GapAnalysis("Error", 0, 0, [], [], str(e))

    async def adapt_code(self, remote_code: str, target_file: str) -> str:
        """
        Adapta código remoto para que encaje en el proyecto local.
        Usa el contexto del archivo destino para mantener el estilo.
        """
        # Intentar leer el archivo destino para tener contexto de estilo
        local_context = ""
        try:
            path = Path(target_file)
            if path.exists():
                local_context = path.read_text(encoding="utf-8")[:1000]
        except:
            pass

        prompt = f"""
        Actúa como un Ingeniero de Adaptación de Código.
        Tu tarea es reescribir el siguiente CÓDIGO REMOTO para que se integre perfectamente en nuestro proyecto existente.
        
        CONTEXTO LOCAL (Estilo, imports, stack):
        ```python
        {local_context}
        ```
        
        CÓDIGO REMOTO A ADAPTAR:
        ```python
        {remote_code}
        ```
        
        Instrucciones:
        1. Mantén la lógica robusta del remoto.
        2. Usa las librerías y patrones del contexto local (ej: si usamos Pydantic v2, úsalo).
        3. Elimina dependencias raras no presentes en el local.
        4. Asegura docstrings en español.
        5. Retorna SOLO el código Python adaptado.
        """
        
        response = await self.router.route_task(
            task_type="coding",
            payload=prompt,
            system_prompt="Escribe código Python production-ready adaptado al stack existente."
        )
        
        adapted_code = response.get("content", "")
        # Limpiar bloques de código
        if "```python" in adapted_code:
            adapted_code = adapted_code.split("```python")[1].split("```")[0]
        elif "```" in adapted_code:
            adapted_code = adapted_code.split("```")[1].split("```")[0]
            
        return adapted_code.strip()

    async def proactive_research(self, task_description: str) -> ResearchReport:
        """
        Investiga proactivamente referencias externas para una tarea.
        
        Args:
            task_description: Descripción a investigar
            
        Returns:
            ResearchReport con referencias y recomendación inicial.
        """
        scout = get_scout()
        
        # 0. Detección de URL de GitHub (Smart GitHub Mode)
        import re
        github_match = re.search(r"https?://github\.com/([\w\-\.]+)/([\w\-\.]+)", task_description)
        results = []
        
        if github_match:
            owner, repo = github_match.groups()
            repo_full_name = f"{owner}/{repo}"
            logger.info(f"Smart GitHub Mode: Detected {repo_full_name}")
            
            # Intentar obtener README directamente
            # Primero listamos estructura para encontrar el nombre exacto del readme
            structure = await scout.get_repository_structure(repo_full_name)
            readme_path = None
            
            if "files" in structure:
                for f in structure["files"]:
                    if f["name"].lower().startswith("readme"):
                        readme_path = f["path"]
                        break
            
            if readme_path:
                readme_res = await scout.download_file_content(repo_full_name, readme_path)
                if readme_res.success:
                    results.append(SearchResult(
                        title=f"{repo_full_name} README",
                        url=f"https://github.com/{repo_full_name}",
                        snippet=readme_res.data[:2000], # Limitar tamaño
                        source_type="ARCHITECTURE",
                        date=datetime.now().strftime("%Y-%m-%d"),
                    ))
        
        # 1. Búsqueda Universal (Solo si no encontramos nada directo o si queremos complementar)
        # Si ya tenemos el README, DuckDuckGo es opcional, pero podemos buscar referencias extra.
        # Para evitar el error de timeout/vacío, si ya tenemos GitHub, saltamos búsqueda web compleja.
        
        if not results:
             # Fallback a búsqueda web si no hay URL o falló GitHub
             results = await scout.universal_search(
                query=task_description,
                max_results=3,
                modernity_years=2
            )

        
        if not results:
             return ResearchReport(task_description, [], "No relevant external references found.")
             
        # 2. Análisis preliminar de recomendación
        snippets = "\n\n".join([f"Source: {r.url}\nType: {r.source_type}\n{r.snippet[:500]}..." for r in results])
        
        prompt = f"""
        Actúa como un Consultor Estratégico de Software.
        Analiza estos resultados de búsqueda para la tarea: "{task_description}"
        
        RESULTADOS:
        {snippets}
        
        Dictamina si debemos:
        1. CLONAR_ADAPTAR: Si hay soluciones robustas existentes (GitHub).
        2. COPIAR_SNIPPET: Si es un problema puntual resuelto en SO.
        3. CONSTRUIR_CERO: Si no hay buenas referencias.
        
        Tu respuesta debe ser una recomendación ejecutiva de 2 líneas.
        """
        
        response = await self.router.route_task(
            task_type="strategic",
            payload=prompt
        )
        
        recommendation = getattr(response, "content", "Review references manually.").strip()
        
        return ResearchReport(task_description, results, recommendation)

    async def compare_and_recommend(self, local_code: str, external_ref: str) -> dict:
        """
        Compara código local con una referencia externa y recomienda acción.
        
        Args:
            local_code: Implementación actual (o vacía)
            external_ref: Código o URL referencia
            
        Returns:
            dict: Reporte diferencial con Verdict, Advantages, AdaptationCost
        """
        prompt = f"""
        Actúa como Senior Architect.
        Compara la implementación Local vs Referencia Externa.
        
        LOCAL:
        {local_code[:1000] if local_code else "(No implemented yet)"}
        
        REF EXTERNA:
        {external_ref[:2000]}
        
        Genera un JSON con:
        - "advantages_ref": Lista de ventajas de la referencia.
        - "adaptation_cost": Estimación de esfuerzo (Low/Medium/High).
        - "verdict": "ADOPT_REF" o "KEEP_LOCAL".
        - "reasoning": Explicación breve.
        """ 
        
        result = await self.router.route_task("strategic", prompt)
        
        # Simple JSON extract
        content = result.get("content", "{}")
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
            
        try:
             return json.loads(content)
        except:
             return {"error": "Failed to parse analysis", "raw": content}
    
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
        
        # 1. ANÁLISIS DETERMINISTA (The Hard Critic)
        # Calculamos hechos matemáticos antes de pedir opiniones.
        from metrics import MetricsEngine
        
        try:
            hard_metrics = MetricsEngine.analyze_code(local_code, local_file)
            metrics_report = f"""
            ### DETERMINISTIC METRICS (Hard Facts):
            - Cyclomatic Complexity: {hard_metrics.cyclomatic_complexity}
            - Maintainability Index: {hard_metrics.maintainability_index}
            - Grade: {hard_metrics.grade}
            - Logical LOC: {hard_metrics.loc}
            
            ### FUNCTION SCORES:
            """ + "\n".join([f"- {f.name}: Complexity={f.complexity}, Grade={f.grade}" for f in hard_metrics.functions])
            
            # CRITICAL GUARD: Fast-Fail if code is trash
            if hard_metrics.grade == "F":
                logger.warning(f"⛔ REJECTED BY METRICS: {local_file} is unmaintainable (Grade F).")
                # Create a synthetic report rejecting it immediately? 
                # Better to include it in the LLM prompt so the LLM knows WHY it sucks.
        except Exception as e:
            logger.error(f"Metrics Engine Failed: {e}")
            metrics_report = "Metrics unavailable (Parse Error)"

        # 2. Análisis LLM (Comparative)
        # Inyectamos las métricas reales para que el LLM no alucine.
        # Construir prompt
        base_prompt = AUDIT_PROMPT_TEMPLATE.format(
            language=language,
            local_code=local_code[:8000],  # Limitar tamaño
            benchmark_code=benchmark_code[:8000],
        )
        
        # Inject metrics at the end of the user prompt
        # We enforce the "Hard Facts" by placing them as "Context"
        final_prompt = f"{base_prompt}\n\n{metrics_report}\n\nINSTRUCTION: You MUST align your verdict with the DETERMINISTIC METRICS above. If Grade is F, you MUST REJECT."
        
        # Llamar al router
        result = await self._call_router(final_prompt)
        
        if "error" in result and "scores" not in result:
            # Error en la llamada
            logger.error(f"Audit error: {result.get('error')}")
            result = await self._fallback_analysis(final_prompt)
        
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
        Ejecuta ciclo completo de auditoría con iteraciones y AUTO-FIX.
        
        Continúa iterando hasta aprobar, alcanzar max iteraciones,
        o estancarse.
        """
        self.reset_iteration_history()
        reports = []
        iteration = 1
        
        from cortex import get_cortex
        from mechanic import get_mechanic
        
        # 1. Active Recall: Verificar historial previo
        history_memories = get_cortex().get_related_memories(f"audit_history {local_file}")
        if history_memories:
            logger.info(f"📜 Found previous audit context for {local_file}")

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
            
            # 2. Active Learning: Persistir estado de salud
            try:
                get_cortex().add_memory(
                    content=f"Audit {local_file} (Iter {iteration}): Grade {report.scorecard.grade.value} (Fitness {report.scorecard.total}). Verdict: {report.verdict.value}",
                    tags=["audit_log", "health_check", f"file:{local_file}"],
                    importance=0.8 if report.verdict == AuditVerdict.REJECTED else 0.3
                )
            except Exception as e:
                logger.warning(f"Failed to save audit memory: {e}")

            # 2. Functional Verification (practitioner check)
            # Only skip if the theoretical audit already REJECTED the code for being trash.
            if report.verdict != AuditVerdict.REJECTED:
                logger.info(f"Triggering Functional Verification for {local_file}...")
                
                from tester import get_tester
                tester = get_tester()
                
                # Resolve absolute path
                path_obj = Path(local_file)
                if not path_obj.is_absolute():
                     # Try to resolve relative to PROJECT_ROOT (which is where evolution.py is)
                     # or better, just use the provided path and hope for the best if absolute
                     path_obj = PROJECT_ROOT / local_file
                
                if not path_obj.exists():
                    # Fallback if PROJECT_ROOT is inside .ai or somewhere else
                    path_obj = Path.cwd() / local_file

                test_passed = await tester.verify_file(path_obj)
                
                if test_passed:
                    logger.info(f"✅ Functional Verification PASSED for {local_file}")
                    # If it was NEEDS_WORK but tests passed, we might still want to fix it 
                    # for architecture reasons, so we respect the original verdict if it's not APPROVED.
                else:
                    logger.error(f"❌ Functional Verification FAILED for {local_file}")
                    # If tests failed, the grade is effectively lowered and it MUST BE REPAIRED.
                    report.verdict = AuditVerdict.NEEDS_WORK
                    report.critical_issues.append(CriticalIssue(
                        category="ARCHITECTURE",
                        severity="CRITICAL",
                        description="Code logic failed automated unit tests.",
                        solution="The logic is functionally broken. Refer to Pytest logs and fix the implementation."
                    ))

            # Si aprobado en teoría y práctica, terminar
            if report.verdict == AuditVerdict.APPROVED:
                logger.info(f"Code APPROVED after {iteration} iteration(s)")
                break
            # Verificar si continuar
            should_continue, reason = self.should_continue_iteration()
            if not should_continue:
                logger.info(f"Stopping iterations: {reason}")
                break
            
            # 3. Active Engineering: Aplicar correcciones (Auto-Fix)
            mechanic = get_mechanic()
            if mechanic and mechanic.is_available:
                logger.info(f"🔧 Iteration {iteration}: Mechanic is patching critical issues...")
                
                fixes_applied = 0
                for issue in report.critical_issues:
                    if issue.severity in ["CRITICAL", "MAJOR"] and issue.solution:
                        # Usar el editor inteligente
                        # Nota: Esto es arriesgado, en producción real se necesitaría revisión humana
                        # o un AST parser muy robusto. Por ahora usamos apply_patch que es search/replace.
                         
                        # Necesitamos identificar qué reemplazar. El LLM debería darnos el 'target' exacto.
                        # Como la estructura actual de 'issue' no garantiza un diff limpio,
                        # usaremos el Mechanic LLM para generar el patch correcto basado en la solución.
                        
                        patch_task = f"Fix issue in {local_file}: {issue.description}. Solution: {issue.solution}"
                        res = await mechanic.run_task(task=patch_task, max_steps=3)
                        
                        if "Done" in res or "success" in res.lower():
                            fixes_applied += 1
                
                if fixes_applied > 0:
                    # Recargar código para la siguiente iteración
                    if Path(local_file).exists():
                        local_code = Path(local_file).read_text(encoding="utf-8")
                else:
                    logger.warning("Mechanic could not apply automated fixes.")
                    
            iteration += 1
        
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
