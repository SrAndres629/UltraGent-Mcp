"""
ULTRAGENT MCP SERVER v2.0
=========================
Servidor MCP para el Sistema de Ingeniería Híbrida Autónoma.

Módulos integrados:
- Core: sync_status, get_memory, get_hud
- Sentinel: filesystem watcher
- Router: 4-tier LLM economy
- Librarian: Tree-sitter + ChromaDB
- Scout/Evolution: GitHub + Fitness Scorecard
- Mechanic/Vision: Docker sandbox + NetworkX
- HUD: Panel de control maestro

Total: 24 tools
"""

import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
MEMORY_FILE = AI_DIR / "memory.md"
HUD_FILE = AI_DIR / "HUD.md"
TASKS_DB = AI_DIR / "tasks.db"
LOGS_DIR = AI_DIR / "logs"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CON FILTRO DE SECRETOS
# ═══════════════════════════════════════════════════════════════════════════════

class SecretFilter(logging.Filter):
    """Filtra información sensible de los logs (API keys, tokens)."""

    PATTERNS = [
        r"sk-[a-zA-Z0-9_-]+",      # OpenAI/Anthropic keys
        r"nvapi-[a-zA-Z0-9_-]+",   # NVIDIA NIM keys
        r"gsk_[a-zA-Z0-9_-]+",     # Groq keys
        r"ghp_[a-zA-Z0-9]+",       # GitHub tokens
        r"sf_[a-zA-Z0-9_-]+",      # SiliconFlow keys
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg"):
            msg = str(record.msg)
            for pattern in self.PATTERNS:
                msg = re.sub(pattern, "[REDACTED]", msg)
            record.msg = msg
        return True


def setup_logging() -> logging.Logger:
    """Configura el sistema de logging con archivo rotativo."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "bootstrap.log"
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    # Formato profesional
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de archivo
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SecretFilter())

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SecretFilter())

    # Logger principal
    logger = logging.getLogger("ultragent")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# BASE DE DATOS (THREAD-SAFE)
# ═══════════════════════════════════════════════════════════════════════════════

def get_db_connection() -> sqlite3.Connection:
    """
    Obtiene una conexión thread-safe a la base de datos SQLite.
    Configura WAL mode para mejor concurrencia.
    """
    conn = sqlite3.connect(
        str(TASKS_DB),
        timeout=30.0,
        isolation_level="DEFERRED",
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# FASTMCP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="ultragent",
    instructions="""
    Ultragent MCP Server - Sistema de Ingeniería Híbrida Autónoma.
    
    Este servidor proporciona acceso al Core .ai/ que contiene:
    - memory.md: Memoria episódica con decisiones arquitectónicas
    - HUD.md: Panel de observabilidad del sistema
    - tasks.db: Base de datos de tareas y estado
    
    Usa sync_status para verificar el estado actual del sistema.
    """,
)


@mcp.tool()
def sync_status() -> dict:
    """
    Sincroniza y retorna el estado actual del Core .ai/.
    
    Lee los archivos de memoria y retorna un resumen del estado del sistema,
    incluyendo las últimas decisiones arquitectónicas y el estado de los protocolos.
    
    Returns:
        dict: Estado del sistema con campos:
            - status: "online" | "degraded" | "offline"
            - memory_summary: Resumen de memory.md
            - protocols_loaded: Número de protocolos asimilados
            - last_sync: Timestamp de sincronización
            - hud_available: Si HUD.md está accesible
    """
    logger.info("sync_status invocado - iniciando sincronización")
    
    result = {
        "status": "offline",
        "memory_summary": None,
        "protocols_loaded": 0,
        "last_sync": datetime.now().isoformat(),
        "hud_available": False,
        "ai_dir_exists": AI_DIR.exists(),
    }
    
    try:
        # Verificar existencia del Core
        if not AI_DIR.exists():
            logger.warning(f"Core .ai/ no encontrado en {AI_DIR}")
            return result
        
        result["status"] = "degraded"
        
        # Leer memory.md
        if MEMORY_FILE.exists():
            content = MEMORY_FILE.read_text(encoding="utf-8")
            
            # Extraer decisiones arquitectónicas
            decisions = re.findall(r"### Decisión Arquitectónica #(\d+):", content)
            result["protocols_loaded"] = len(decisions)
            
            # Extraer las primeras 500 caracteres como resumen
            result["memory_summary"] = content[:500] + "..." if len(content) > 500 else content
            logger.info(f"memory.md leído: {len(decisions)} decisiones encontradas")
        
        # Verificar HUD
        if HUD_FILE.exists():
            result["hud_available"] = True
            logger.info("HUD.md disponible")
        
        # Verificar base de datos
        if TASKS_DB.exists():
            try:
                conn = get_db_connection()
                cursor = conn.execute("SELECT COUNT(*) FROM protocol_log")
                count = cursor.fetchone()[0]
                conn.close()
                result["protocols_in_db"] = count
                logger.info(f"tasks.db: {count} protocolos registrados")
            except Exception as e:
                logger.error(f"Error leyendo tasks.db: {e}")
                result["db_error"] = str(e)
        
        result["status"] = "online"
        logger.info("sync_status completado - sistema online")
        
    except Exception as e:
        logger.error(f"Error en sync_status: {e}")
        result["error"] = str(e)
    
    return result


@mcp.tool()
def get_memory() -> str:
    """
    Retorna el contenido completo de memory.md.
    
    Útil para que el Arquitecto recupere el contexto completo
    de las decisiones arquitectónicas anteriores.
    
    Returns:
        str: Contenido de memory.md o mensaje de error
    """
    logger.info("get_memory invocado")
    
    if not MEMORY_FILE.exists():
        logger.warning("memory.md no existe")
        return "ERROR: memory.md no encontrado. Ejecutar setup_fs.py"
    
    content = MEMORY_FILE.read_text(encoding="utf-8")
    logger.info(f"get_memory: {len(content)} bytes retornados")
    return content


@mcp.tool()
def get_hud() -> str:
    """
    Retorna el contenido completo de HUD.md.
    
    El HUD (Heads-Up Display) muestra el estado en tiempo real
    de los lóbulos, APIs externas y protocolos.
    
    Returns:
        str: Contenido de HUD.md o mensaje de error
    """
    logger.info("get_hud invocado")
    
    if not HUD_FILE.exists():
        logger.warning("HUD.md no existe")
        return "ERROR: HUD.md no encontrado. Ejecutar setup_fs.py"
    
    content = HUD_FILE.read_text(encoding="utf-8")
    logger.info(f"get_hud: {len(content)} bytes retornados")
    return content


@mcp.tool()
def get_sentinel_status() -> dict:
    """
    Retorna el estado actual del módulo Sentinel.
    
    Incluye información sobre:
    - Si el Sentinel está corriendo
    - Directorio monitoreado
    - Número de eventos procesados
    - Últimos 5 eventos detectados
    
    Returns:
        dict: Estado del Sentinel con campos:
            - running: bool
            - watch_path: str
            - events_processed: int
            - uptime_seconds: float
            - recent_events: list[dict]
    """
    logger.info("get_sentinel_status invocado")
    
    try:
        from sentinel import get_sentinel
        sentinel = get_sentinel()
        status = sentinel.get_status()
        logger.info(f"Sentinel status: running={status['running']}, events={status['events_processed']}")
        return status
    except ImportError as e:
        logger.error(f"Sentinel no disponible: {e}")
        return {
            "error": "Sentinel module not found",
            "running": False,
            "message": "Ejecutar: uv sync para instalar dependencias"
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del Sentinel: {e}")
        return {
            "error": str(e),
            "running": False,
        }


@mcp.tool()
def clear_sentinel_signals() -> dict:
    """
    Limpia los eventos pendientes en signals.json.
    
    Útil para resetear el estado después de procesar
    todos los eventos pendientes.
    
    Returns:
        dict: Resultado de la operación
    """
    logger.info("clear_sentinel_signals invocado")
    
    try:
        from sentinel import get_sentinel
        sentinel = get_sentinel()
        sentinel.clear_signals()
        return {"success": True, "message": "signals.json limpiado"}
    except Exception as e:
        logger.error(f"Error limpiando signals: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# OMNI-ROUTER TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_router_status() -> dict:
    """
    Retorna el estado actual del Omni-Router.
    
    Incluye información sobre:
    - Estadísticas de llamadas (total, exitosas, fallidas, failovers)
    - Estado de cada proveedor (Circuit Breaker)
    - Consumo de tokens por tier
    
    Returns:
        dict: Estado del router con stats, providers y budget
    """
    logger.info("get_router_status invocado")
    
    try:
        from router import get_router
        router = get_router()
        status = router.get_status()
        logger.info(f"Router status: {status['stats']}")
        return status
    except ImportError as e:
        logger.error(f"Router no disponible: {e}")
        return {
            "error": "Router module not found",
            "message": "Ejecutar: uv sync para instalar dependencias"
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del Router: {e}")
        return {"error": str(e)}


@mcp.tool()
def route_task(task_type: str, payload: str, system_prompt: str = "") -> dict:
    """
    Enruta una tarea al tier apropiado con failover automático.
    
    El Router clasifica automáticamente la tarea y selecciona
    el proveedor óptimo basado en complejidad/costo.
    
    Tiers disponibles:
    - SPEED: fix_syntax, unit_test, boilerplate, quick_question
    - CODING: generate_code, refactor, implement_feature, debug
    - VISUAL: analyze_image, diagram_to_code, swarm
    - STRATEGIC: architecture_review, security_audit, final_review
    
    Args:
        task_type: Tipo de tarea (generate_code, fix_syntax, etc.)
        payload: Contenido/prompt de la tarea
        system_prompt: Prompt de sistema opcional
        
    Returns:
        dict: Resultado con success, content, provider, tier, tokens_used, latency_ms
    """
    logger.info(f"route_task invocado: {task_type}")
    
    try:
        import asyncio
        from router import get_router
        
        router = get_router()
        
        # Ejecutar async en sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                router.route_task(
                    task_type=task_type,
                    payload=payload,
                    system_prompt=system_prompt if system_prompt else None,
                )
            )
        finally:
            loop.close()
        
        return {
            "success": result.success,
            "content": result.content,
            "provider": result.provider,
            "tier": result.tier.value,
            "tokens_used": result.tokens_used,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }
        
    except ImportError as e:
        logger.error(f"Router no disponible: {e}")
        return {"success": False, "error": "Router module not found"}
    except Exception as e:
        logger.error(f"Error en route_task: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def ask_swarm(task: str, subtasks: list, system_prompt: str = "") -> dict:
    """
    Procesa múltiples subtareas en paralelo usando Swarms.
    
    El Router distribuye las subtareas entre diferentes tiers
    según su complejidad, ejecutándolas concurrentemente.
    
    Args:
        task: Descripción general de la tarea principal
        subtasks: Lista de subtareas a procesar
        system_prompt: Prompt de sistema compartido
        
    Returns:
        dict: Resultados con lista de respuestas y resumen
    """
    logger.info(f"ask_swarm invocado: {len(subtasks)} subtareas")
    
    try:
        import asyncio
        from router import get_router
        
        router = get_router()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                router.ask_swarm(
                    task=task,
                    subtasks=subtasks,
                    system_prompt=system_prompt if system_prompt else None,
                )
            )
        finally:
            loop.close()
        
        return {
            "success": True,
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "results": [
                {
                    "success": r.success,
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "provider": r.provider,
                    "tier": r.tier.value,
                    "tokens_used": r.tokens_used,
                    "error": r.error,
                }
                for r in results
            ],
        }
        
    except ImportError as e:
        logger.error(f"Router no disponible: {e}")
        return {"success": False, "error": "Router module not found"}
    except Exception as e:
        logger.error(f"Error en ask_swarm: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_token_usage() -> dict:
    """
    Retorna el consumo de tokens por tier.
    
    Útil para monitorear el presupuesto de la sesión
    y el uso por cada tier de inteligencia.
    
    Returns:
        dict: total_used, limit, remaining, by_tier
    """
    logger.info("get_token_usage invocado")
    
    try:
        from router import get_router
        router = get_router()
        return router.get_token_usage()
    except Exception as e:
        logger.error(f"Error en get_token_usage: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARIAN TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_code(
    query: str,
    n_results: int = 5,
    language: str = "",
    node_type: str = "",
) -> dict:
    """
    Búsqueda semántica en el código usando lenguaje natural.
    
    Utiliza embeddings GPU para encontrar funciones, clases y métodos
    relevantes basados en la descripción semántica de la query.
    
    Args:
        query: Consulta en lenguaje natural (ej: "función que valida emails")
        n_results: Número máximo de resultados (default: 5)
        language: Filtrar por lenguaje (python, javascript, typescript)
        node_type: Filtrar por tipo (function, class, method, interface)
        
    Returns:
        dict: Resultados con name, signature, file_path, line, relevance
    """
    logger.info(f"search_code invocado: '{query[:50]}'")
    
    try:
        from librarian import get_librarian
        librarian = get_librarian()
        
        results = librarian.semantic_search(
            query=query,
            n_results=n_results,
            language=language if language else None,
            node_type=node_type if node_type else None,
        )
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
        
    except ImportError as e:
        logger.error(f"Librarian no disponible: {e}")
        return {"success": False, "error": "Librarian module not found"}
    except Exception as e:
        logger.error(f"Error en search_code: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_file_skeleton(file_path: str) -> dict:
    """
    Obtiene el esqueleto estructural de un archivo de código.
    
    Extrae firmas de funciones, clases y métodos SIN el cuerpo
    de las implementaciones. Útil para análisis rápido de arquitectura.
    
    Lenguajes soportados: .py, .js, .jsx, .ts, .tsx
    
    Args:
        file_path: Ruta absoluta al archivo
        
    Returns:
        dict: Esqueleto con elements (name, type, signature, docstring, line)
    """
    logger.info(f"get_file_skeleton invocado: {file_path}")
    
    try:
        from librarian import get_librarian
        librarian = get_librarian()
        
        skeleton = librarian.get_file_skeleton(file_path)
        return skeleton
        
    except ImportError as e:
        logger.error(f"Librarian no disponible: {e}")
        return {"error": "Librarian module not found", "file": file_path}
    except Exception as e:
        logger.error(f"Error en get_file_skeleton: {e}")
        return {"error": str(e), "file": file_path}


@mcp.tool()
def index_file(file_path: str) -> dict:
    """
    Indexa un archivo en la biblioteca de código.
    
    Extrae esqueletos (firmas + docstrings) y los almacena
    con embeddings vectoriales para búsqueda semántica.
    
    Args:
        file_path: Ruta absoluta al archivo
        
    Returns:
        dict: Resultado con skeletons indexados
    """
    logger.info(f"index_file invocado: {file_path}")
    
    try:
        from librarian import get_librarian
        librarian = get_librarian()
        return librarian.index_file(file_path)
    except Exception as e:
        logger.error(f"Error en index_file: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_librarian_status() -> dict:
    """
    Retorna el estado del Librarian.
    
    Incluye:
    - Número de esqueletos indexados
    - Estadísticas de uso
    - Lenguajes soportados
    - Estado del modelo de embeddings
    
    Returns:
        dict: Estado del Librarian
    """
    logger.info("get_librarian_status invocado")
    
    try:
        from librarian import get_librarian
        librarian = get_librarian()
        return librarian.get_status()
    except Exception as e:
        logger.error(f"Error en get_librarian_status: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SCOUT / EVOLUTION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def search_github_repos(
    query: str,
    language: str = "python",
    min_stars: int = 500,
    max_results: int = 5,
) -> dict:
    """
    Busca repositorios Gold Standard en GitHub.
    
    Filtra por métricas de salud: estrellas, forks, actividad,
    presencia de tests y typing.
    
    Args:
        query: Términos de búsqueda (ej: "fastapi template")
        language: Lenguaje de programación
        min_stars: Mínimo de estrellas requeridas
        max_results: Número máximo de resultados
        
    Returns:
        dict: Repositorios con health_score y is_gold_standard
    """
    logger.info(f"search_github_repos invocado: '{query}'")
    
    try:
        from scout import get_scout
        scout = get_scout()
        
        repos = await scout.search_repositories(
            query=query,
            language=language,
            min_stars=min_stars,
            max_results=max_results,
        )
        
        return {
            "success": True,
            "query": query,
            "repos": [r.to_dict() for r in repos],
            "count": len(repos),
            "gold_standard_count": sum(1 for r in repos if r.is_gold_standard()),
        }
        
    except ImportError as e:
        logger.error(f"Scout no disponible: {e}")
        return {"success": False, "error": "Scout module not found"}
    except Exception as e:
        logger.error(f"Error en search_github_repos: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def benchmark_with_github(
    local_file: str,
    project_type: str = "api",
    language: str = "python",
) -> dict:
    """
    Realiza auditoría comparativa contra repositorios Gold Standard.
    
    1. Busca benchmarks en GitHub por tipo de proyecto
    2. Descarga esqueletos del mejor repo
    3. Ejecuta análisis comparativo con Evolution
    4. Genera reporte con Fitness Score
    
    Args:
        local_file: Ruta al archivo local a evaluar
        project_type: Tipo de proyecto (api, cli, web, library, agent)
        language: Lenguaje de programación
        
    Returns:
        dict: Reporte de auditoría con scorecard y issues
    """
    logger.info(f"benchmark_with_github invocado: {local_file}")
    
    try:
        from pathlib import Path
        from scout import get_scout
        from evolution import get_evolution
        
        # Leer código local
        local_path = Path(local_file)
        if not local_path.exists():
            return {"success": False, "error": f"File not found: {local_file}"}
        
        local_code = local_path.read_text(encoding="utf-8")
        
        # Buscar Gold Standards
        scout = get_scout()
        gold_repos = await scout.harvest_gold_standard(
            project_type=project_type,
            language=language,
        )
        
        if not gold_repos:
            return {
                "success": False,
                "error": "No Gold Standard repos found",
            }
        
        # Usar el mejor repo
        best_repo = gold_repos[0]
        
        # Descargar un archivo similar (README para ahora)
        readme_result = await scout.get_readme(best_repo.full_name)
        benchmark_code = readme_result.data if readme_result.success else ""
        
        # Si hay estructura, intentar descargar archivo similar
        structure = await scout.get_repository_structure(best_repo.full_name)
        for file in structure.get("files", []):
            if file["name"].endswith(".py") and "main" in file["name"].lower():
                file_result = await scout.download_file_content(
                    best_repo.full_name,
                    file["path"],
                )
                if file_result.success:
                    benchmark_code = file_result.data
                    break
        
        # Ejecutar auditoría
        evolution = get_evolution()
        report = await evolution.audit_code(
            local_code=local_code,
            benchmark_code=benchmark_code,
            local_file=str(local_path.name),
            benchmark_repo=best_repo.full_name,
            language=language,
        )
        
        # Guardar reporte
        report_path = report.save()
        
        return {
            "success": True,
            "benchmark_repo": best_repo.full_name,
            "benchmark_stars": best_repo.stars,
            "scorecard": report.scorecard.to_dict(),
            "verdict": report.verdict.value,
            "critical_issues": [i.to_dict() for i in report.critical_issues],
            "report_path": str(report_path),
        }
        
    except ImportError as e:
        logger.error(f"Scout/Evolution no disponible: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error en benchmark_with_github: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_scout_status() -> dict:
    """
    Retorna el estado del Scout.
    
    Incluye:
    - Si tiene token de GitHub configurado
    - Rate limit restante
    - Estadísticas de búsquedas y descargas
    
    Returns:
        dict: Estado del Scout
    """
    logger.info("get_scout_status invocado")
    
    try:
        from scout import get_scout
        scout = get_scout()
        return scout.get_status()
    except Exception as e:
        logger.error(f"Error en get_scout_status: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_evolution_status() -> dict:
    """
    Retorna el estado del Evolution Auditor.
    
    Incluye:
    - Auditorías realizadas
    - Historial de scores
    - Límites de iteración
    
    Returns:
        dict: Estado del Evolution
    """
    logger.info("get_evolution_status invocado")
    
    try:
        from evolution import get_evolution
        evolution = get_evolution()
        return evolution.get_status()
    except Exception as e:
        logger.error(f"Error en get_evolution_status: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANIC / VISION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def test_code_securely(
    code: str,
    requirements: list[str] | None = None,
    timeout: int = 30,
) -> dict:
    """
    Ejecuta código Python en un contenedor Docker aislado.
    
    El código se ejecuta con:
    - Límites de memoria (512MB) y CPU (50%)
    - Sin acceso a red ni al filesystem del host
    - Auto-destrucción del contenedor al terminar
    
    Args:
        code: Código Python a ejecutar
        requirements: Paquetes pip opcionales a instalar
        timeout: Tiempo máximo en segundos (default: 30, max: 120)
        
    Returns:
        dict: {success, stdout, stderr, execution_time}
    """
    logger.info(f"test_code_securely invocado (timeout={timeout}s)")
    
    try:
        from mechanic import get_mechanic
        mechanic = get_mechanic()
        
        if not mechanic.is_available:
            return {
                "success": False,
                "error": "Docker no disponible",
            }
        
        result = mechanic.run_in_sandbox(
            script=code,
            requirements=requirements,
            timeout=timeout,
        )
        
        return result.to_dict()
        
    except ImportError as e:
        logger.error(f"Mechanic no disponible: {e}")
        return {"success": False, "error": "Mechanic module not found"}
    except Exception as e:
        logger.error(f"Error en test_code_securely: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def visualize_architecture(
    output_filename: str | None = None,
) -> dict:
    """
    Genera un mapa visual de dependencias del proyecto.
    
    Analiza todos los archivos .py y genera un grafo PNG donde:
    - Verde: archivos del proyecto
    - Gris: dependencias externas
    - ROJO: ciclos de dependencias (problema!)
    
    Args:
        output_filename: Nombre del archivo (sin path). Default: auto-generado.
        
    Returns:
        dict: {success, graph_path, nodes, edges, cycles, hotspots}
    """
    logger.info("visualize_architecture invocado")
    
    try:
        from vision import get_vision, REPORTS_DIR
        from pathlib import Path
        
        vision = get_vision()
        
        output_path = None
        if output_filename:
            output_path = str(REPORTS_DIR / output_filename)
        
        report = vision.generate_dependency_graph(output_path=output_path)
        
        return {
            "success": True,
            "graph_path": report.graph_path,
            "node_count": len(report.nodes),
            "edge_count": len(report.edges),
            "cycles": report.cycles,
            "hotspots": report.hotspots,
        }
        
    except ImportError as e:
        logger.error(f"Vision no disponible: {e}")
        return {"success": False, "error": "Vision module not found"}
    except Exception as e:
        logger.error(f"Error en visualize_architecture: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_mechanic_status() -> dict:
    """
    Retorna el estado del Mechanic (Docker sandbox).
    
    Incluye:
    - Si Docker está disponible
    - Límites de recursos configurados
    - Estadísticas de ejecuciones
    
    Returns:
        dict: Estado del Mechanic
    """
    logger.info("get_mechanic_status invocado")
    
    try:
        from mechanic import get_mechanic
        mechanic = get_mechanic()
        return mechanic.get_status()
    except Exception as e:
        logger.error(f"Error en get_mechanic_status: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_vision_status() -> dict:
    """
    Retorna el estado del Vision (architecture mapper).
    
    Incluye:
    - Directorio de proyecto
    - Estadísticas de escaneos
    - Configuración de grafos
    
    Returns:
        dict: Estado del Vision
    """
    logger.info("get_vision_status invocado")
    
    try:
        from vision import get_vision
        vision = get_vision()
        return vision.get_status()
    except Exception as e:
        logger.error(f"Error en get_vision_status: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# HUD MANAGER TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_full_status() -> dict:
    """
    Retorna el estado completo del sistema desde el HUD.
    
    Incluye:
    - Estado de todos los módulos
    - Contenido del HUD.md
    - Misión actual
    - Uptime
    
    Returns:
        dict: Estado completo del sistema
    """
    logger.info("get_full_status invocado")
    
    try:
        from hud_manager import get_hud_manager
        hud = get_hud_manager()
        return hud.get_full_status()
    except Exception as e:
        logger.error(f"Error en get_full_status: {e}")
        return {"error": str(e)}


@mcp.tool()
def set_mission_goal(goal: str) -> dict:
    """
    Define el objetivo de la misión actual.
    
    Este objetivo se mostrará en la cabecera del HUD y
    servirá como guía para las decisiones del sistema.
    
    Args:
        goal: Descripción del objetivo (ej: "Implementar feature X")
        
    Returns:
        dict: Confirmación con el nuevo goal
    """
    logger.info(f"set_mission_goal invocado: {goal}")
    
    try:
        from hud_manager import get_hud_manager
        hud = get_hud_manager()
        hud.set_mission_goal(goal)
        hud.refresh_dashboard(force=True)
        
        return {
            "success": True,
            "mission_goal": goal,
        }
    except Exception as e:
        logger.error(f"Error en set_mission_goal: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def export_session(output_filename: str | None = None) -> dict:
    """
    Exporta la sesión completa como archivo ZIP.
    
    Incluye:
    - Memoria (.ai/memory.md)
    - Logs (.ai/logs/)
    - Reportes (.ai/reports/)
    - HUD (.ai/HUD.md)
    
    Args:
        output_filename: Nombre del archivo ZIP (opcional)
        
    Returns:
        dict: Path al archivo exportado
    """
    logger.info("export_session invocado")
    
    try:
        from hud_manager import get_hud_manager
        hud = get_hud_manager()
        
        output_path = None
        if output_filename:
            output_path = str(Path(output_filename))
        
        zip_path = hud.export_session(output_path)
        
        return {
            "success": True,
            "export_path": zip_path,
        }
    except Exception as e:
        logger.error(f"Error en export_session: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ULTRAGENT MCP SERVER v2.0 - Iniciando...")
    logger.info(f"AI_DIR: {AI_DIR}")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info(f"Tools: {len(mcp._tool_manager._tools)}")
    logger.info("=" * 60)
    
    # Iniciar servidor MCP
    mcp.run()
