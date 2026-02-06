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

import asyncio
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, List, Dict

from dotenv import load_dotenv
from fastmcp import FastMCP

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEOUT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOOL_TIMEOUT = 15.0  # 15 seconds max per tool operation

T = TypeVar('T')

async def with_timeout(
    coro: Any,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
    operation_name: str = "operation"
) -> dict:
    """
    Wraps an async coroutine with timeout protection.
    
    Returns:
        dict: {"success": True, "result": ...} on success
              {"success": False, "error": ..., "suggestion": ...} on failure
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return {"success": True, "result": result}
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Timeout after {timeout}s in {operation_name}",
            "suggestion": "Try with a simpler query or check API connectivity",
            "error_type": "TIMEOUT"
        }
    except Exception as e:
        error_msg = str(e)
        error_guide = get_error_guide(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "suggestion": error_guide["fix"],
            "error_category": error_guide["category"],
            "error_guide_url": error_guide["guide"]
        }

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PATHS (DYNAMIC CONTEXT)
# ═══════════════════════════════════════════════════════════════════════════════

# SERVER_DIR is where the actual code resides
SERVER_DIR = Path(__file__).parent
# Target project where data will be stored
PROJECT_ROOT = Path.cwd()

# Prioritize local .env but allow global fallback for core configurations
load_dotenv(PROJECT_ROOT / ".env")
if not os.getenv("GITHUB_TOKEN"):
    load_dotenv(SERVER_DIR / ".env")

AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
CACHE_DIR = AI_DIR / "cache" / "scout"
REPORTS_DIR = AI_DIR / "reports"
MEMORY_FILE = AI_DIR / "memory.md"
HUD_FILE = AI_DIR / "HUD.md"
TASKS_DB = AI_DIR / "tasks.db"
LOGS_DIR = AI_DIR / "logs"

# Ensure core directories exist in context
AI_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


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


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION CONTROL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
# (N/A - Unified in sync_system_status)


# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARIAN TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def semantic_search(
    action: str,
    query: str = "",
    file_path: str = "",
    n_results: int = 5,
    language: str = "",
    node_type: str = "",
    project_root: str = ""
) -> dict:
    """
    Descubrimiento Semántico y Arquitectónico de Código.
    
    Unifica la búsqueda por lenguaje natural, la inspección de esqueletos
    y el indexado de nuevos módulos.
    
    Acciones:
    - 'search': Búsqueda semántica (NLP) de lógica y funciones.
    - 'skeleton': Obtiene la estructura funcional de un archivo (firmas).
    - 'index': Indexa o re-escanea un archivo para la base de datos vectorial.
    
    Args:
        project_root: Optional path to target project directory.
    """
    logger.info(f"semantic_search: action={action}, query='{query}', file='{file_path}', project_root='{project_root}'")
    
    try:
        from librarian import get_librarian
        lib = get_librarian(project_root) if project_root else get_librarian()
        
        if action == "search":
            result = await with_timeout(
                asyncio.to_thread(
                    lib.semantic_search,
                    query=query, n_results=n_results, 
                    language=language if language else None,
                    node_type=node_type if node_type else None
                ),
                timeout=15.0,
                operation_name="semantic_search"
            )
            if not result["success"]: return result
            return {"success": True, "results": result["result"]}
            
        elif action == "skeleton":
            if not file_path: return {"error": "Se requiere 'file_path'"}
            result = await with_timeout(
                asyncio.to_thread(lib.get_file_skeleton, file_path),
                timeout=10.0,
                operation_name="get_file_skeleton"
            )
            if not result["success"]: return result
            return result["result"]
            
        elif action == "index":
            if not file_path: return {"error": "Se requiere 'file_path'"}
            result = await with_timeout(
                asyncio.to_thread(lib.index_file, file_path),
                timeout=25.0,
                operation_name="index_file"
            )
            if not result["success"]: return result
            return result["result"]

        elif action == "find_usage":
            if not query: return {"error": "Se requiere 'query' (nombre del símbolo)"}
            return lib.find_symbol_usage(query)

        elif action == "scan_debt":
            # Escaneo de deuda técnica en un archivo o directorio
            if not file_path: return {"error": "Se requiere 'file_path'"}
            path = Path(file_path)
            
            if path.is_file():
                return lib.scan_debt(file_path)
            elif path.is_dir():
                # Escaneo recursivo simple
                report = {"files_scanned": 0, "total_issues": 0, "details": []}
                for f in path.rglob("*.py"):
                    res = lib.scan_debt(str(f))
                    if res.get("issues"):
                        report["details"].append(res)
                        report["total_issues"] += res["issue_count"]
                    report["files_scanned"] += 1
                return report
            return {"error": "Path no válido"}
            
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en semantic_search: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CORTEX TOOLS (Memory Atoms)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def manage_cortex(
    action: str, 
    content: str = "", 
    tags: list[str] = None, 
    importance: float = 1.0,
    project_root: str = ""
) -> dict:
    """
    Gestión integral del Cortex (Memoria de Proyecto .ai).
    
    Acciones:
    - 'add': Guarda un nuevo átomo de memoria (recuerdo).
    - 'list': Retorna todos los recuerdos del proyecto.
    - 'read_raw': Retorna el contenido bruto de memory.md.
    - 'init': Inicializa la estructura cerebral en un nuevo proyecto.
    
    Args:
        project_root: Optional path to target project directory.
    """
    logger.info(f"manage_cortex: action={action}, project_root={project_root}")
    try:
        from cortex import get_cortex, AI_DIR
        ctx = get_cortex(project_root) if project_root else get_cortex()
        
        if action == "add":
            result = await with_timeout(
                asyncio.to_thread(ctx.add_memory, content, tags, importance),
                timeout=10.0,
                operation_name="cortex_add"
            )
            if not result["success"]: return result
            return {"success": True, "memory_id": result["result"]}
            
        elif action == "list":
            result = await with_timeout(
                asyncio.to_thread(ctx.get_all_memories),
                timeout=10.0,
                operation_name="cortex_list"
            )
            if not result["success"]: return result
            memories = result["result"]
            return {
                "success": True, 
                "memories": [
                    {"id": m.id, "content": m.content, "tags": m.tags, "importance": m.importance, "created_at": str(m.created_at)} for m in memories
                ]
            }
        elif action == "related": # NEW ACTION
            if not content: return {"error": "Se requiere 'content' (node_name)"}
            memories = ctx.get_related_memories(content)
            return {
                "success": True,
                "memories": [
                    {"content": m.content, "tags": m.tags, "importance": m.importance} for m in memories
                ]
            }
        elif action == "search":
            from librarian import get_librarian
            # Cross-module usage for semantic search in memory
            result = await with_timeout(
                asyncio.to_thread(get_librarian(project_root).semantic_search, query=content, n_results=5, node_type="memory"),
                timeout=15.0,
                operation_name="cortex_search"
            )
            if not result["success"]: return result
            return {"success": True, "results": result["result"]}
            
        elif action == "read_raw":
            if not MEMORY_FILE.exists(): return {"error": "memory.md no encontrado"}
            return {"success": True, "content": MEMORY_FILE.read_text(encoding="utf-8")}
            
        elif action == "init":
            AI_DIR.mkdir(parents=True, exist_ok=True)
            if not MEMORY_FILE.exists():
                MEMORY_FILE.write_text("# Proyecto: .ai Memory\n\nArchivo de memoria de largo plazo.", encoding="utf-8")
            return {"success": True, "message": "Cortex inicializado"}
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en manage_cortex: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL CENTER TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def sync_system_status(action: str = "sync", export_name: str = None, mission_goal: str = None) -> dict:
    """
    Centro de Control y Observabilidad de Ultragent.
    
    Unifica la sincronización de estado, la gestión de objetivos de misión
    y la exportación de bitácoras de sesión.
    
    Acciones:
    - 'sync': Snapshot 360 de salud, misiones, tokens y eventos.
    - 'set_goal': Define el objetivo actual (Mission Goal) del sistema.
    - 'export': Genera un ZIP con toda la memoria y logs de la sesión.
    """
    logger.info(f"sync_system_status: action={action}")
    try:
        from hud import get_hud
        hud = get_hud()
        
        if action == "sync":
            result = await with_timeout(
                asyncio.to_thread(hud.get_snapshot),
                timeout=10.0,
                operation_name="hud_snapshot"
            )
            if not result["success"]: return result
            return result["result"]
            
        elif action == "set_goal":
            if not mission_goal: return {"error": "Se requiere 'mission_goal'"}
            result = await with_timeout(
                asyncio.to_thread(hud.update_mission_goal, mission_goal),
                timeout=5.0,
                operation_name="set_goal"
            )
            if not result["success"]: return result
            return {"success": True, "message": "Objetivo actualizado"}
            
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en sync_system_status: {e}")
        return {"success": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# SCOUT / EVOLUTION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def strategic_consultant(
    action: str,
    query: str = "",
    local_file: str = "",
    project_type: str = "api",
    language: str = "python",
    min_stars: int = 500
) -> dict:
    """
    Consultoría Estratégica Senior y Scouting de "Gold Standards".
    
    Unifica las capacidades de búsqueda en GitHub, Benchmarking comparativo
    y recomendaciones arquitectónicas proactivas.
    
    Acciones:
    - 'research': Investigación universal sobre una meta (Goal) y recomendación.
    - 'scout': Busca los mejores repositorios (Gold Standards) para referencia.
    - 'benchmark': Compara un archivo local contra el mejor repo del mundo en su categoría.
    - 'plan': Genera un plan de ingeniería con LLM.
    - 'route': Enruta una tarea al tier óptimo del router.
    """
    logger.info(f"strategic_consultant: action={action}, query='{query}'")
    
    try:
        from scout import get_scout
        from evolution import get_evolution
        scout = get_scout()
        evolution = get_evolution()
        
        if action == "research":
            result = await with_timeout(
                evolution.proactive_research(query),
                timeout=20.0,
                operation_name="proactive_research"
            )
            if not result["success"]:
                return result
            return result["result"].to_dict()
            
        elif action == "scout":
            result = await with_timeout(
                scout.search_repositories(query=query, language=language, min_stars=min_stars, max_results=5),
                timeout=15.0,
                operation_name="GitHub search"
            )
            if not result["success"]:
                result["suggestion"] = "Verify GITHUB_TOKEN is set in .env file"
                return result
            return {"success": True, "repos": [r.to_dict() for r in result["result"]]}
            
        elif action == "benchmark":
            if not local_file: 
                return {"error": "Se requiere 'local_file'", "suggestion": "Provide the path to the file to benchmark"}
            p = Path(local_file)
            if not p.exists(): 
                return {"error": "Archivo no encontrado", "suggestion": f"Verify the path: {local_file}"}
            
            # Gold repos with timeout
            gold_result = await with_timeout(
                scout.harvest_gold_standard(project_type=project_type, language=language),
                timeout=15.0,
                operation_name="harvest_gold_standard"
            )
            if not gold_result["success"]:
                return gold_result
            gold_repos = gold_result["result"]
            if not gold_repos: 
                return {"error": "No hay benchmarks disponibles", "suggestion": "Try a different project_type or language"}
            
            best_repo = gold_repos[0]
            audit_result = await with_timeout(
                evolution.audit_code(
                    local_code=p.read_text(encoding="utf-8"), 
                    local_file=str(p.name), 
                    benchmark_repo=best_repo.full_name, 
                    language=language
                ),
                timeout=30.0,
                operation_name="code_audit"
            )
            if not audit_result["success"]:
                return audit_result
            report = audit_result["result"]
            return {"success": True, "scorecard": report.scorecard.to_dict(), "report_path": str(report.save())}

        elif action == "plan":
            from router import get_router
            prompt = f"Como arquitecto senior, genera un plan detallado para: {query}. Proyecto: {project_type}. Lenguaje: {language}"
            result = await with_timeout(
                get_router().route_task(task_type="STRATEGIC", payload=prompt),
                timeout=30.0,
                operation_name="LLM planning"
            )
            if not result["success"]:
                result["suggestion"] = "Check API keys for GEMINI_API_KEY or GROQ_API_KEY"
                return result
            return {"success": True, "plan": result["result"].content}

        elif action == "route":
            from router import get_router
            result = await with_timeout(
                get_router().route_task(task_type=project_type, payload=query),
                timeout=25.0,
                operation_name="LLM routing"
            )
            if not result["success"]:
                return result
            r = result["result"]
            return {"success": r.success, "content": r.content, "tier": r.tier.value}

        elif action == "swarm":
            from router import get_router
            result = await with_timeout(
                get_router().ask_swarm(task=query, subtasks=[query]),
                timeout=45.0,
                operation_name="swarm execution"
            )
            if not result["success"]:
                return result
            return {"success": True, "results": [r.content for r in result["result"]]}

        elif action == "autofix":
            # ACCIÓN DE INGENIERÍA ACTIVA (Code Fixing)
            if not local_file: return {"error": "Se requiere 'local_file'"}
            p = Path(local_file)
            if not p.exists(): return {"error": "Archivo no encontrado"}
            
            # Buscar benchmark si no se provee
            benchmark_code = ""
            if not query:
                # Auto-detectar benchmark
                gold_result = await with_timeout(
                    scout.harvest_gold_standard(project_type=project_type, language=language),
                    timeout=15.0, 
                    operation_name="auto_benchmark"
                )
                if gold_result["success"] and gold_result["result"]:
                    # Descargar código del benchmark (simulado o real)
                    # Por simplicidad usamos placeholder o lógica de scout si tuviera download
                    benchmark_code = "STANDARD_BENCHMARK_CODE" 
            
            local_code = p.read_text(encoding="utf-8")
            
            reports = await with_timeout(
                evolution.full_audit_cycle(
                    local_code=local_code,
                    benchmark_code=benchmark_code,
                    local_file=str(p),
                    language=language
                ),
                timeout=120.0, # Ciclo largo
                operation_name="autofix_cycle"
            )
            
            if not reports["success"]: return reports # Timeout wrapper envelops result
            
            # El resultado de with_timeout es un dict wrapper si success, o el valor directo?
            # with_timeout retorna {"success": True, "result": value}
            actual_reports = reports["result"]
            
            final_verdict = actual_reports[-1].verdict.value if actual_reports else "UNKNOWN"
            return {
                "success": True, 
                "iterations": len(actual_reports),
                "final_verdict": final_verdict,
                "reports": [r.scorecard.to_dict() for r in actual_reports]
            }

        else:
            return {
                "error": f"Acción desconocida: {action}", 
                "suggestion": "Valid actions: research, scout, benchmark, plan, route, swarm"
            }
            
    except Exception as e:
        logger.error(f"Error en strategic_consultant: {e}")
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


# ═══════════════════════════════════════════════════════════════════════════════
# VISION TOOLS (Hyper-V Rendering)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def visualize_architecture(action: str = "render", output_filename: str = None, project_root: str = "") -> dict:
    """
    Motor de Visualización Arquitectónica (Vision Hyper-V).
    
    Genera mapas de dependencia y reportes visuales del sistema.
    
    Acciones:
    - 'render': Genera el grafo PNG/HTML con dependencias y ciclos.
    - 'status': Retorna métricas de escaneo y configuración de visión.
    
    Args:
        project_root: Optional path to target project directory.
    """
    logger.info(f"visualize_architecture: action={action}, project_root={project_root}")
    try:
        from vision import get_vision, REPORTS_DIR
        vision = get_vision(project_root) if project_root else get_vision()
        
        if action == "render":
            def _render():
                output_path = str(REPORTS_DIR / output_filename) if output_filename else None
                return vision.generate_dependency_graph(output_path=output_path)
            
            result = await with_timeout(
                asyncio.to_thread(_render),
                timeout=30.0,
                operation_name="architecture_render"
            )
            if not result["success"]: return result
            report = result["result"]
            return {
                "success": True,
                "nodes": len(report.nodes),
                "edges": len(report.edges),
                "cycles": len(report.cycles),
                "graph_path": report.graph_path
            }
            
        elif action == "mermaid":
            mermaid_code = await vision.generate_mermaid_graph()
            return {
                "success": True,
                "mermaid": mermaid_code,
                "instructions": "Copy this into a mermaid editor or save as .mermaid"
            }
            
        elif action == "status":
            result = await with_timeout(
                asyncio.to_thread(vision.get_status),
                timeout=10.0,
                operation_name="vision_status"
            )
            if not result["success"]: return result
            return result["result"]
            
        elif action == "cycles":
            result = await with_timeout(
                asyncio.to_thread(vision.get_cycles_report),
                timeout=15.0,
                operation_name="vision_cycles"
            )
            if not result["success"]: return result
            return result["result"]
            
        else:
            return {"error": f"Acción desconocida: {action}"}
    except Exception as e:
        logger.error(f"Error en visualize_architecture: {e}")
        return {"success": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# MECHANIC TOOLS (Execution)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION TOOLS (Mechanic)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def test_code_securely(
    code: str,
    requirements: list[str] | None = None,
    timeout: int = 30,
) -> dict:
    """
    Validación rápida de código en Sandbox efímero (Legacy).
    
    Ejecuta scripts Python en un contenedor Docker aislado con límites de 
    recurso estrictos y auto-destrucción.
    """
    logger.info(f"test_code_securely: timeout={timeout}s")
    try:
        from mechanic import get_mechanic
        mechanic = get_mechanic()
        if not mechanic.is_available: return {"success": False, "error": "Docker no disponible"}
        
        result = mechanic.run_in_sandbox(script=code, requirements=requirements, timeout=timeout)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error en test_code_securely: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def issue_command(
    command: str,
    mission_id: str = None
) -> dict:
    """
    S.O.T.A COMMAND: Dar una orden de alto nivel al Enjambre Soberano.
    Activa al COMMANDER para orquestar la misión a través de múltiples agentes.
    
    Args:
        command: La orden técnica o estratégica.
        mission_id: ID opcional para continuar una misión existente.
    """
    logger.info(f"issue_command: '{command[:50]}'")
    try:
        from mechanic import get_mechanic
        mechanic = get_mechanic()
        
        if mechanic is None:
             return {"success": False, "error": "Mechanic offline"}

        # Forzamos la ejecución vía run_swarm_mission
        result = await mechanic.run_swarm_mission(command)
        return {"success": True, "result": result, "mission_id": mission_id}
        
    except Exception as e:
        logger.error(f"Command Error: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def mission_status(
    mission_id: str = None
) -> dict:
    """
    S.O.T.A MONITOR: Ver el estado global de las misiones del enjambre.
    Lee la Verdad (Blackboard) persistente.
    """
    try:
        from agent_manager import get_agent_manager
        manager = get_agent_manager()
        messages = manager.get_messages(task_id=mission_id)
        
        return {
            "success": True,
            "mission_id": mission_id,
            "history": [
                {
                    "from": m.sender_role.upper(),
                    "to": (m.target_role.upper() if m.target_role else "ALL"),
                    "content": m.content[:200] + ("..." if len(m.content) > 200 else ""),
                    "time": m.timestamp
                } for m in messages
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def run_agentic_task(
    task: str,
    target_directory: str,
    max_steps: int = 10,  # Reduced for native safety
    agent_type: str = "NativeGemini" # Updated default
) -> dict:
    """
    MISIÓN AGÉNTICA: Ejecución autómoma nativa (Ultragent Loop).
    
    El agente usa Gemini Flash (Free Tier) para planificar y ejecutar tareas
    directamente en el sistema (sin Docker overhead).
    
    Args:
        task: Descripción de la misión (ej: "Optimizar el sistema de logs")
        target_directory: Directorio raíz del proyecto (absoluto)
        max_steps: Límite de iteraciones (default: 10)
    """
    logger.info(f"run_agentic_task (Native): '{task[:50]}'")
    try:
        from mechanic import get_mechanic
        mechanic = get_mechanic()
        
        if mechanic is None:
             return {"success": False, "error": "Mechanic no inicializado (Falta SDK o API Key)"}

        # Ejecutar bucle nativo
        result = await mechanic.run_task(task, max_steps=max_steps)
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Error en run_agentic_task: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def heal_system_node(node_path: str, issue: str) -> dict:
    """
    Inicia el protocolo de auto-curación para un nodo específico.
    Usa el sistema inmunológico (Impact Analysis + Mechanics).
    """
    logger.info(f"heal_system_node: {node_path}")
    try:
        from neuro_architect import get_neuro_architect
        impact = get_neuro_architect().analyze_impact(node_path)
        
        from mechanic import get_mechanic
        # El mecánico ahora recibe el impacto para priorizar riesgos
        return await get_mechanic().heal_node(node_path, issue, impact_data=impact.to_dict())
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_error_guide(error_code: str) -> dict:
    """
    Motor de Consulta de Troubleshooting (Error Guide).
    Busca soluciones conocidas en la memoria atómica y doc técnica.
    """
    logger.info(f"get_error_guide: {error_code}")
    try:
        from librarian import get_librarian
        result = await with_timeout(
            asyncio.to_thread(get_librarian().semantic_search, query=f"Solución para error {error_code}", node_type="memory", n_results=3),
            timeout=10.0,
            operation_name="error_guide_search"
        )
        if not result["success"]: return result
        return {"success": True, "guides": result["result"]}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# NEURO-VISION TOOLS (Hyper-V)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def analyze_impact(
    action: str,
    target_node: str = "",
    start_node: str = "",
    end_node: str = "",
    project_root: str = ""
) -> dict:
    """
    Inteligencia Predictiva y Trazado de Flujos (Neuro-Architect).
    
    Analiza riesgos antes de editar y traza conexiones lógicas.
    
    Acciones:
    - 'impact': Predice efectos secundarios de modificar un archivo.
    - 'trace': Traza el flujo de datos exacto entre dos nodos.
    - 'brain_state': Retorna el estado actual del grafo neural (completo).
    - 'brain_min': Retorna el estado comprimido (óptimo para tokens).
    
    Args:
        project_root: Optional path to target project directory.
    """
    logger.info(f"analyze_impact: action={action}, target={target_node}, project_root={project_root}")
    try:
        from neuro_architect import get_neuro_architect
        neuro = get_neuro_architect(project_root) if project_root else get_neuro_architect()
        
        if action == "impact":
            if not target_node: return {"error": "Se requiere 'target_node'"}
            result = await with_timeout(
                asyncio.to_thread(neuro.analyze_impact, target_node), # Changed from predict_impact to analyze_impact based on original code
                timeout=20.0,
                operation_name="impact_analysis"
            )
            if not result["success"]: return result
            impact = result["result"].to_dict() # Added .to_dict() as analyze_impact returns an object
            # Enriquecer con memorias atómicas
            from cortex import get_cortex
            related = get_cortex().get_related_memories(target_node)
            impact["linked_memories"] = [m.content for m in related[:5]]
            return impact
            
        elif action == "trace":
            if not start_node or not end_node:
                return {"error": "Se requieren 'start_node' y 'end_node'"}
            result = await with_timeout(
                asyncio.to_thread(neuro.trace_flow, start_node, end_node),
                timeout=20.0,
                operation_name="flow_tracing"
            )
            if not result["success"]: return result
            return result["result"]
            
        elif action == "bundle":
            # Retorna un paquete completo para el Agente (Impacto + Relaciones + Riesgos)
            if not target_node: return {"error": "Se requiere 'target_node'"}
            
            impact_result = await with_timeout(
                asyncio.to_thread(neuro.analyze_impact, target_node),
                timeout=20.0,
                operation_name="bundle_impact_analysis"
            )
            if not impact_result["success"]: return impact_result
            impact = impact_result["result"].to_dict()

            flow_result = await with_timeout(
                asyncio.to_thread(neuro.get_compressed_brain_state),
                timeout=10.0,
                operation_name="bundle_graph_topology"
            )
            if not flow_result["success"]: return flow_result
            flow = flow_result["result"]

            return {"success": True, "bundle_type": "ARCHITECT_CONTEXT", "impact_analysis": impact, "graph_topology": flow}
            
        elif action == "brain_state":
            return neuro.get_brain_state()
        elif action == "brain_min":
            return {"compressed_state": neuro.get_compressed_brain_state()}
        elif action == "focus":
            return neuro.get_next_focus()
        else:
            return {"error": f"Acción desconocida: {action}"}
    except Exception as e:
        logger.error(f"Error en analyze_impact: {e}")
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE TOOLS (God Mode)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def manage_infrastructure(action: str, target: str = "") -> dict:
    """
    Controlador Maestro de Infraestructura (Cloudflare/Vercel/Supabase).
    
    Acciones:
    - 'purge_cache': Limpieza global de caché (CDN + Edge).
    - 'sync_secrets': Sincroniza .env local con bóvedas remotas.
    - 'status': Ver estado de conexiones.
    """
    # Importación lazy para evitar ciclos
    from infrastructure import get_infrastructure
    infra = get_infrastructure()
    
    try:
        if action == "purge_cache":
            result = await with_timeout(
                asyncio.to_thread(infra.purge_global_cache, target),
                timeout=30.0,
                operation_name="infra_purge"
            )
            if not result["success"]: return result
            return {"success": True, "details": result["result"]}
            
        elif action == "sync_secrets":
            result = await with_timeout(
                asyncio.to_thread(infra.sync_secrets, target),
                timeout=30.0,
                operation_name="infra_sync"
            )
            if not result["success"]: return result
            return {"success": True, "details": result["result"]}
            
        elif action == "status":
            return infra.get_status()
            
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en manage_infrastructure: {e}")
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# MCP RESOURCES (Documentation for Agents)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource("ultragent://docs/overview")
def get_overview() -> str:
    """Overview of Ultragent MCP capabilities."""
    return """# Ultragent MCP - Quick Reference

## Core Tools (Start Here)
- `sync_system_status`: Health check & sync. Call with action='sync' first.
- `manage_cortex`: Project memory (add/list decisions)

## Research Tools
- `strategic_consultant`: GitHub scouting & code benchmarking
  - action='scout': Find Gold Standard repos
  - action='research': Universal search with recommendations
  - action='benchmark': Compare local code vs best practices

## Code Analysis Tools
- `semantic_search`: Find code by natural language query
- `analyze_impact`: Predict side effects of changes
- `visualize_architecture`: Generate dependency graphs

## Execution Tools  
- `run_agentic_task`: Autonomous task execution (Gemini-powered)
- `test_code_securely`: Run code in Docker sandbox

## Recommended Workflow
1. sync_system_status(action='sync') - Check system health
2. strategic_consultant(action='scout') - Find reference implementations
3. semantic_search(action='search') - Understand existing code
4. analyze_impact(action='impact') - Before making changes
"""

@mcp.resource("ultragent://docs/errors")
def get_error_guide() -> str:
    """Common errors and solutions."""
    return """# Ultragent Error Guide

## TIMEOUT Errors
- **Cause**: External API took too long (GitHub, LLM, etc.)
- **Solution**: Check API keys in .env, try simpler query

## "Path not defined" 
- **Cause**: Module import order bug (fixed in v2.1)
- **Solution**: Restart MCP server

## "No GITHUB_TOKEN"
- **Cause**: Scout needs GitHub API access
- **Solution**: Add GITHUB_TOKEN to .env file

## "Docker unavailable"
- **Cause**: Mechanic needs Docker for sandboxing
- **Solution**: Start Docker Desktop or use lightweight mode

## API Key Checklist (.env)
- GITHUB_TOKEN: For GitHub searches
- GEMINI_API_KEY: For strategic planning
- GROQ_API_KEY: For fast operations
"""

@mcp.resource("ultragent://docs/tools")
def get_tools_reference() -> str:
    """Detailed tool reference."""
    return """# Tool Reference

## strategic_consultant
**Purpose**: External research and code comparison
**Actions**:
| Action | Timeout | Description |
|--------|---------|-------------|
| scout | 15s | Search GitHub for repos |
| research | 20s | Web search + recommendations |
| benchmark | 30s | Compare code vs Gold Standard |
| plan | 30s | Generate engineering plan |
| route | 25s | Route task to optimal LLM |

## semantic_search
**Purpose**: Code understanding via NLP
**Actions**: search, skeleton, index, scan_debt

## analyze_impact
**Purpose**: Predict change effects
**Actions**: impact, trace, brain_state, brain_min
"""

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
