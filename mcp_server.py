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

PROJECT_ROOT = Path.cwd() # Runs in context of the target project
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
def sync_system_status(action: str = "sync", mission_goal: str = None, export_name: str = None) -> dict:
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
        from hud_manager import get_hud_manager
        hud = get_hud_manager()

        if action == "set_goal":
            if not mission_goal: return {"error": "Se requiere 'mission_goal'"}
            hud.set_mission_goal(mission_goal)
            hud.refresh_dashboard(force=True)
            return {"success": True, "goal": mission_goal}

        elif action == "export":
            zip_path = hud.export_session(export_name)
            return {"success": True, "export_path": zip_path}

        elif action == "reset":
            try:
                from sentinel import get_sentinel
                get_sentinel().clear_signals()
                return {"success": True, "message": "Señales de Sentinel reiniciadas"}
            except Exception as e: return {"error": str(e)}

        elif action == "health":
            health = {"status": "ok", "checks": {}}
            import docker
            try:
                client = docker.from_env()
                health["checks"]["docker"] = "online" if client.ping() else "error"
            except Exception: health["checks"]["docker"] = "offline"
            health["checks"]["db"] = "ok" if TASKS_DB.exists() else "missing"
            health["checks"]["cortex"] = "ok" if MEMORY_FILE.exists() else "warning"
            return health

        elif action == "sync":
            status = {
                "timestamp": datetime.now().isoformat(),
                "core": {"status": "online", "ai_dir": str(AI_DIR)},
                "hud": hud.get_full_status(),
                "sentinel": {"running": False, "events": 0},
                "scout": {}, "evolution": {}, "librarian": {},
                "mechanic": {}, "vision": {},
                "intelligence": {"total_tokens": 0, "tiers": {}}
            }
            # Carga perezosa de estados de módulos
            try:
                from sentinel import get_sentinel
                s_stat = get_sentinel().get_status()
                status["sentinel"] = {"running": s_stat["running"], "events": s_stat["events_processed"]}
            except Exception: pass
            
            try:
                from scout import get_scout
                from evolution import get_evolution
                status["scout"] = get_scout().get_status()
                status["evolution"] = get_evolution().get_status()
            except Exception: pass

            try:
                from librarian import get_librarian
                status["librarian"] = get_librarian().get_status()
            except Exception: pass

            try:
                from mechanic import get_mechanic
                from vision import get_vision
                status["mechanic"] = get_mechanic().get_status()
                status["vision"] = get_vision().get_status()
            except Exception: pass

            try:
                from router import get_router
                usage = get_router().get_token_usage()
                status["intelligence"] = {"total_tokens": usage["total_used"], "tiers": usage["by_tier"]}
            except Exception: pass

            return status
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en sync_system_status: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION CONTROL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
# (N/A - Unified in sync_system_status)


# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARIAN TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def semantic_search(
    action: str,
    query: str = "",
    file_path: str = "",
    n_results: int = 5,
    language: str = "",
    node_type: str = ""
) -> dict:
    """
    Descubrimiento Semántico y Arquitectónico de Código.
    
    Unifica la búsqueda por lenguaje natural, la inspección de esqueletos
    y el indexado de nuevos módulos.
    
    Acciones:
    - 'search': Búsqueda semántica (NLP) de lógica y funciones.
    - 'skeleton': Obtiene la estructura funcional de un archivo (firmas).
    - 'index': Indexa o re-escanea un archivo para la base de datos vectorial.
    """
    logger.info(f"semantic_search: action={action}, query='{query}', file='{file_path}'")
    
    try:
        from librarian import get_librarian
        lib = get_librarian()
        
        if action == "search":
            results = lib.semantic_search(
                query=query, n_results=n_results, 
                language=language if language else None,
                node_type=node_type if node_type else None
            )
            return {"success": True, "results": results}
            
        elif action == "skeleton":
            if not file_path: return {"error": "Se requiere 'file_path'"}
            skeleton = lib.get_file_skeleton(file_path)
            return skeleton
            
        elif action == "index":
            if not file_path: return {"error": "Se requiere 'file_path'"}
            return lib.index_file(file_path)

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
def manage_cortex(action: str, content: str = "", tags: list[str] = None, importance: float = 1.0) -> dict:
    """
    Gestión integral del Cortex (Memoria de Proyecto .ai).
    
    Acciones:
    - 'add': Guarda un nuevo átomo de memoria (recuerdo).
    - 'list': Retorna todos los recuerdos del proyecto.
    - 'read_raw': Retorna el contenido bruto de memory.md.
    - 'init': Inicializa la estructura cerebral en un nuevo proyecto.
    """
    logger.info(f"manage_cortex: action={action}")
    try:
        from cortex import get_cortex
        cortex = get_cortex()
        
        if action == "add":
            mid = cortex.add_memory(content, tags, importance)
            return {"success": True, "memory_id": mid}
        elif action == "list":
            memories = cortex.get_all_memories()
            return {
                "success": True, 
                "memories": [
                    {"id": m.id, "content": m.content, "tags": m.tags, "importance": m.importance, "created_at": str(m.created_at)} for m in memories
                ]
            }
        elif action == "search":
            from librarian import get_librarian
            # Cross-module usage for semantic search in memory
            results = get_librarian().semantic_search(query=content, n_results=5, node_type="memory")
            return {"success": True, "results": results}
        elif action == "read_raw":
            if not MEMORY_FILE.exists(): return {"error": "memory.md no encontrado"}
            return {"success": True, "content": MEMORY_FILE.read_text(encoding="utf-8")}
        elif action == "init":
            return {"success": True, "message": "Cortex inicializado"}
        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en manage_cortex: {e}")
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
    """
    logger.info(f"strategic_consultant: action={action}, query='{query}'")
    
    try:
        from scout import get_scout
        from evolution import get_evolution
        scout = get_scout()
        evolution = get_evolution()
        
        if action == "research":
            report = await evolution.proactive_research(query)
            return report.to_dict()
            
        elif action == "scout":
            repos = await scout.search_repositories(query=query, language=language, min_stars=min_stars, max_results=5)
            return {"success": True, "repos": [r.to_dict() for r in repos]}
            
        elif action == "benchmark":
            if not local_file: return {"error": "Se requiere 'local_file'"}
            p = Path(local_file)
            if not p.exists(): return {"error": "Archivo no encontrado"}
            gold_repos = await scout.harvest_gold_standard(project_type=project_type, language=language)
            if not gold_repos: return {"error": "No hay benchmarks disponibles"}
            best_repo = gold_repos[0]
            report = await evolution.audit_code(local_code=p.read_text(encoding="utf-8"), local_file=str(p.name), benchmark_repo=best_repo.full_name, language=language)
            return {"success": True, "scorecard": report.scorecard.to_dict(), "report_path": str(report.save())}

        elif action == "plan":
            # Genera un plan de ingeniería basado en el estado actual y meta
            from router import get_router
            prompt = f"Como arquitecto senior, genera un plan detallado para: {query}. Proyecto: {project_type}. Lenguaje: {language}"
            result = await get_router().route_task(task_type="STRATEGIC", payload=prompt)
            return {"success": True, "plan": result.content}

        elif action == "route":
            from router import get_router
            result = await get_router().route_task(task_type=project_type, payload=query)
            return {"success": result.success, "content": result.content, "tier": result.tier.value}

        elif action == "swarm":
            from router import get_router
            results = await get_router().ask_swarm(task=query, subtasks=[query])
            return {"success": True, "results": [r.content for r in results]}

        else:
            return {"error": f"Acción desconocida: {action}"}
            
    except Exception as e:
        logger.error(f"Error en strategic_consultant: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# VISION TOOLS (Hyper-V Rendering)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def visualize_architecture(action: str = "render", output_filename: str = None) -> dict:
    """
    Motor de Visualización Arquitectónica (Vision Hyper-V).
    
    Genera mapas de dependencia y reportes visuales del sistema.
    
    Acciones:
    - 'render': Genera el grafo PNG/HTML con dependencias y ciclos.
    - 'status': Retorna métricas de escaneo y configuración de visión.
    """
    logger.info(f"visualize_architecture: action={action}")
    try:
        from vision import get_vision, REPORTS_DIR
        vision = get_vision()
        if action == "render":
            output_path = str(REPORTS_DIR / output_filename) if output_filename else None
            report = vision.generate_dependency_graph(output_path=output_path)
            return {"success": True, "graph_path": report.graph_path, "cycles": report.cycles}
        elif action == "status":
            return vision.get_status()
        elif action == "cycles":
            return vision.get_cycles_report()
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
    Usa el sistema inmunológico (Mechanic + Cortex + NeuroArchitect).
    """
    logger.info(f"heal_system_node: {node_path}")
    try:
        from mechanic import get_mechanic
        return await get_mechanic().heal_node(node_path, issue)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# NEURO-VISION TOOLS (Hyper-V)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def analyze_impact(
    action: str,
    target_node: str = "",
    start_node: str = "",
    end_node: str = ""
) -> dict:
    """
    Inteligencia Predictiva y Trazado de Flujos (Neuro-Architect).
    
    Analiza riesgos antes de editar y traza conexiones lógicas.
    
    Acciones:
    - 'impact': Predice efectos secundarios de modificar un archivo.
    - 'trace': Traza el flujo de datos exacto entre dos nodos.
    - 'brain_state': Retorna el estado actual del grafo neural (completo).
    - 'brain_min': Retorna el estado comprimido (óptimo para tokens).
    """
    logger.info(f"analyze_impact: action={action}, target={target_node}")
    try:
        from neuro_architect import get_neuro_architect
        neuro = get_neuro_architect()
        
        if action == "impact":
            if not target_node: return {"error": "Se requiere 'target_node'"}
            return neuro.analyze_impact(target_node).to_dict()
        elif action == "trace":
            if not (start_node and end_node): return {"error": "Se requiere 'start_node' y 'end_node'"}
            return neuro.trace_flow(start_node, end_node)
        elif action == "bundle":
            # Retorna un paquete completo para el Agente (Impacto + Relaciones + Riesgos)
            if not target_node: return {"error": "Se requiere 'target_node'"}
            impact = neuro.analyze_impact(target_node).to_dict()
            flow = neuro.get_compressed_brain_state()
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
