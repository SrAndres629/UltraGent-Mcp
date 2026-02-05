"""
ULTRAGENT v2.0 - Main Entry Point
==================================
Orquesta el arranque de todos los subsistemas.

Componentes:
- MCP Server (FastMCP)
- Sentinel (Watchdog filesystem)
- HUD Refresher (observabilidad periódica)

Uso:
    uv run main.py
    
    # O con argumentos:
    uv run main.py --hud-interval 5 --log-level DEBUG
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██╗   ██╗██╗  ████████╗██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗║
║    ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝║
║    ██║   ██║██║     ██║   ██████╔╝███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ║
║    ██║   ██║██║     ██║   ██╔══██╗██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ║
║    ╚██████╔╝███████╗██║   ██║  ██║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ║
║     ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ║
║                                                                               ║
║                    🤖 Hybrid Autonomous Engineering System                    ║
║                              Version {version}                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")

# Eventos de control
shutdown_event = Event()

# Logger
logger = logging.getLogger("ultragent.main")


# ═══════════════════════════════════════════════════════════════════════════════
# HUD REFRESHER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def hud_refresher_thread(
    interval: float = 5.0,
    stop_event: Optional[Event] = None,
) -> None:
    """
    Thread que actualiza el HUD periódicamente.
    
    Args:
        interval: Segundos entre actualizaciones
        stop_event: Evento para detener el thread
    """
    from hud_manager import get_hud_manager
    
    hud = get_hud_manager()
    logger.info(f"HUD Refresher iniciado (interval={interval}s)")
    
    while not (stop_event and stop_event.is_set()):
        try:
            hud.refresh_dashboard()
        except Exception as e:
            logger.error(f"Error refreshing HUD: {e}")
        
        # Esperar con chequeo frecuente de stop_event
        for _ in range(int(interval * 10)):
            if stop_event and stop_event.is_set():
                break
            time.sleep(0.1)
    
    logger.info("HUD Refresher detenido")


# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def sentinel_thread(stop_event: Optional[Event] = None) -> None:
    """
    Thread que ejecuta el Sentinel.
    
    Args:
        stop_event: Evento para detener el thread
    """
    try:
        from sentinel import get_sentinel
        
        sentinel = get_sentinel()
        logger.info("Sentinel thread iniciado")
        
        # Iniciar observación
        sentinel.start()
        
        # Esperar hasta shutdown
        while not (stop_event and stop_event.is_set()):
            time.sleep(0.5)
        
        # Detener sentinel
        sentinel.stop()
        
    except Exception as e:
        logger.error(f"Error en Sentinel thread: {e}")
    
    logger.info("Sentinel thread detenido")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO") -> None:
    """Configura el sistema de logging."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                AI_DIR / "logs" / "ultragent.log",
                mode="a",
                encoding="utf-8",
            ),
        ],
    )
    
    # Reducir verbosidad de librerías externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def signal_handler(signum, frame):
    """Manejador de señales para shutdown graceful."""
    logger.info(f"Señal recibida: {signum}. Iniciando shutdown...")
    shutdown_event.set()


def print_startup_info() -> None:
    """Imprime información de arranque."""
    print(BANNER.format(version=VERSION))
    print(f"  📁 Project Root: {PROJECT_ROOT}")
    print(f"  📂 AI Directory: {AI_DIR}")
    print(f"  🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def validate_environment() -> list[str]:
    """Valida el entorno y retorna warnings."""
    warnings = []
    
    # Verificar .env
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        warnings.append("⚠️  .env file not found - using defaults")
    
    # Verificar API keys
    if not os.getenv("KIMI_API_KEY"):
        warnings.append("⚠️  KIMI_API_KEY not set - Router tier STRATEGIC limited")
    
    if not os.getenv("GITHUB_TOKEN"):
        warnings.append("⚠️  GITHUB_TOKEN not set - Scout rate limited")
    
    # Verificar Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception:
        warnings.append("⚠️  Docker not available - Mechanic sandbox disabled")
    
    return warnings


def main(
    hud_interval: float = 5.0,
    log_level: str = "INFO",
    no_sentinel: bool = False,
    no_hud: bool = False,
) -> int:
    """
    Punto de entrada principal.
    
    Args:
        hud_interval: Intervalo de actualización del HUD en segundos
        log_level: Nivel de logging
        no_sentinel: Deshabilitar Sentinel
        no_hud: Deshabilitar HUD refresher
        
    Returns:
        Exit code (0 = success)
    """
    # Crear directorios necesarios
    (AI_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (AI_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (AI_DIR / "workspace").mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    setup_logging(log_level)
    
    # Banner
    print_startup_info()
    
    # Validar entorno
    warnings = validate_environment()
    for w in warnings:
        print(f"  {w}")
    if warnings:
        print()
    
    # Configurar señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    threads = []
    
    try:
        # Iniciar HUD refresher
        if not no_hud:
            hud_thread = Thread(
                target=hud_refresher_thread,
                args=(hud_interval, shutdown_event),
                daemon=True,
                name="HUD-Refresher",
            )
            hud_thread.start()
            threads.append(hud_thread)
            logger.info("HUD Refresher thread started")
        
        # Iniciar Sentinel
        if not no_sentinel:
            sent_thread = Thread(
                target=sentinel_thread,
                args=(shutdown_event,),
                daemon=True,
                name="Sentinel",
            )
            sent_thread.start()
            threads.append(sent_thread)
            logger.info("Sentinel thread started")
        
        # Generar HUD inicial
        from hud_manager import get_hud_manager
        hud = get_hud_manager()
        hud.set_mission_goal("ULTRAGENT v2.0 - System Operational")
        hud.refresh_dashboard(force=True)
        
        print("  ✅ All systems initialized")
        print("  🚀 Starting MCP Server...")
        print()
        print("=" * 79)
        
        # Iniciar MCP Server (bloquea hasta shutdown)
        from mcp_server import mcp
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error en main: {e}")
        return 1
    finally:
        # Shutdown graceful
        logger.info("Initiating graceful shutdown...")
        shutdown_event.set()
        
        # Esperar threads
        for t in threads:
            t.join(timeout=2.0)
        
        # HUD final
        try:
            from hud_manager import get_hud_manager
            hud = get_hud_manager()
            hud.refresh_dashboard(force=True)
        except Exception:
            pass
        
        logger.info("Shutdown complete")
        print("\n  👋 ULTRAGENT shutdown complete. Goodbye!")
    
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ULTRAGENT v2.0 - Hybrid Autonomous Engineering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                      # Start with defaults
  uv run main.py --hud-interval 10    # HUD updates every 10s
  uv run main.py --log-level DEBUG    # Verbose logging
  uv run main.py --no-sentinel        # Without filesystem watcher
        """,
    )
    
    parser.add_argument(
        "--hud-interval",
        type=float,
        default=5.0,
        help="HUD refresh interval in seconds (default: 5.0)",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--no-sentinel",
        action="store_true",
        help="Disable Sentinel filesystem watcher",
    )
    
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD automatic refresh",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"ULTRAGENT v{VERSION}",
    )
    
    args = parser.parse_args()
    
    sys.exit(main(
        hud_interval=args.hud_interval,
        log_level=args.log_level,
        no_sentinel=args.no_sentinel,
        no_hud=args.no_hud,
    ))
