"""
ULTRAGENT DAEMON v0.1 (The Watcher)
===================================
Servicio de monitoreo en segundo plano (Lightweight).
Usa `watchdog` para detectar cambios y ejecutar auditorías rápidas.

Requisitos: pip install watchdog
Uso: python daemon.py
"""

import sys
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from metrics import MetricsEngine

# Configuración
PROJECT_ROOT = Path.cwd()
IGNORE_DIRS = {".git", "__pycache__", "venv", ".ai", "node_modules", "logs"}
WATCH_EXTENSIONS = {".py", ".js", ".ts"}

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [WATCHER] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("UltragentDaemon")

class QualityGuardian(FileSystemEventHandler):
    """Manejador de eventos que audita calidad en tiempo real."""
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        
        # Filtros rápidos
        if path.suffix not in WATCH_EXTENSIONS:
            return
        if any(part in str(path) for part in IGNORE_DIRS):
            return
            
        logger.info(f"⚡ Detected change in: {path.name}")
        self._audit_file(path)

    def _audit_file(self, path: Path):
        """Ejecuta MetricsEngine sobre el archivo modificado."""
        try:
            # Short sleep to ensure write is complete
            time.sleep(0.1)
            content = path.read_text(encoding="utf-8", errors="ignore")
            
            # 1. Deterministic Check
            if path.suffix == ".py":
                metrics = MetricsEngine.analyze_code(content, str(path))
                
                if metrics.grade in ["D", "F"]:
                    logger.warning(f"🚨 QUALITY ALERT: {path.name} dropped to Grade {metrics.grade}!")
                    logger.warning(f"   Complexity: {metrics.cyclomatic_complexity}")
                    logger.warning(f"   Maintainability: {metrics.maintainability_index}")
                    
                    # TODO: Trigger Evolution Agent via API/CLI?
                    # For now, we just scream in the logs.
                else:
                    logger.info(f"✅ Quality OK (Grade {metrics.grade}) for {path.name}")
            
        except Exception as e:
            logger.error(f"Failed to audit {path.name}: {e}")

def start_daemon():
    """Inicia el loop de monitoreo."""
    logger.info(f"👁️ Ultragent Daemon Watching: {PROJECT_ROOT}")
    event_handler = QualityGuardian()
    observer = Observer()
    observer.schedule(event_handler, str(PROJECT_ROOT), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_daemon()
