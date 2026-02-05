"""
ULTRAGENT SENTINEL v0.1
=======================
Módulo de monitoreo reactivo basado en eventos para el Core .ai/.

Implementa:
- Watchdog FileSystemEventHandler con debounce de 2 segundos
- Filtros de exclusión para archivos temporales/ocultos
- Actualización automática de HUD.md
- Sistema de señales (signals.json) para comunicación inter-lóbulos
- Rate limiting para evitar event flooding
"""

import json
import logging
import os
import re
import sqlite3
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Timer
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
HUD_FILE = AI_DIR / "HUD.md"
SIGNALS_FILE = AI_DIR / "signals.json"
TASKS_DB = AI_DIR / "tasks.db"
LOGS_DIR = AI_DIR / "logs"

# Configuración de Sentinel
DEBOUNCE_SECONDS = 2.0
MAX_EVENTS_PER_SECOND = 10
MAX_EVENT_HISTORY = 50
WATCH_RECURSIVE = True

# Patrones de exclusión (archivos que NO deben disparar eventos)
EXCLUDED_PATTERNS = [
    r"/\.",              # Archivos ocultos Unix (.gitignore, .env)
    r"\\\.",             # Archivos ocultos Windows
    r"\.tmp$",           # Temporales
    r"~$",               # Backup de editores
    r"\.swp$",           # Vim swap
    r"\.swo$",           # Vim swap
    r"__pycache__",      # Python cache
    r"\.pyc$",           # Python compiled
    r"node_modules",     # NPM
    r"\.git/",           # Git internals
    r"\.git\\",          # Git internals Windows
    r"\.ai/logs/",       # Nuestros propios logs
    r"\.ai\\logs\\",     # Logs Windows
    r"tasks\.db-wal$",   # SQLite WAL
    r"tasks\.db-shm$",   # SQLite SHM
    r"signals\.json$",   # Nuestro propio archivo de señales
]

# Extensiones monitoreadas (whitelist opcional)
MONITORED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".json", ".yaml", ".yml",
    ".html", ".css", ".scss",
    ".sql", ".toml", ".env",
}

# Logger
logger = logging.getLogger("ultragent.sentinel")


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTO PROCESADO
# ═══════════════════════════════════════════════════════════════════════════════

class SentinelEvent:
    """Representa un evento de filesystem procesado y estabilizado."""

    def __init__(
        self,
        event_type: str,
        src_path: str,
        timestamp: Optional[datetime] = None,
        source: str = "unknown",
    ):
        self.event_type = event_type
        self.src_path = src_path
        self.timestamp = timestamp or datetime.now()
        self.source = source  # "human" | "ultragent" | "unknown"
        self.processed = False

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "src_path": self.src_path,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "processed": self.processed,
        }

    def to_hud_line(self) -> str:
        """Formatea el evento como línea legible para HUD.md."""
        icon = {
            "created": "📄",
            "modified": "✏️",
            "deleted": "🗑️",
            "moved": "📦",
        }.get(self.event_type, "❓")

        filename = Path(self.src_path).name
        time_str = self.timestamp.strftime("%H:%M:%S")

        return f"| {icon} {self.event_type.upper()} | `{filename}` | {time_str} | {self.source} |"


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLER DE EVENTOS CON DEBOUNCE
# ═══════════════════════════════════════════════════════════════════════════════

class UltragentEventHandler(FileSystemEventHandler):
    """
    Handler de eventos con debounce para estabilizar archivos en escritura.
    
    Implementa:
    - Debounce de 2 segundos (configurable)
    - Filtrado de archivos excluidos
    - Rate limiting para evitar flooding
    """

    def __init__(
        self,
        on_stable_event: Callable[[SentinelEvent], None],
        debounce_seconds: float = DEBOUNCE_SECONDS,
    ):
        super().__init__()
        self._on_stable_event = on_stable_event
        self._debounce_seconds = debounce_seconds
        self._pending_timers: dict[str, Timer] = {}
        self._lock = Lock()
        self._last_event_time: dict[str, datetime] = {}

    def _should_ignore(self, path: str) -> bool:
        """Verifica si el path debe ser ignorado."""
        # Verificar patrones de exclusión
        for pattern in EXCLUDED_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return True

        # Verificar extensión (si está en whitelist)
        ext = Path(path).suffix.lower()
        if ext and ext not in MONITORED_EXTENSIONS:
            # Solo ignorar si tiene extensión y no está en whitelist
            # Archivos sin extensión o directorios se procesan
            if ext not in {"", None}:
                return True

        return False

    def _detect_source(self, path: str) -> str:
        """
        Detecta si el cambio fue hecho por humano o por Ultragent.
        
        Busca el header [ULTRAGENT:MODIFIED] en el archivo.
        """
        try:
            file_path = Path(path)
            if file_path.exists() and file_path.is_file():
                # Solo leer primeras líneas para eficiencia
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_lines = f.read(500)
                    if "[ULTRAGENT:MODIFIED]" in first_lines:
                        return "ultragent"
        except (PermissionError, OSError, IOError):
            pass
        return "human"

    def _schedule_debounce(self, event_type: str, src_path: str) -> None:
        """Programa el procesamiento del evento con debounce."""
        with self._lock:
            # Cancelar timer existente para este path
            if src_path in self._pending_timers:
                self._pending_timers[src_path].cancel()

            # Crear nuevo timer
            def process_after_debounce():
                with self._lock:
                    if src_path in self._pending_timers:
                        del self._pending_timers[src_path]

                # Detectar fuente del cambio
                source = self._detect_source(src_path)

                # Crear evento procesado
                event = SentinelEvent(
                    event_type=event_type,
                    src_path=src_path,
                    source=source,
                )

                logger.info(
                    f"Evento estabilizado: {event_type} {Path(src_path).name} "
                    f"(source={source})"
                )

                # Invocar callback
                try:
                    self._on_stable_event(event)
                except Exception as e:
                    logger.error(f"Error procesando evento: {e}")

            timer = Timer(self._debounce_seconds, process_after_debounce)
            self._pending_timers[src_path] = timer
            timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_ignore(event.src_path):
            return
        logger.debug(f"on_created: {event.src_path}")
        self._schedule_debounce("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_ignore(event.src_path):
            return
        logger.debug(f"on_modified: {event.src_path}")
        self._schedule_debounce("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_ignore(event.src_path):
            return
        logger.debug(f"on_deleted: {event.src_path}")
        # Para deleted no aplicamos debounce - es instantáneo
        sentinel_event = SentinelEvent(
            event_type="deleted",
            src_path=event.src_path,
            source="unknown",
        )
        try:
            self._on_stable_event(sentinel_event)
        except Exception as e:
            logger.error(f"Error procesando delete: {e}")

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_ignore(event.src_path):
            return
        logger.debug(f"on_moved: {event.src_path} → {event.dest_path}")
        # Para moved usamos dest_path
        self._schedule_debounce("moved", getattr(event, "dest_path", event.src_path))

    def cancel_all_pending(self) -> None:
        """Cancela todos los timers pendientes."""
        with self._lock:
            for timer in self._pending_timers.values():
                timer.cancel()
            self._pending_timers.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class UltragentSentinel:
    """
    Sistema nervioso periférico de Ultragent.
    
    Monitorea el filesystem y notifica cambios a otros lóbulos
    via HUD.md y signals.json.
    """

    def __init__(
        self,
        watch_path: Optional[Path] = None,
        debounce_seconds: float = DEBOUNCE_SECONDS,
    ):
        self._watch_path = watch_path or AI_DIR / "workspace"
        self._debounce_seconds = debounce_seconds
        self._observer: Optional[Observer] = None
        self._handler: Optional[UltragentEventHandler] = None
        self._event_history: deque[SentinelEvent] = deque(maxlen=MAX_EVENT_HISTORY)
        self._lock = Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Estadísticas
        self._stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_ignored": 0,
            "start_time": None,
        }

    def _on_stable_event(self, event: SentinelEvent) -> None:
        """Callback invocado cuando un evento se estabiliza tras debounce."""
        with self._lock:
            self._stats["events_processed"] += 1
            self._event_history.append(event)

        # Actualizar HUD.md
        self._update_hud(event)

        # Actualizar signals.json
        self._update_signals(event)

        # Registrar en base de datos
        self._log_to_db(event)

    def _update_hud(self, event: SentinelEvent) -> None:
        """Actualiza HUD.md con el evento."""
        try:
            if not HUD_FILE.exists():
                logger.warning("HUD.md no existe, no se puede actualizar")
                return

            content = HUD_FILE.read_text(encoding="utf-8")

            # Buscar sección de eventos del Sentinel
            marker = "## 📡 Eventos Recientes (Sentinel)"
            event_line = event.to_hud_line()

            if marker in content:
                # Insertar nuevo evento después del header de tabla
                lines = content.split("\n")
                new_lines = []
                inserted = False

                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if marker in line and not inserted:
                        # Buscar la línea de separación de tabla (|---|---|)
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if lines[j].startswith("|---"):
                                # Insertar después del separador
                                new_lines.extend(lines[i + 1 : j + 1])
                                new_lines.append(event_line)
                                inserted = True
                                # Saltar las líneas ya añadidas
                                lines = lines[:i + 1] + lines[j + 1:]
                                break
                        break

                if inserted:
                    content = "\n".join(new_lines)
            else:
                # Añadir sección completa al final
                section = f"""

{marker}

| Evento | Archivo | Hora | Fuente |
|--------|---------|------|--------|
{event_line}

"""
                content += section

            HUD_FILE.write_text(content, encoding="utf-8")
            logger.debug(f"HUD.md actualizado con evento: {event.event_type}")

        except PermissionError as e:
            logger.error(f"Permiso denegado al actualizar HUD.md: {e}")
        except IOError as e:
            logger.error(f"Error I/O al actualizar HUD.md: {e}")
        except Exception as e:
            logger.error(f"Error inesperado al actualizar HUD.md: {e}")

    def _update_signals(self, event: SentinelEvent) -> None:
        """Actualiza signals.json para comunicación inter-lóbulos."""
        try:
            signals = {"pending": [], "last_update": None}

            if SIGNALS_FILE.exists():
                try:
                    signals = json.loads(SIGNALS_FILE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    logger.warning("signals.json corrupto, reinicializando")

            # Añadir evento a pending
            signals["pending"].append(event.to_dict())

            # Limitar tamaño de pending
            if len(signals["pending"]) > MAX_EVENT_HISTORY:
                signals["pending"] = signals["pending"][-MAX_EVENT_HISTORY:]

            signals["last_update"] = datetime.now().isoformat()

            SIGNALS_FILE.write_text(
                json.dumps(signals, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("signals.json actualizado")

        except PermissionError as e:
            logger.error(f"Permiso denegado al actualizar signals.json: {e}")
        except IOError as e:
            logger.error(f"Error I/O al actualizar signals.json: {e}")
        except Exception as e:
            logger.error(f"Error al actualizar signals.json: {e}")

    def _log_to_db(self, event: SentinelEvent) -> None:
        """Registra el evento en la base de datos SQLite."""
        try:
            if not TASKS_DB.exists():
                logger.warning("tasks.db no existe")
                return

            conn = sqlite3.connect(
                str(TASKS_DB),
                timeout=10.0,
                isolation_level="DEFERRED",
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")

            conn.execute(
                """
                INSERT INTO sentinel_events 
                (event_type, file_path, timestamp, status, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.src_path,
                    event.timestamp.isoformat(),
                    "PROCESSED",
                    event.source,
                ),
            )
            conn.commit()
            conn.close()
            logger.debug(f"Evento registrado en DB: {event.event_type}")

        except sqlite3.OperationalError as e:
            logger.error(f"Error SQLite: {e}")
        except Exception as e:
            logger.error(f"Error al registrar en DB: {e}")

    def start(self) -> None:
        """Inicia el monitoreo del filesystem."""
        if self._running:
            logger.warning("Sentinel ya está corriendo")
            return

        # Verificar que el directorio a monitorear existe
        if not self._watch_path.exists():
            self._watch_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {self._watch_path}")

        # Crear handler y observer
        self._handler = UltragentEventHandler(
            on_stable_event=self._on_stable_event,
            debounce_seconds=self._debounce_seconds,
        )

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self._watch_path),
            recursive=WATCH_RECURSIVE,
        )

        self._observer.start()
        self._running = True
        self._stats["start_time"] = datetime.now()

        logger.info(
            f"🛡️ Sentinel iniciado - Monitoreando: {self._watch_path} "
            f"(debounce={self._debounce_seconds}s)"
        )

    def stop(self) -> None:
        """Detiene el monitoreo del filesystem."""
        if not self._running:
            return

        if self._handler:
            self._handler.cancel_all_pending()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)

        self._running = False
        logger.info("🛡️ Sentinel detenido")

    def get_status(self) -> dict:
        """Retorna el estado actual del Sentinel."""
        with self._lock:
            uptime = None
            if self._stats["start_time"]:
                uptime = (datetime.now() - self._stats["start_time"]).total_seconds()

            return {
                "running": self._running,
                "watch_path": str(self._watch_path),
                "debounce_seconds": self._debounce_seconds,
                "events_processed": self._stats["events_processed"],
                "uptime_seconds": uptime,
                "recent_events": [e.to_dict() for e in list(self._event_history)[-5:]],
            }

    def get_recent_events(self, limit: int = 5) -> list[dict]:
        """Retorna los últimos N eventos procesados."""
        with self._lock:
            return [e.to_dict() for e in list(self._event_history)[-limit:]]

    def clear_signals(self) -> None:
        """Limpia los eventos pendientes en signals.json."""
        try:
            signals = {
                "pending": [],
                "last_update": datetime.now().isoformat(),
                "cleared_by": "sentinel",
            }
            SIGNALS_FILE.write_text(
                json.dumps(signals, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("signals.json limpiado")
        except Exception as e:
            logger.error(f"Error limpiando signals.json: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

_sentinel_instance: Optional[UltragentSentinel] = None
_sentinel_lock = Lock()


def get_sentinel() -> UltragentSentinel:
    """Obtiene la instancia singleton del Sentinel."""
    global _sentinel_instance
    with _sentinel_lock:
        if _sentinel_instance is None:
            _sentinel_instance = UltragentSentinel()
        return _sentinel_instance


def start_sentinel_thread() -> threading.Thread:
    """Inicia el Sentinel en un thread separado."""
    sentinel = get_sentinel()

    def run():
        sentinel.start()
        # Mantener el thread vivo
        while sentinel._running:
            threading.Event().wait(1.0)

    thread = threading.Thread(target=run, name="SentinelThread", daemon=True)
    thread.start()
    return thread


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Configurar logging para CLI
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("ULTRAGENT SENTINEL v0.1 - Modo Standalone")
    print("=" * 60)
    print(f"Monitoreando: {AI_DIR / 'workspace'}")
    print("Presiona Ctrl+C para detener")
    print("=" * 60)

    sentinel = get_sentinel()
    sentinel.start()

    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        print("\nDeteniendo Sentinel...")
        sentinel.stop()
        sys.exit(0)
