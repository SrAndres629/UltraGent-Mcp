"""
ULTRAGENT - SETUP FILESYSTEM
=============================
Script de inicialización de la estructura de directorios del Core .ai/.

Ejecutar este script ANTES del primer uso de Ultragent para crear
los directorios necesarios y validar el entorno.

Uso:
    python setup_fs.py
    
    # O con uv:
    uv run setup_fs.py
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / ".ai"

# Estructura de directorios a crear
DIRECTORIES = [
    AI_DIR,
    AI_DIR / "logs",
    AI_DIR / "memory",
    AI_DIR / "workspace",
    AI_DIR / "cache",
]

# Archivos críticos que deben existir
CRITICAL_FILES = [
    AI_DIR / "memory.md",
    AI_DIR / "HUD.md",
    AI_DIR / "tasks.db",
]


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE ENTORNO
# ═══════════════════════════════════════════════════════════════════════════════

def validate_environment() -> bool:
    """
    Valida que el entorno tiene las herramientas necesarias.
    
    Returns:
        bool: True si el entorno es válido
    """
    print("\n" + "=" * 60)
    print("ULTRAGENT - VALIDACIÓN DE ENTORNO")
    print("=" * 60)
    
    errors = []
    
    # Verificar Python version
    py_version = sys.version_info
    print(f"✓ Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 11):
        errors.append("Python 3.11+ requerido")
    
    # Verificar uv (opcional pero recomendado)
    if shutil.which("uv"):
        print("✓ uv: Instalado")
    else:
        print("⚠ uv: No encontrado (opcional, pero recomendado)")
        print("  Instalar: pip install uv")
    
    # Verificar que podemos escribir en el directorio
    try:
        test_file = PROJECT_ROOT / ".write_test"
        test_file.touch()
        test_file.unlink()
        print("✓ Permisos de escritura: OK")
    except Exception as e:
        errors.append(f"Sin permisos de escritura: {e}")
    
    if errors:
        print("\n❌ ERRORES ENCONTRADOS:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("\n✓ Entorno válido")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CREACIÓN DE ESTRUCTURA
# ═══════════════════════════════════════════════════════════════════════════════

def create_directories() -> None:
    """Crea la estructura de directorios del Core .ai/."""
    print("\n" + "-" * 60)
    print("CREANDO ESTRUCTURA DE DIRECTORIOS")
    print("-" * 60)
    
    for directory in DIRECTORIES:
        if directory.exists():
            print(f"⊙ {directory.relative_to(PROJECT_ROOT)} (ya existe)")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory.relative_to(PROJECT_ROOT)} (creado)")


def initialize_database() -> None:
    """Inicializa la base de datos SQLite si no existe."""
    print("\n" + "-" * 60)
    print("INICIALIZANDO BASE DE DATOS")
    print("-" * 60)
    
    db_path = AI_DIR / "tasks.db"
    db_existed = db_path.exists()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Tabla de tareas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'PENDIENTE',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            lobe TEXT,
            description TEXT,
            log TEXT
        )
    """)
    
    # Tabla de log de protocolos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS protocol_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_id INTEGER,
            status TEXT,
            assimilated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tabla de eventos del Sentinel (para futuro uso)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            file_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'PENDING',
            processed_at DATETIME,
            result TEXT
        )
    """)
    
    # Habilitar WAL mode para mejor concurrencia
    cursor.execute("PRAGMA journal_mode=WAL")
    
    conn.commit()
    conn.close()
    
    if db_existed:
        print(f"⊙ tasks.db (ya existía, tablas verificadas)")
    else:
        print(f"✓ tasks.db (creado con esquema inicial)")


def create_default_files() -> None:
    """Crea archivos por defecto si no existen."""
    print("\n" + "-" * 60)
    print("VERIFICANDO ARCHIVOS CRÍTICOS")
    print("-" * 60)
    
    # memory.md
    memory_file = AI_DIR / "memory.md"
    if not memory_file.exists():
        memory_file.write_text(
            "# 🧠 ULTRAGENT - MEMORIA EPISÓDICA\n"
            "> *\"El código no es texto; es infraestructura.\"*\n\n"
            "---\n\n"
            f"## 📅 {datetime.now().strftime('%Y-%m-%d')} | INICIALIZACIÓN\n\n"
            "Sistema inicializado por setup_fs.py.\n",
            encoding="utf-8",
        )
        print(f"✓ memory.md (creado con plantilla)")
    else:
        print(f"⊙ memory.md (ya existe)")
    
    # HUD.md
    hud_file = AI_DIR / "HUD.md"
    if not hud_file.exists():
        hud_file.write_text(
            "# 📊 ULTRAGENT HUD - Panel de Observabilidad\n"
            f"> Última actualización: {datetime.now().isoformat()}\n\n"
            "---\n\n"
            "## 🔋 Estado del Sistema\n\n"
            "| Componente | Estado |\n"
            "|------------|--------|\n"
            "| Core .ai/ | 🟢 ONLINE |\n\n"
            "---\n",
            encoding="utf-8",
        )
        print(f"✓ HUD.md (creado con plantilla)")
    else:
        print(f"⊙ HUD.md (ya existe)")
    
    # .gitignore para .ai/
    gitignore = AI_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Ignorar archivos sensibles\n"
            "*.log\n"
            "tasks.db-wal\n"
            "tasks.db-shm\n"
            "cache/*\n"
            "workspace/*\n",
            encoding="utf-8",
        )
        print(f"✓ .gitignore (creado)")
    else:
        print(f"⊙ .gitignore (ya existe)")


def create_env_file() -> None:
    """Copia .env.example a .env si no existe."""
    print("\n" + "-" * 60)
    print("CONFIGURACIÓN DE ENTORNO")
    print("-" * 60)
    
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / ".env.example"
    
    if env_file.exists():
        print(f"⊙ .env (ya existe)")
    elif example_file.exists():
        shutil.copy(example_file, env_file)
        print(f"✓ .env (copiado desde .env.example)")
        print(f"  ⚠ Recuerda configurar tus API keys en .env")
    else:
        print(f"⚠ .env.example no encontrado")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTE FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    """Imprime un resumen del estado final."""
    print("\n" + "=" * 60)
    print("RESUMEN DE INICIALIZACIÓN")
    print("=" * 60)
    
    print("\n📁 Estructura de directorios:")
    for directory in DIRECTORIES:
        status = "✓" if directory.exists() else "❌"
        print(f"   {status} {directory.relative_to(PROJECT_ROOT)}/")
    
    print("\n📄 Archivos críticos:")
    for file in CRITICAL_FILES:
        if file.exists():
            size = file.stat().st_size
            print(f"   ✓ {file.name} ({size} bytes)")
        else:
            print(f"   ❌ {file.name} (no encontrado)")
    
    print("\n" + "=" * 60)
    print("✓ ULTRAGENT CORE INICIALIZADO")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("  1. Configura tus API keys en .env")
    print("  2. Ejecuta: uv sync")
    print("  3. Inicia el servidor: uv run mcp_server.py")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """Función principal de inicialización."""
    print("\n" + "█" * 60)
    print("     ULTRAGENT - FILESYSTEM SETUP v0.1")
    print("█" * 60)
    
    # Validar entorno
    if not validate_environment():
        print("\n❌ Corrija los errores antes de continuar.")
        return 1
    
    # Crear estructura
    create_directories()
    initialize_database()
    create_default_files()
    create_env_file()
    
    # Mostrar resumen
    print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
