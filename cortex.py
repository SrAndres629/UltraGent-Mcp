"""
ULTRAGENT CORTEX v0.1
=====================
Módulo de memoria atómica adaptado de Memora.

Implementa:
- Almacenamiento de "Átomos de Memoria" (Hechos, Decisiones, Contexto)
- Persistencia Híbrida: SQLite (Metadatos) + ChromaDB (Vectores)
- Vinculación semántica entre memorias
- Deduplicación proactiva (vía LLM)
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional, List, Dict

from librarian import get_librarian

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path.cwd() # Isolate memory per project (User Request)
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
CORTEX_DB = AI_DIR / "cortex.db"

logger = logging.getLogger("ultragent.cortex")

# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Memory:
    """Representa un átomo de conocimiento."""
    content: str
    id: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "metadata": self.metadata
        })

# ═══════════════════════════════════════════════════════════════════════════════
# CORTEX MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class Cortex:
    """Gestor de memoria atómica y conocimiento semántico."""
    
    def __init__(self):
        self._db_path = CORTEX_DB
        self._lock = Lock()
        self._librarian = get_librarian()
        self._init_db()
        logger.info(f"Cortex inicializado: {self._db_path}")

    def _init_db(self):
        """Inicializa la base de datos SQLite para metadatos."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        tags TEXT,
                        importance REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS links (
                        source_id INTEGER,
                        target_id INTEGER,
                        edge_type TEXT,
                        weight REAL DEFAULT 1.0,
                        PRIMARY KEY (source_id, target_id, edge_type)
                    )
                """)

    def add_memory(self, content: str, tags: List[str] = None, importance: float = 1.0, meta: Dict = None) -> int:
        """Añade un nuevo átomo de memoria y lo indexa vectorialmente."""
        tags = tags or []
        meta = meta or {}
        
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO memories (content, tags, importance, metadata) VALUES (?, ?, ?, ?)",
                    (content, ",".join(tags), importance, json.dumps(meta))
                )
                memory_id = cursor.lastrowid
                
        # Indexar en Librarian (ChromaDB)
        virtual_path = f"memory://{memory_id}"
        try:
            # Ahora usamos la indexación real del Librarian
            self._librarian.index_memory(str(memory_id), content, tags)
            logger.info(f"Memoria [{memory_id}] guardada e indexada: {content[:50]}...")
        except Exception as e:
            logger.warning(f"No se pudo indexar la memoria {memory_id}: {e}")
            
        return memory_id

    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Busca memorias relacionadas por semántica (vía Librarian)."""
        results = self._librarian.semantic_search(
            query=query, 
            n_results=limit, 
            node_type="memory"
        )
        # Enriquecer con datos de SQLite si es necesario
        return results

    def get_all_memories(self) -> list[Memory]:
        """Retorna todas las memorias guardadas."""
        memories = []
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT id, content, tags, importance, created_at, metadata FROM memories")
            for row in cursor:
                memories.append(Memory(
                    id=row[0],
                    content=row[1],
                    tags=row[2].split(",") if row[2] else [],
                    importance=row[3],
                    created_at=datetime.fromisoformat(row[4]) if "T" in row[4] else row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                ))
        return memories

_cortex_instance = None
def get_cortex():
    global _cortex_instance
    if _cortex_instance is None:
        _cortex_instance = Cortex()
    return _cortex_instance

if __name__ == "__main__":
    # Test rápido
    c = get_cortex()
    mid = c.add_memory("Implementada Phase 1 de Mission 002 con éxito", ["mission", "status"])
    print(f"Memoria creada: {mid}")
    for m in c.get_all_memories():
        print(f"- {m.content} ({m.tags})")
