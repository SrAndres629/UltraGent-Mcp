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
    
    def __init__(self, project_root: Optional[str] = None):
        target_root = Path(project_root) if project_root else PROJECT_ROOT
        ai_dir = target_root / os.getenv("AI_CORE_DIR", ".ai")
        self._db_path = ai_dir / "cortex.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = Lock()
        self._librarian = get_librarian(project_root)
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
        
        # Inyectar tags en metadata para persistencia vectorial coherente
        if tags and "tags" not in meta:
            meta["tags"] = ",".join(tags)

        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO memories (content, tags, importance, metadata) VALUES (?, ?, ?, ?)",
                    (content, ",".join(tags), importance, json.dumps(meta))
                )
                memory_id = cursor.lastrowid
                
        # Indexar en Librarian (ChromaDB)
        try:
            # Ahora usamos la indexación real del Librarian con metadata enriquecida
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
                    created_at=datetime.fromisoformat(row[4]) if isinstance(row[4], str) and "T" in row[4] else row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                ))
        return memories

    def get_related_memories(self, node_name: str) -> List[Memory]:
        """Busca memorias vinculadas a un nodo usando búsqueda vectorial semántica."""
        node_clean = node_name.lower().replace(".py", "").replace("_", " ")
        
        # 1. Búsqueda semántica usando el nombre del nodo como query
        # Esto encontrará memorias contextualmente relevantes, no solo coincidencia de texto
        results = self._librarian.semantic_search(
            query=f"concept or context related to {node_clean}", 
            n_results=5,
            node_type="memory"
        )
        
        # 2. Mapear resultados a objetos Memory
        memories = []
        seen_ids = set()
        
        # Mapeo rápido de ID -> Memoria para recuperación O(1)
        all_map = {m.id: m for m in self.get_all_memories()}
        
        for res in results:
            # semantic_search retorna dicts con 'file_path' que es el ID de la memoria en este caso
            try:
                mem_id = int(res.get('file_path', '0')) # En index_memory usamos ID como path
                if mem_id in all_map and mem_id not in seen_ids:
                    memories.append(all_map[mem_id])
                    seen_ids.add(mem_id)
            except Exception:
                continue
                
        # 3. Fallback a coincidencia exacta de tags si la búsqueda semántica es pobre
        if not memories:
            for m in all_map.values():
                if any(node_clean in tag.lower() for tag in m.tags):
                   memories.append(m)
                
        return sorted(memories, key=lambda x: x.importance, reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_cortex_instance: Optional[Cortex] = None
_cortex_lock = Lock()
_current_cortex_project: Optional[str] = None

def get_cortex(project_root: Optional[str] = None) -> Cortex:
    """
    Obtiene la instancia singleton del Cortex.
    
    Args:
        project_root: Optional path to target project. If provided and different
                      from current, a new Cortex instance is created.
    """
    global _cortex_instance, _current_cortex_project
    with _cortex_lock:
        if _cortex_instance is None or _current_cortex_project != project_root:
            _cortex_instance = Cortex(project_root=project_root)
            _current_cortex_project = project_root
        return _cortex_instance

if __name__ == "__main__":
    # Test rápido
    c = get_cortex()
    mid = c.add_memory("Implementada Phase 1 de Mission 002 con éxito", ["mission", "status"])
    print(f"Memoria creada: {mid}")
    for m in c.get_all_memories():
        print(f"- {m.content} ({m.tags})")
