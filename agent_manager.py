"""
ULTRAGENT AGENT MANAGER v2.0 (Sovereign Command Core)
====================================================
Gestor de Agentes Especializados con Pizarra Persistente (C2).

Este módulo implementa el "Centro de Mando" capaz de coordinar agentes
a través de múltiples contextos de Antigravity usando persistencia en disco.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ultragent.agent_manager")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PERSISTENTE
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path.cwd()
AI_DIR = PROJECT_ROOT / ".ai"
SWARM_DIR = AI_DIR / "swarm"
SWARM_DIR.mkdir(parents=True, exist_ok=True)

BLACKBOARD_FILE = SWARM_DIR / "blackboard.json"
MISSIONS_FILE = SWARM_DIR / "missions.json"

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    AUDITOR = "auditor"
    QA = "qa"
    COMMANDER = "commander"
    OPERATOR = "operator"

@dataclass
class AgentMessage:
    sender_role: AgentRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    target_role: Optional[AgentRole] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            sender_role=AgentRole(data["sender_role"]),
            content=data["content"],
            timestamp=data["timestamp"],
            target_role=AgentRole(data["target_role"]) if data.get("target_role") else None,
            task_id=data.get("task_id"),
            metadata=data.get("metadata", {})
        )

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT MANAGER (PERSISTENT BLACKBOARD)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentManager:
    """
    Sistema de Mando y Control (C2) con Persistencia.
    La 'Verdad Matemática' reside en los archivos JSON, no en la memoria.
    """
    def __init__(self):
        self._lock = Lock()
        self._init_files()
        logger.info(f"Sovereign Agent Manager Active. Persistence at {SWARM_DIR}")

    def _init_files(self):
        if not BLACKBOARD_FILE.exists():
            BLACKBOARD_FILE.write_text("[]", encoding="utf-8")
        if not MISSIONS_FILE.exists():
            MISSIONS_FILE.write_text("{}", encoding="utf-8")

    def _load_blackboard(self) -> List[AgentMessage]:
        with self._lock:
            try:
                data = json.loads(BLACKBOARD_FILE.read_text(encoding="utf-8"))
                return [AgentMessage.from_dict(m) for m in data]
            except Exception as e:
                logger.error(f"Failed to load blackboard: {e}")
                return []

    def _save_blackboard(self, messages: List[AgentMessage]):
        with self._lock:
            try:
                data = [asdict(m) for m in messages]
                BLACKBOARD_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to save blackboard: {e}")

    def post_message(self, sender: AgentRole, content: str, target: Optional[AgentRole] = None, task_id: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Publica un mensaje persistente en la pizarra."""
        messages = self._load_blackboard()
        msg = AgentMessage(
            sender_role=AgentRole(sender),
            content=content,
            target_role=AgentRole(target) if target else None,
            task_id=task_id,
            metadata=metadata or {}
        )
        messages.append(msg)
        self._save_blackboard(messages)
        logger.info(f"🛰️ [POST] {sender.upper()} -> {target.upper() if target else 'ALL'}: {content[:50]}...")
        return msg

    def get_messages(self, role: Optional[AgentRole] = None, task_id: Optional[str] = None, limit: int = 20) -> List[AgentMessage]:
        """Recupera mensajes persistentes."""
        messages = self._load_blackboard()
        if role:
            messages = [m for m in messages if m.target_role == role or m.target_role is None]
        if task_id:
            messages = [m for m in messages if m.task_id == task_id]
        return messages[-limit:]

    def clear_blackboard(self):
        self._save_blackboard([])
        logger.info("Blackboard Wiped.")

    def get_role_description(self, role: AgentRole) -> str:
        descriptions = {
            AgentRole.RESEARCHER: "Investigador S.O.T.A. Tu misión es extraer inteligencia técnica y benchmarks.",
            AgentRole.CODER: "Ingeniero de Software Senior. Tu misión es implementar soluciones matemáticas y robustas.",
            AgentRole.AUDITOR: "Auditor de Seguridad y Calidad. Tu misión es encontrar vulnerabilidades y deuda técnica.",
            AgentRole.QA: "Ingeniero de Verificación. Tu misión es automatizar el rigor de las pruebas.",
            AgentRole.COMMANDER: "Orquestador Soberano. Tu misión es dividir misiones complejas y verificar el éxito del enjambre.",
            AgentRole.OPERATOR: "Agente de enlace. Tu misión es ejecutar comandos directos e informar el estado."
        }
        return descriptions.get(role, "Agente especializado.")

    def harvest_cross_chat_intelligence(self, query: str) -> List[str]:
        """
        [ADVANCED] Busca en los logs de TODAS las sesiones de Antigravity.
        Matemáticamente una búsqueda semántica sobre el historial global.
        """
        logger.info(f"🧠 Harvesting Cross-Chat Intelligence for: {query}")
        intelligence = []
        
        # Ruta estándar de Antigravity
        user_home = Path.home()
        brain_dir = user_home / ".gemini" / "antigravity" / "brain"
        
        if not brain_dir.exists():
            return ["No history found."]

        # Usar ripgrep o búsqueda simple en archivos .txt de logs
        # Por simplicidad ahora usamos una búsqueda de texto en los últimos 5 archivos de log recientes
        from librarian import get_librarian
        lib = get_librarian()
        
        # Intentar buscar por similitud si tenemos el modelo cargado
        # ... (Llamada al buscador semántico del librarian sobre los logs)
        
        return ["Feature in development: Semantic Cross-Chat Harvesting."]

# Singleton
_manager_instance = None
def get_agent_manager():
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AgentManager()
    return _manager_instance
