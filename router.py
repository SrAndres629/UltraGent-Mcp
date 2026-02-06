"""
ULTRAGENT OMNI-ROUTER v0.1
==========================
Módulo de arbitraje de APIs con failover automático y gestión de tokens.

Implementa:
- Circuit Breaker para providers fallidos
- Exponential Backoff con reintentos
- Budget Guardian para límites de tokens
- Swarms para procesamiento paralelo
- Clasificación automática de tareas por tier
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
TASKS_DB = AI_DIR / "tasks.db"
HUD_FILE = AI_DIR / "HUD.md"
ROUTER_LOG = AI_DIR / "logs" / "router.log"

# Configuración de resiliencia
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
BASE_TIMEOUT = 30.0
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_RESET_SECONDS = 60
SESSION_TOKEN_LIMIT = 100_000

# Logger
logger = logging.getLogger("ultragent.router")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS Y TIPOS
# ═══════════════════════════════════════════════════════════════════════════════

class Tier(Enum):
    """Tiers de inteligencia del Omni-Router."""
    VISUAL = "visual"       # Kimi K2.5 via NVIDIA NIM
    CODING = "coding"       # DeepSeek V3 via SiliconFlow
    SPEED = "speed"         # Llama 3.3 via Groq
    STRATEGIC = "strategic" # Gemini / Claude


class ProviderStatus(Enum):
    """Estado del Circuit Breaker."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    INACTIVE = "inactive"


@dataclass
class RouterResponse:
    """Respuesta del router."""
    success: bool
    content: str
    provider: str
    tier: Tier
    tokens_used: int
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ProviderState:
    """Estado de un proveedor para Circuit Breaker."""
    name: str
    status: ProviderStatus = ProviderStatus.ACTIVE
    failures: int = 0
    last_failure: Optional[datetime] = None
    total_calls: int = 0
    total_tokens: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE TAREAS
# ═══════════════════════════════════════════════════════════════════════════════

TASK_TO_TIER: dict[str, Tier] = {
    # SPEED tier - operaciones rápidas
    "fix_syntax": Tier.SPEED,
    "unit_test": Tier.SPEED,
    "boilerplate": Tier.SPEED,
    "translate": Tier.SPEED,
    "format": Tier.SPEED,
    "quick_question": Tier.SPEED,
    
    # CODING tier - generación de código
    "generate_code": Tier.CODING,
    "refactor": Tier.CODING,
    "implement_feature": Tier.CODING,
    "debug": Tier.CODING,
    "optimize": Tier.CODING,
    "api_design": Tier.CODING,
    
    # VISUAL tier - multimodal
    "analyze_image": Tier.VISUAL,
    "diagram_to_code": Tier.VISUAL,
    "ui_review": Tier.VISUAL,
    "swarm": Tier.VISUAL,
    
    # STRATEGIC tier - decisiones críticas
    "architecture_review": Tier.STRATEGIC,
    "security_audit": Tier.STRATEGIC,
    "final_review": Tier.STRATEGIC,
    "critical_decision": Tier.STRATEGIC,
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROVEEDORES BASE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseProvider(ABC):
    """Clase base para proveedores de LLM (Open/Closed Principle)."""
    
    def __init__(self, name: str, api_key: str, base_url: str, model: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.state = ProviderState(name=name)
    
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """
        Ejecuta una completion.
        
        Returns:
            tuple[str, int]: (contenido, tokens_usados)
        """
        pass
    
    def is_available(self) -> bool:
        """Verifica si el proveedor está disponible (Circuit Breaker)."""
        if self.state.status == ProviderStatus.INACTIVE:
            # Verificar si pasó el tiempo de reset
            if self.state.last_failure:
                elapsed = datetime.now() - self.state.last_failure
                if elapsed > timedelta(seconds=CIRCUIT_BREAKER_RESET_SECONDS):
                    self.state.status = ProviderStatus.DEGRADED
                    self.state.failures = 0
                    logger.info(f"Provider {self.name} reseteado a DEGRADED")
                    return True
            return False
        return True
    
    def mark_success(self) -> None:
        """Marca una llamada exitosa."""
        self.state.total_calls += 1
        if self.state.status == ProviderStatus.DEGRADED:
            self.state.status = ProviderStatus.ACTIVE
            logger.info(f"Provider {self.name} restaurado a ACTIVE")
    
    def mark_failure(self) -> None:
        """Marca una llamada fallida (Circuit Breaker)."""
        self.state.failures += 1
        self.state.last_failure = datetime.now()
        
        if self.state.failures >= CIRCUIT_BREAKER_THRESHOLD:
            self.state.status = ProviderStatus.INACTIVE
            logger.warning(
                f"Circuit Breaker OPEN para {self.name} "
                f"(failures={self.state.failures})"
            )


class OpenAICompatibleProvider(BaseProvider):
    """Proveedor compatible con API de OpenAI."""
    
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Ejecuta completion via API compatible con OpenAI."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async with httpx.AsyncClient(timeout=BASE_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit en {self.name}")
            
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            
            return content, tokens


# ═══════════════════════════════════════════════════════════════════════════════
# PROVEEDORES ESPECÍFICOS
# ═══════════════════════════════════════════════════════════════════════════════

class GroqProvider(OpenAICompatibleProvider):
    """Proveedor Groq (Tier SPEED)."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        super().__init__(
            name="groq",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
        )


class SiliconFlowProvider(OpenAICompatibleProvider):
    """Proveedor SiliconFlow / DeepSeek (Tier CODING)."""
    
    def __init__(self):
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
        super().__init__(
            name="siliconflow",
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
            model="deepseek-ai/DeepSeek-V3",
        )


class NVIDIANIMProvider(OpenAICompatibleProvider):
    """Proveedor NVIDIA NIM / Kimi (Tier VISUAL)."""
    
    def __init__(self):
        api_key = os.getenv("NVIDIA_NIM_API_KEY", "")
        super().__init__(
            name="nvidia_nim",
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model="nvidia/llama-3.1-nemotron-70b-instruct",
        )


class GeminiProvider(BaseProvider):
    """Proveedor Google Gemini (Tier STRATEGIC)."""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        super().__init__(
            name="gemini",
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.0-flash",
        )
    
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """Ejecuta completion via Gemini API."""
        
        # Convertir formato OpenAI a Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        # Asegurar que el path del modelo sea correcto sin duplicar /models/
        model_name = self.model.split("/")[-1]
        clean_base = self.base_url.rstrip("/")
        if clean_base.endswith("/models"):
            clean_base = clean_base[:-7]
            
        url = (
            f"{clean_base}/models/{model_name}:generateContent"
            f"?key={self.api_key}"
        )
        print(f"DEBUG - Final Gemini URL: {url}")
        
        async with httpx.AsyncClient(timeout=BASE_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit en {self.name}")
            
            response.raise_for_status()
            data = response.json()
            
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            # Gemini no siempre retorna usage
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            
            return content, tokens


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════

class RouterError(Exception):
    """Error base del router."""
    pass


class RateLimitError(RouterError):
    """Error de rate limit (429)."""
    pass


class BudgetExceededError(RouterError):
    """Error de presupuesto excedido."""
    pass


class AllProvidersFailedError(RouterError):
    """Todos los proveedores fallaron."""
    pass


class InvalidTaskError(RouterError):
    """Tipo de tarea no reconocido."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# BUDGET GUARDIAN
# ═══════════════════════════════════════════════════════════════════════════════

class BudgetGuard:
    """Controla el consumo de tokens por sesión."""
    
    def __init__(self, session_limit: int = SESSION_TOKEN_LIMIT):
        self._limit = session_limit
        self._used = 0
        self._lock = Lock()
        self._tier_usage: dict[str, int] = {}
    
    def check_and_consume(self, tokens: int, tier: str) -> bool:
        """
        Verifica y consume tokens.
        
        Raises:
            BudgetExceededError: Si se excede el límite
        """
        with self._lock:
            if self._used + tokens > self._limit:
                raise BudgetExceededError(
                    f"Budget excedido: {self._used}/{self._limit} tokens"
                )
            self._used += tokens
            self._tier_usage[tier] = self._tier_usage.get(tier, 0) + tokens
            return True
    
    def get_usage(self) -> dict:
        """Retorna estadísticas de uso."""
        with self._lock:
            return {
                "total_used": self._used,
                "limit": self._limit,
                "remaining": self._limit - self._used,
                "by_tier": dict(self._tier_usage),
            }
    
    def reset(self) -> None:
        """Resetea el contador de sesión."""
        with self._lock:
            self._used = 0
            self._tier_usage.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# OMNI-ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

# --- RL: Contextual Bandits (Weight Manager) ---
import json
from pathlib import Path
from typing import Dict, Any

# Ensure we have the path constants we need
try:
    from mcp_server import AI_DIR
except ImportError:
    # Fallback if circular import or missing
    AI_DIR = Path.cwd() / ".ai"

class WeightManager:
    """Gestor de pesos para Reinforcement Learning (Lite - Contextual Bandits)."""
    def __init__(self):
        self._weights_file = AI_DIR / "cortex" / "weights.json"
        self._weights: Dict[str, float] = {}
        self._load_weights()
    
    def _load_weights(self):
        if self._weights_file.exists():
            try:
                content = self._weights_file.read_text(encoding="utf-8")
                self._weights = json.loads(content)
            except Exception as e:
                logger.warning(f"RL: Failed to load weights ({e}). Resetting to empty.")
                self._weights = {}
        else:
            # Create dir if needed
            self._weights_file.parent.mkdir(parents=True, exist_ok=True)
            self._weights = {}

    def get_weight(self, key: str, default: float = 1.0) -> float:
        return self._weights.get(key, default)

    def update_weight(self, key: str, reward: float, learning_rate: float = 0.1):
        """Alpha-update: W_new = W_old + LR * (Reward - W_old)"""
        old_W = self.get_weight(key)
        new_W = old_W + learning_rate * (reward - old_W)
        self._weights[key] = round(new_W, 4)
        self._save_weights()

    def _save_weights(self):
        try:
            self._weights_file.write_text(json.dumps(self._weights, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"RL: Failed to save weights: {e}")

# Global RL Manager instance
rl_manager = WeightManager()

class OmniRouter:
    """
    Router inteligente con arbitraje de APIs y failover automático.
    
    Características:
    - Clasificación automática de tareas por tier
    - Circuit Breaker por proveedor
    - Failover en <500ms
    - Budget Guardian
    - Swarms para procesamiento paralelo
    """
    
    def __init__(self):
        self._providers: dict[Tier, list[BaseProvider]] = {
            Tier.SPEED: [GroqProvider()],
            Tier.CODING: [SiliconFlowProvider()],
            Tier.VISUAL: [NVIDIANIMProvider()],
            Tier.STRATEGIC: [GeminiProvider()],
        }
        
        # Fallbacks por tier
        self._fallbacks: dict[Tier, list[Tier]] = {
            Tier.SPEED: [Tier.CODING, Tier.STRATEGIC],
            Tier.CODING: [Tier.SPEED, Tier.STRATEGIC],
            Tier.VISUAL: [Tier.CODING, Tier.STRATEGIC, Tier.SPEED],
            Tier.STRATEGIC: [Tier.CODING, Tier.SPEED],
        }
        
        self._budget = BudgetGuard()
        self._lock = Lock()
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "failovers": 0,
        }
    
    def classify_task(self, task_type: str) -> Tier:
        """Clasifica una tarea en su tier correspondiente."""
        tier = TASK_TO_TIER.get(task_type.lower())
        if tier is None:
            # Default a CODING para tareas desconocidas
            logger.warning(f"Task type desconocido: {task_type}, usando CODING")
            return Tier.CODING
        return tier
    
    async def _call_with_retry(
        self,
        provider: BaseProvider,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Llama al proveedor con exponential backoff y Agentic Self-Repair (ToolMedic)."""
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    wait_time = INITIAL_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(f"Retry {attempt}/{MAX_RETRIES} for {provider.name} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                
                content, tokens = await provider.complete(
                    messages, temperature, max_tokens
                )
                
                if content:
                    return content, tokens
                    
            except Exception as e:
                last_error = e
                # Check for specific "Repairable" errors
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    logger.error(f"🚑 ToolMedic Alert: {provider.name} has AUTH ERROR. Triggering Self-Repair.")
                    # In a full autonomous system, this would call:
                    # await get_mechanic().run_task(f"Fix API Key for {provider.name}. Error: {e}")
                    # For now, we log the alert to be picked up by the Supervisor.
        
        raise RouterError(f"All retries failed for {provider.name}. Last error: {last_error}")
    
    async def route_task(
        self,
        task_type: str,
        payload: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> RouterResponse:
        """
        Enruta una tarea al tier apropiado con failover automático.
        
        Args:
            task_type: Tipo de tarea (generate_code, fix_syntax, etc.)
            payload: Contenido/prompt de la tarea
            system_prompt: Prompt de sistema opcional
            temperature: Temperatura del modelo
            max_tokens: Máximo de tokens de salida
            
        Returns:
            RouterResponse con el resultado
        """
        tier = self.classify_task(task_type)
        
        if tier not in self._providers:
            # Fallback a CODING si el tier no existe
            tier = Tier.CODING

        with self._lock:
            self._stats["total_calls"] += 1
        
        # RL: PROVIDER SELECTION based on Weights
        # We sort providers by their weight for this specific task type
        providers = self._providers.get(tier, [])
        
        # Sort logic: Weight * Availability
        # Keys are "provider_name:task_type"
        providers.sort(
            key=lambda p: rl_manager.get_weight(f"{p.name}:{task_type}", 1.0),
            reverse=True
        )

        with self._lock:
            self._stats["total_calls"] += 1
        
        # Construir mensajes
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": payload})
        
        # Intentar tier primario
        tiers_to_try = [tier] + self._fallbacks.get(tier, [])
        
        for current_tier in tiers_to_try:
            # Re-fetch in case tier changed loop (though providers list above was tier specific)
            # Actually we need to re-sort for each fallback tier
            current_providers = self._providers.get(current_tier, [])
            current_providers.sort(
                 key=lambda p: rl_manager.get_weight(f"{p.name}:{task_type}", 1.0),
                 reverse=True
            )
            
            for provider in current_providers:
                if not provider.is_available():
                    continue
                
                if not provider.api_key:
                    logger.debug(f"API key no configurada para {provider.name}")
                    continue
                
                try:
                    start = time.perf_counter()
                    content, tokens = await self._call_with_retry(
                        provider, messages, temperature, max_tokens
                    )
                    latency = (time.perf_counter() - start) * 1000
                    
                    # RL: REWARD SIGNAL (Success = +0.1, Fast = +0.05)
                    reward = 1.0
                    if latency < 2000: reward += 0.2
                    rl_manager.update_weight(f"{provider.name}:{task_type}", reward)
                    
                    # Registrar tokens
                    self._budget.check_and_consume(tokens, current_tier.value)
                    
                    # Registrar en DB
                    self._log_to_db(provider.name, current_tier, tokens, latency, True)
                    
                    with self._lock:
                        self._stats["successful_calls"] += 1
                        if current_tier != tier:
                            self._stats["failovers"] += 1
                    
                    return RouterResponse(
                        success=True,
                        content=content,
                        provider=provider.name,
                        tier=current_tier,
                        tokens_used=tokens,
                        latency_ms=latency,
                    )
                    
                except (RateLimitError, RouterError) as e:
                    # RL: PENALTY SIGNAL (Failure = -0.5)
                    rl_manager.update_weight(f"{provider.name}:{task_type}", 0.0)
                    logger.warning(f"Failover desde {provider.name}: {e}")
                    continue
                    
                except BudgetExceededError as e:
                    logger.error(f"Budget excedido: {e}")
                    return RouterResponse(
                        success=False,
                        content="",
                        provider=provider.name,
                        tier=current_tier,
                        tokens_used=0,
                        latency_ms=0,
                        error=str(e),
                    )
        
        # Todos los proveedores fallaron
        with self._lock:
            self._stats["failed_calls"] += 1
        
        raise AllProvidersFailedError(
            f"Todos los proveedores fallaron para tier {tier.value}"
        )
    
    async def ask_swarm(
        self,
        task: str,
        subtasks: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[RouterResponse]:
        """
        Procesa subtareas en paralelo (Swarms).
        
        Args:
            task: Descripción general de la tarea
            subtasks: Lista de subtareas a procesar
            system_prompt: Prompt de sistema compartido
            
        Returns:
            Lista de RouterResponse, uno por subtarea
        """
        logger.info(f"🐝 Swarm iniciado: {len(subtasks)} subtareas")
        
        # Clasificar cada subtarea
        async def process_subtask(i: int, subtask: str) -> RouterResponse:
            try:
                # Inferir tipo de tarea del contenido
                task_type = self._infer_task_type(subtask)
                return await self.route_task(
                    task_type=task_type,
                    payload=subtask,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                return RouterResponse(
                    success=False,
                    content="",
                    provider="none",
                    tier=Tier.SPEED,
                    tokens_used=0,
                    latency_ms=0,
                    error=str(e),
                )
        
        # Ejecutar en paralelo
        tasks = [
            process_subtask(i, subtask)
            for i, subtask in enumerate(subtasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convertir excepciones a RouterResponse
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                final_results.append(RouterResponse(
                    success=False,
                    content="",
                    provider="none",
                    tier=Tier.SPEED,
                    tokens_used=0,
                    latency_ms=0,
                    error=str(r),
                ))
            else:
                final_results.append(r)
        
        logger.info(
            f"🐝 Swarm completado: "
            f"{sum(1 for r in final_results if r.success)}/{len(subtasks)} exitosos"
        )
        
        return final_results
    
    def _infer_task_type(self, content: str) -> str:
        """Infiere el tipo de tarea del contenido."""
        content_lower = content.lower()
        
        if any(w in content_lower for w in ["fix", "error", "bug", "syntax"]):
            return "fix_syntax"
        if any(w in content_lower for w in ["test", "unittest", "pytest"]):
            return "unit_test"
        if any(w in content_lower for w in ["refactor", "clean", "optimize"]):
            return "refactor"
        if any(w in content_lower for w in ["implement", "create", "build"]):
            return "generate_code"
        if any(w in content_lower for w in ["review", "audit", "security"]):
            return "security_audit"
        
        return "generate_code"  # Default
    
    def _log_to_db(
        self,
        provider: str,
        tier: Tier,
        tokens: int,
        latency: float,
        success: bool,
    ) -> None:
        """Registra la llamada en la base de datos."""
        try:
            if not TASKS_DB.exists():
                return
            
            conn = sqlite3.connect(str(TASKS_DB), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Verificar si existe la tabla router_logs
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='router_logs'"
            )
            if not cursor.fetchone():
                conn.execute("""
                    CREATE TABLE router_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        tokens INTEGER,
                        latency_ms REAL,
                        success INTEGER
                    )
                """)
            
            conn.execute(
                """
                INSERT INTO router_logs 
                (timestamp, provider, tier, tokens, latency_ms, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    provider,
                    tier.value,
                    tokens,
                    latency,
                    1 if success else 0,
                ),
            )
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging to DB: {e}")
    
    def get_status(self) -> dict:
        """Retorna estado del router."""
        providers_status = {}
        for tier, providers in self._providers.items():
            for p in providers:
                providers_status[p.name] = {
                    "tier": tier.value,
                    "status": p.state.status.value,
                    "failures": p.state.failures,
                    "has_key": bool(p.api_key),
                }
        
        return {
            "stats": dict(self._stats),
            "providers": providers_status,
            "budget": self._budget.get_usage(),
        }
    
    def get_token_usage(self) -> dict:
        """Retorna uso de tokens para HUD."""
        return self._budget.get_usage()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_router_instance: Optional[OmniRouter] = None
_router_lock = Lock()


def get_router() -> OmniRouter:
    """Obtiene la instancia singleton del router."""
    global _router_instance
    with _router_lock:
        if _router_instance is None:
            _router_instance = OmniRouter()
        return _router_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def _test_router():
    """Test básico del router."""
    router = get_router()
    
    print("=" * 60)
    print("OMNI-ROUTER v0.1 - Test")
    print("=" * 60)
    print(f"Status: {json.dumps(router.get_status(), indent=2)}")
    print("=" * 60)
    
    # Test de clasificación
    tests = ["generate_code", "fix_syntax", "architecture_review", "unknown"]
    for task in tests:
        tier = router.classify_task(task)
        print(f"  {task} → {tier.value}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    asyncio.run(_test_router())
