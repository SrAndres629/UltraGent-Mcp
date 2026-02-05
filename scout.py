"""
ULTRAGENT SCOUT v0.1
====================
Módulo de investigación externa con integración GitHub API.

Implementa:
- Búsqueda de repositorios "Gold Standard" por topic, lenguaje y estrellas
- Filtrado por métricas de salud (commits, issues, tests, typing)
- Descarga de esqueletos de archivos para análisis comparativo
- Rate limiting y manejo de cuotas de GitHub API
- Cache en ChromaDB para evitar llamadas repetidas
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
CACHE_DIR = AI_DIR / "cache" / "scout"
REPORTS_DIR = AI_DIR / "reports"

# GitHub API
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Thresholds para "Gold Standard"
GOLD_STANDARD_THRESHOLDS = {
    "min_stars": 500,
    "min_forks": 50,
    "max_open_issues_ratio": 0.3,
    "max_days_since_update": 180,
}

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = 30  # Sin token: 10, con token: 30
REQUEST_DELAY_SECONDS = 2.0

# Logger
logger = logging.getLogger("ultragent.scout")


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RepositoryHealth:
    """Métricas de salud de un repositorio."""
    
    name: str
    full_name: str
    url: str
    stars: int
    forks: int
    open_issues: int
    language: str
    topics: list[str]
    last_updated: datetime
    has_tests: bool = False
    has_typing: bool = False
    has_readme: bool = True
    commit_frequency: float = 0.0  # commits/week
    
    @property
    def health_score(self) -> float:
        """Calcula score de salud (0-100)."""
        score = 0.0
        
        # Estrellas (max 30 puntos)
        score += min(30, (self.stars / 1000) * 10)
        
        # Forks (max 15 puntos)
        score += min(15, (self.forks / 100) * 5)
        
        # Issues ratio (max 15 puntos)
        if self.stars > 0:
            issues_ratio = self.open_issues / self.stars
            score += max(0, 15 - (issues_ratio * 50))
        else:
            score += 7.5
        
        # Actividad reciente (max 20 puntos)
        days_old = (datetime.now() - self.last_updated).days
        score += max(0, 20 - (days_old / 10))
        
        # Tests (10 puntos)
        if self.has_tests:
            score += 10
        
        # Typing (10 puntos)
        if self.has_typing:
            score += 10
        
        return min(100, max(0, score))
    
    def is_gold_standard(self) -> bool:
        """Verifica si el repo cumple criterios Gold Standard."""
        if self.stars < GOLD_STANDARD_THRESHOLDS["min_stars"]:
            return False
        if self.forks < GOLD_STANDARD_THRESHOLDS["min_forks"]:
            return False
        
        days_old = (datetime.now() - self.last_updated).days
        if days_old > GOLD_STANDARD_THRESHOLDS["max_days_since_update"]:
            return False
        
        if self.stars > 0:
            issues_ratio = self.open_issues / self.stars
            if issues_ratio > GOLD_STANDARD_THRESHOLDS["max_open_issues_ratio"]:
                return False
        
        return True
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "url": self.url,
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "language": self.language,
            "topics": self.topics,
            "last_updated": self.last_updated.isoformat(),
            "has_tests": self.has_tests,
            "has_typing": self.has_typing,
            "health_score": self.health_score,
            "is_gold_standard": self.is_gold_standard(),
        }


@dataclass
class ScoutResult:
    """Resultado de una operación Scout."""
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    rate_limit_remaining: int = -1
    cached: bool = False


class GitHubAPIError(Exception):
    """Error de la API de GitHub."""
    pass


class RateLimitExceededError(GitHubAPIError):
    """Rate limit excedido."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SCOUT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class ScoutAgent:
    """
    Agente de investigación para repositorios GitHub.
    
    Busca y analiza repositorios "Gold Standard" para
    benchmarking y auditoría comparativa.
    """
    
    def __init__(self):
        self._token = GITHUB_TOKEN
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        self._lock = Lock()
        self._last_request_time: Optional[datetime] = None
        self._rate_limit_remaining = 60 if self._token else 10
        self._rate_limit_reset: Optional[datetime] = None
        
        # Estadísticas
        self._stats = {
            "searches": 0,
            "repos_analyzed": 0,
            "files_downloaded": 0,
            "cache_hits": 0,
            "api_calls": 0,
        }
        
        logger.info(
            f"ScoutAgent inicializado "
            f"(token={'✓' if self._token else '✗'})"
        )
    
    def _get_headers(self) -> dict:
        """Genera headers para la API de GitHub."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "UltragentScout/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    async def _respect_rate_limit(self) -> None:
        """Espera si es necesario para respetar rate limits."""
        # Verificar si estamos en cooldown de rate limit
        if self._rate_limit_reset and datetime.now() < self._rate_limit_reset:
            wait_seconds = (self._rate_limit_reset - datetime.now()).total_seconds()
            if wait_seconds > 0:
                logger.warning(f"Rate limit - esperando {wait_seconds:.0f}s")
                await asyncio.sleep(min(wait_seconds, 60))
        
        # Delay entre requests
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < REQUEST_DELAY_SECONDS:
                await asyncio.sleep(REQUEST_DELAY_SECONDS - elapsed)
        
        self._last_request_time = datetime.now()
    
    def _update_rate_limit(self, response: httpx.Response) -> None:
        """Actualiza información de rate limit desde headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        
        if remaining:
            self._rate_limit_remaining = int(remaining)
        
        if reset:
            self._rate_limit_reset = datetime.fromtimestamp(int(reset))
    
    async def _api_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> ScoutResult:
        """Realiza una request a la API de GitHub."""
        await self._respect_rate_limit()
        
        url = f"{GITHUB_API_BASE}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                )
                
                self._update_rate_limit(response)
                
                with self._lock:
                    self._stats["api_calls"] += 1
                
                if response.status_code == 403:
                    if self._rate_limit_remaining == 0:
                        raise RateLimitExceededError(
                            f"Rate limit excedido. Reset: {self._rate_limit_reset}"
                        )
                    raise GitHubAPIError(f"Forbidden: {response.text}")
                
                if response.status_code == 404:
                    return ScoutResult(
                        success=False,
                        error="Not found",
                        rate_limit_remaining=self._rate_limit_remaining,
                    )
                
                response.raise_for_status()
                
                return ScoutResult(
                    success=True,
                    data=response.json(),
                    rate_limit_remaining=self._rate_limit_remaining,
                )
                
        except httpx.TimeoutException:
            return ScoutResult(success=False, error="Timeout")
        except RateLimitExceededError as e:
            return ScoutResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"API error: {e}")
            return ScoutResult(success=False, error=str(e))
    
    async def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        topic: Optional[str] = None,
        min_stars: int = 500,
        max_results: int = 10,
    ) -> list[RepositoryHealth]:
        """
        Busca repositorios en GitHub.
        
        Args:
            query: Términos de búsqueda
            language: Filtrar por lenguaje (python, javascript, etc.)
            topic: Filtrar por topic
            min_stars: Mínimo de estrellas
            max_results: Máximo de resultados
            
        Returns:
            list[RepositoryHealth]: Repositorios ordenados por health score
        """
        with self._lock:
            self._stats["searches"] += 1
        
        # Construir query
        search_query = query
        if language:
            search_query += f" language:{language}"
        if topic:
            search_query += f" topic:{topic}"
        search_query += f" stars:>={min_stars}"
        
        params = {
            "q": search_query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 30),
        }
        
        result = await self._api_request("/search/repositories", params)
        
        if not result.success:
            logger.error(f"Search failed: {result.error}")
            return []
        
        repos = []
        for item in result.data.get("items", []):
            try:
                updated_at = datetime.fromisoformat(
                    item["updated_at"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                
                repo = RepositoryHealth(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    stars=item["stargazers_count"],
                    forks=item["forks_count"],
                    open_issues=item["open_issues_count"],
                    language=item.get("language") or "Unknown",
                    topics=item.get("topics", []),
                    last_updated=updated_at,
                )
                repos.append(repo)
                
                with self._lock:
                    self._stats["repos_analyzed"] += 1
                    
            except Exception as e:
                logger.warning(f"Error parsing repo {item.get('name')}: {e}")
                continue
        
        # Ordenar por health score
        repos.sort(key=lambda r: r.health_score, reverse=True)
        
        # Analizar repos para tests/typing
        for repo in repos[:5]:  # Solo los top 5
            await self._analyze_repo_quality(repo)
        
        logger.info(
            f"Search '{query}': {len(repos)} repos "
            f"({sum(1 for r in repos if r.is_gold_standard())} gold standard)"
        )
        
        return repos
    
    async def _analyze_repo_quality(self, repo: RepositoryHealth) -> None:
        """Analiza calidad del repo (tests, typing)."""
        # Buscar archivos de test
        result = await self._api_request(
            f"/repos/{repo.full_name}/contents",
        )
        
        if result.success and isinstance(result.data, list):
            for item in result.data:
                name = item.get("name", "").lower()
                if name in ("tests", "test", "pytest.ini", "conftest.py"):
                    repo.has_tests = True
                if name in ("py.typed", "mypy.ini", ".mypy.ini"):
                    repo.has_typing = True
                if name == "pyproject.toml" or name == "setup.cfg":
                    # Podría contener config de typing
                    repo.has_typing = True
    
    async def get_repository_structure(
        self,
        full_name: str,
        path: str = "",
    ) -> dict:
        """
        Obtiene la estructura de archivos de un repositorio.
        
        Args:
            full_name: Nombre completo del repo (owner/repo)
            path: Ruta dentro del repo
            
        Returns:
            dict: Estructura de archivos
        """
        result = await self._api_request(
            f"/repos/{full_name}/contents/{path}",
        )
        
        if not result.success:
            return {"error": result.error}
        
        structure = {
            "files": [],
            "directories": [],
        }
        
        if isinstance(result.data, list):
            for item in result.data:
                if item["type"] == "file":
                    structure["files"].append({
                        "name": item["name"],
                        "path": item["path"],
                        "size": item["size"],
                        "download_url": item.get("download_url"),
                    })
                elif item["type"] == "dir":
                    structure["directories"].append({
                        "name": item["name"],
                        "path": item["path"],
                    })
        
        return structure
    
    async def download_file_content(
        self,
        full_name: str,
        file_path: str,
    ) -> ScoutResult:
        """
        Descarga el contenido de un archivo.
        
        Args:
            full_name: Nombre completo del repo
            file_path: Ruta del archivo
            
        Returns:
            ScoutResult con el contenido
        """
        # Verificar cache
        cache_key = f"{full_name.replace('/', '_')}_{file_path.replace('/', '_')}"
        cache_file = self._cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            # Verificar TTL (24 horas)
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=24):
                with self._lock:
                    self._stats["cache_hits"] += 1
                return ScoutResult(
                    success=True,
                    data=cache_file.read_text(encoding="utf-8"),
                    cached=True,
                )
        
        result = await self._api_request(
            f"/repos/{full_name}/contents/{file_path}",
        )
        
        if not result.success:
            return result
        
        import base64
        
        content = result.data.get("content", "")
        encoding = result.data.get("encoding", "base64")
        
        if encoding == "base64":
            try:
                decoded = base64.b64decode(content).decode("utf-8")
            except Exception as e:
                return ScoutResult(success=False, error=f"Decode error: {e}")
        else:
            decoded = content
        
        # Guardar en cache
        try:
            cache_file.write_text(decoded, encoding="utf-8")
            with self._lock:
                self._stats["files_downloaded"] += 1
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return ScoutResult(success=True, data=decoded)
    
    async def get_readme(self, full_name: str) -> ScoutResult:
        """Obtiene el README de un repositorio."""
        result = await self._api_request(f"/repos/{full_name}/readme")
        
        if not result.success:
            return result
        
        import base64
        
        content = result.data.get("content", "")
        try:
            decoded = base64.b64decode(content).decode("utf-8")
            return ScoutResult(success=True, data=decoded)
        except Exception as e:
            return ScoutResult(success=False, error=str(e))
    
    async def harvest_gold_standard(
        self,
        project_type: str,
        language: str = "python",
    ) -> list[RepositoryHealth]:
        """
        Busca y retorna repositorios Gold Standard para un tipo de proyecto.
        
        Args:
            project_type: Tipo de proyecto (api, cli, web, etc.)
            language: Lenguaje de programación
            
        Returns:
            list[RepositoryHealth]: Repos Gold Standard ordenados por score
        """
        # Queries específicas por tipo
        TYPE_QUERIES = {
            "api": "api rest framework",
            "cli": "cli command line tool",
            "web": "web application framework",
            "library": "library package",
            "mcp": "model context protocol",
            "agent": "ai agent autonomous",
        }
        
        query = TYPE_QUERIES.get(project_type, project_type)
        
        repos = await self.search_repositories(
            query=query,
            language=language,
            min_stars=GOLD_STANDARD_THRESHOLDS["min_stars"],
            max_results=20,
        )
        
        # Filtrar solo Gold Standards
        gold = [r for r in repos if r.is_gold_standard()]
        
        logger.info(f"Harvested {len(gold)} Gold Standard repos for '{project_type}'")
        
        return gold
    
    def get_status(self) -> dict:
        """Retorna estado del Scout."""
        return {
            "has_token": bool(self._token),
            "rate_limit_remaining": self._rate_limit_remaining,
            "rate_limit_reset": (
                self._rate_limit_reset.isoformat()
                if self._rate_limit_reset else None
            ),
            "cache_dir": str(self._cache_dir),
            "stats": dict(self._stats),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_scout_instance: Optional[ScoutAgent] = None
_scout_lock = Lock()


def get_scout() -> ScoutAgent:
    """Obtiene la instancia singleton del Scout."""
    global _scout_instance
    with _scout_lock:
        if _scout_instance is None:
            _scout_instance = ScoutAgent()
        return _scout_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

async def _test_scout():
    """Test básico del Scout."""
    scout = get_scout()
    
    print("=" * 60)
    print("ULTRAGENT SCOUT v0.1 - Test")
    print("=" * 60)
    print(f"Status: {scout.get_status()}")
    print("=" * 60)
    
    # Buscar repos
    print("\nBuscando repos 'fastapi template'...")
    repos = await scout.search_repositories(
        query="fastapi template",
        language="python",
        min_stars=100,
        max_results=5,
    )
    
    for repo in repos:
        print(f"\n  {repo.name} ⭐{repo.stars}")
        print(f"    Health: {repo.health_score:.1f}")
        print(f"    Gold Standard: {repo.is_gold_standard()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    asyncio.run(_test_scout())
