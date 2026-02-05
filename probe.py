"""
ULTRAGENT PROBE v1.0
====================
Script de diagnóstico de conectividad API.

Realiza "Ping Cognitivo" a cada plataforma:
- Tier Strategic (Gemini) - Hello World
- Tier Coding (SiliconFlow/DeepSeek) - Función Python simple
- Tier Speed (Groq/Llama) - Tokens por segundo
- Tier Scout (GitHub) - Búsqueda de prueba

Uso:
    uv run probe.py
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

# Configuración
PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / ".ai"
HUD_FILE = AI_DIR / "HUD.md"

# Timeouts
REQUEST_TIMEOUT = 30.0


@dataclass
class ProbeResult:
    """Resultado de un probe."""
    tier: str
    provider: str
    status: str  # "✅ OK" | "❌ FAIL" | "⚠️ NO KEY"
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    error: Optional[str] = None
    response_preview: Optional[str] = None


async def probe_groq() -> ProbeResult:
    """Probe Tier Speed (Groq)."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return ProbeResult(
            tier="⚡ Speed",
            provider="Groq",
            status="⚠️ NO KEY",
            error="GROQ_API_KEY not set",
        )
    
    try:
        start = time.perf_counter()
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "user", "content": "Say 'Hello Ultragent' in exactly 3 words."}
                    ],
                    "max_tokens": 20,
                },
            )
        
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            tps = total_tokens / (latency / 1000) if latency > 0 else 0
            content = data["choices"][0]["message"]["content"]
            
            return ProbeResult(
                tier="⚡ Speed",
                provider="Groq",
                status="✅ OK",
                latency_ms=latency,
                tokens_per_second=tps,
                response_preview=content[:50],
            )
        else:
            return ProbeResult(
                tier="⚡ Speed",
                provider="Groq",
                status="❌ FAIL",
                latency_ms=latency,
                error=f"HTTP {response.status_code}",
            )
            
    except Exception as e:
        return ProbeResult(
            tier="⚡ Speed",
            provider="Groq",
            status="❌ FAIL",
            error=str(e),
        )


async def probe_siliconflow() -> ProbeResult:
    """Probe Tier Coding (SiliconFlow/DeepSeek)."""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not api_key:
        return ProbeResult(
            tier="🛠️ Coding",
            provider="SiliconFlow",
            status="⚠️ NO KEY",
            error="SILICONFLOW_API_KEY not set",
        )
    
    try:
        start = time.perf_counter()
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [
                        {"role": "user", "content": "Write a Python function that adds two numbers. Only code, no explanation."}
                    ],
                    "max_tokens": 100,
                },
            )
        
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            return ProbeResult(
                tier="🛠️ Coding",
                provider="SiliconFlow",
                status="✅ OK",
                latency_ms=latency,
                response_preview=content[:80].replace("\n", " "),
            )
        else:
            return ProbeResult(
                tier="🛠️ Coding",
                provider="SiliconFlow",
                status="❌ FAIL",
                latency_ms=latency,
                error=f"HTTP {response.status_code}: {response.text[:100]}",
            )
            
    except Exception as e:
        return ProbeResult(
            tier="🛠️ Coding",
            provider="SiliconFlow",
            status="❌ FAIL",
            error=str(e),
        )


async def probe_github() -> ProbeResult:
    """Probe Tier Scout (GitHub)."""
    token = os.getenv("GITHUB_TOKEN")
    
    if not token or token.startswith("ghp_your"):
        return ProbeResult(
            tier="🕵️ Scout",
            provider="GitHub",
            status="⚠️ NO KEY",
            error="GITHUB_TOKEN not set or placeholder",
        )
    
    try:
        start = time.perf_counter()
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                "https://api.github.com/search/repositories",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                params={
                    "q": "fastapi language:python stars:>1000",
                    "per_page": 1,
                },
            )
        
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            total = data.get("total_count", 0)
            rate_remaining = response.headers.get("x-ratelimit-remaining", "?")
            
            return ProbeResult(
                tier="🕵️ Scout",
                provider="GitHub",
                status="✅ OK",
                latency_ms=latency,
                response_preview=f"Found {total} repos, rate limit: {rate_remaining}",
            )
        else:
            return ProbeResult(
                tier="🕵️ Scout",
                provider="GitHub",
                status="❌ FAIL",
                latency_ms=latency,
                error=f"HTTP {response.status_code}",
            )
            
    except Exception as e:
        return ProbeResult(
            tier="🕵️ Scout",
            provider="GitHub",
            status="❌ FAIL",
            error=str(e),
        )


async def probe_gemini() -> ProbeResult:
    """Probe Tier Strategic (Gemini)."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return ProbeResult(
            tier="💎 Strategic",
            provider="Gemini",
            status="⚠️ NO KEY",
            error="GEMINI_API_KEY not set",
        )
    
    try:
        start = time.perf_counter()
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {"parts": [{"text": "Say 'Hello Ultragent' in exactly 3 words."}]}
                    ],
                },
            )
        
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            
            return ProbeResult(
                tier="💎 Strategic",
                provider="Gemini",
                status="✅ OK",
                latency_ms=latency,
                response_preview=content[:50],
            )
        else:
            return ProbeResult(
                tier="💎 Strategic",
                provider="Gemini",
                status="❌ FAIL",
                latency_ms=latency,
                error=f"HTTP {response.status_code}",
            )
            
    except Exception as e:
        return ProbeResult(
            tier="💎 Strategic",
            provider="Gemini",
            status="❌ FAIL",
            error=str(e),
        )


def update_hud_connectivity(results: list[ProbeResult]) -> None:
    """Actualiza HUD.md con resultados de conectividad."""
    connectivity_section = "\n## 📡 API CONNECTIVITY\n\n"
    connectivity_section += "| Tier | Provider | Status | Latency | Details |\n"
    connectivity_section += "|------|----------|--------|---------|--------|\n"
    
    for r in results:
        latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms else "N/A"
        details = r.response_preview or r.error or ""
        connectivity_section += f"| {r.tier} | {r.provider} | {r.status} | {latency_str} | {details[:40]} |\n"
    
    connectivity_section += f"\n*Last probe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # Leer HUD actual o crear nuevo
    if HUD_FILE.exists():
        content = HUD_FILE.read_text(encoding="utf-8")
        
        # Reemplazar sección existente o añadir
        if "## 📡 API CONNECTIVITY" in content:
            import re
            content = re.sub(
                r"## 📡 API CONNECTIVITY.*?(?=\n## |\n---|\Z)",
                connectivity_section,
                content,
                flags=re.DOTALL
            )
        else:
            # Añadir antes de "---" final o al final
            if content.rstrip().endswith("---"):
                content = content.rstrip()[:-3] + connectivity_section + "\n---\n"
            else:
                content += "\n" + connectivity_section
    else:
        content = f"# 🎛️ ULTRAGENT HUD\n\n{connectivity_section}"
    
    HUD_FILE.parent.mkdir(parents=True, exist_ok=True)
    HUD_FILE.write_text(content, encoding="utf-8")


async def run_all_probes() -> list[ProbeResult]:
    """Ejecuta todos los probes en paralelo."""
    print("=" * 60)
    print("ULTRAGENT PROBE v1.0 - Diagnóstico de Conectividad API")
    print("=" * 60)
    print()
    
    # Ejecutar probes en paralelo
    results = await asyncio.gather(
        probe_groq(),
        probe_siliconflow(),
        probe_github(),
        probe_gemini(),
    )
    
    # Mostrar resultados
    print("📡 RESULTADOS DE CONECTIVIDAD:")
    print("-" * 60)
    
    for r in results:
        latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms else "N/A"
        tps_str = f" ({r.tokens_per_second:.0f} tok/s)" if r.tokens_per_second else ""
        print(f"  {r.tier:15} | {r.provider:12} | {r.status} | {latency_str}{tps_str}")
        if r.error:
            print(f"                   └─ Error: {r.error}")
        if r.response_preview:
            print(f"                   └─ Response: {r.response_preview[:50]}")
    
    print("-" * 60)
    
    # Contar estados
    ok_count = sum(1 for r in results if "OK" in r.status)
    fail_count = sum(1 for r in results if "FAIL" in r.status)
    nokey_count = sum(1 for r in results if "NO KEY" in r.status)
    
    print(f"\n📊 RESUMEN: {ok_count} OK | {fail_count} FAIL | {nokey_count} NO KEY")
    
    # Actualizar HUD
    update_hud_connectivity(results)
    print(f"\n✅ HUD actualizado: {HUD_FILE}")
    
    # Retornar keys faltantes
    missing = [r for r in results if "NO KEY" in r.status]
    if missing:
        print("\n⚠️  KEYS FALTANTES:")
        for r in missing:
            key_name = {
                "Groq": "GROQ_API_KEY",
                "SiliconFlow": "SILICONFLOW_API_KEY",
                "GitHub": "GITHUB_TOKEN",
                "Gemini": "GEMINI_API_KEY",
            }.get(r.provider, "UNKNOWN")
            print(f"   - {key_name}")
    
    print()
    return results


if __name__ == "__main__":
    asyncio.run(run_all_probes())
