# 🏗️ ULTRAGENT v2.0 - ARQUITECTURA TÉCNICA

> **Documento para revisión por agentes IA**

---

## 📐 DIAGRAMA DE MÓDULOS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ULTRAGENT v2.0                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  main.py    │───▶│ mcp_server  │───▶│  24 Tools   │                 │
│  │  (entry)    │    │   (core)    │    │   (API)     │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│         │                  │                                            │
│         ▼                  ▼                                            │
│  ┌─────────────┐    ┌─────────────────────────────────────────────┐    │
│  │  sentinel   │    │              MÓDULOS LÓBULO                  │    │
│  │ (watchdog)  │    ├─────────────┬─────────────┬─────────────────┤    │
│  └─────────────┘    │   router    │  librarian  │  scout/evolution │    │
│         │           │ (4-tier LLM)│ (code index)│ (GitHub audit)   │    │
│         ▼           └─────────────┴─────────────┴─────────────────┘    │
│  ┌─────────────┐    ┌─────────────┬─────────────┐                      │
│  │ hud_manager │    │  mechanic   │   vision    │                      │
│  │ (dashboard) │    │  (sandbox)  │  (graphs)   │                      │
│  └─────────────┘    └─────────────┴─────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FLUJO DE DATOS

```
[Usuario/AI] 
     │
     ▼
[MCP Server] ─────────────────────────────────────────┐
     │                                                │
     ├──▶ route_task() ──▶ [Omni-Router]              │
     │         │                                      │
     │         ├──▶ Tier SPEED (Groq) ──▶ 300+ tok/s  │
     │         ├──▶ Tier CODING (SiliconFlow)         │
     │         ├──▶ Tier VISUAL (Kimi)                │
     │         └──▶ Tier STRATEGIC (Gemini)           │
     │                                                │
     ├──▶ search_code() ──▶ [Librarian] ──▶ ChromaDB  │
     │                                                │
     ├──▶ benchmark_with_github() ──▶ [Scout+Evolution]
     │                                                │
     ├──▶ test_code_securely() ──▶ [Mechanic] ──▶ Docker
     │                                                │
     └──▶ visualize_architecture() ──▶ [Vision] ──▶ PNG
```

---

## 📦 DEPENDENCIAS (pyproject.toml)

| Paquete | Propósito |
|---------|-----------|
| `fastmcp` | Servidor MCP |
| `httpx` | Cliente HTTP async |
| `watchdog` | Monitoreo filesystem |
| `tree-sitter` | Parsing de código |
| `chromadb` | Base vectorial |
| `docker` | SDK Docker |
| `networkx` | Grafos de dependencia |
| `matplotlib` | Visualización |

---

## 🔐 SEGURIDAD IMPLEMENTADA

| Medida | Implementación |
|--------|----------------|
| API Keys | `.env` en .gitignore, nunca en repo |
| Sandbox | Docker avec network_disabled |
| Resources | 512MB RAM, 50% CPU, 30s timeout |
| Volumes | Read-only mounts |
| Env Vars | No se pasan al container |
| Logging | Filtro de secrets |

---

## 📊 DECISIONES ARQUITECTÓNICAS (ADRs)

Ver `.ai/memory.md` para el historial completo de 11 decisiones:

1. **ADR-001:** FastMCP como framework MCP
2. **ADR-002:** Estructura `.ai/` para estado persistente
3. **ADR-003:** Watchdog para Sentinel
4. **ADR-004:** 4-Tier Economy con Circuit Breaker
5. **ADR-005:** Tree-sitter + ChromaDB para Librarian
6. **ADR-006:** Swarm paralelo para tareas batch
7. **ADR-007:** GitHub Health Scoring
8. **ADR-008:** Fitness Scorecard con 4 métricas
9. **ADR-009:** Cache con patrón @lru_cache
10. **ADR-010:** Docker sandbox con límites estrictos
11. **ADR-011:** HUD Manager con throttling 1s

---

## 🧪 TESTING MANUAL

```bash
# 1. Verificar importación de todos los módulos
uv run python -c "
import mcp_server
import sentinel
import router
import librarian
import scout
import evolution
import mechanic
import vision
import hud_manager
print('✅ All modules imported successfully')
"

# 2. Verificar conteo de tools
uv run python -c "
from mcp_server import mcp
tools = len(mcp._tool_manager._tools)
print(f'Tools: {tools}')
assert tools == 24, f'Expected 24, got {tools}'
print('✅ Tool count verified')
"

# 3. Generar HUD
uv run python -c "
from hud_manager import get_hud_manager
hud = get_hud_manager()
hud.refresh_dashboard(force=True)
print('✅ HUD generated')
"

# 4. Probe de conectividad
uv run probe.py
```

---

*Arquitectura documentada para auditoría IA - ULTRAGENT v2.0*
