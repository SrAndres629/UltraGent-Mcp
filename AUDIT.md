# 🔍 ULTRAGENT v2.0 - AI AUDIT SPECIFICATION

> **Purpose:** Este documento está diseñado para que un agente IA (Gemini, Claude, GPT) pueda verificar si el proyecto cumple con los objetivos arquitectónicos establecidos.

---

## 📋 CHECKLIST DE VERIFICACIÓN

### ORDEN 1: Infrastructure Setup
- [ ] `mcp_server.py` existe y exporta servidor FastMCP
- [ ] Tool `sync_status` retorna estado del sistema
- [ ] Tool `get_memory` lee `.ai/memory.md`
- [ ] Tool `get_hud` lee `.ai/HUD.md`

### ORDEN 2: Sentinel Reactive Core
- [ ] `sentinel.py` implementa observador de filesystem
- [ ] Usa `watchdog` para monitoreo
- [ ] Debounce de 500ms implementado
- [ ] Eventos guardados en `signals.json`
- [ ] Tool `get_sentinel_status` disponible

### ORDEN 3: Omni-Router Economy
- [ ] `router.py` implementa 4 tiers:
  - [ ] Tier SPEED (Groq)
  - [ ] Tier CODING (SiliconFlow/DeepSeek)
  - [ ] Tier VISUAL (Kimi)
  - [ ] Tier STRATEGIC (Gemini)
- [ ] Circuit Breaker con failover automático
- [ ] Token budget tracking
- [ ] Tools: `route_task`, `ask_swarm`, `get_router_status`, `get_token_usage`

### ORDEN 4: Librarian Knowledge Layer
- [ ] `librarian.py` implementa indexación de código
- [ ] Usa Tree-sitter para parsing
- [ ] ChromaDB para embeddings vectoriales
- [ ] Búsqueda semántica implementada
- [ ] Tools: `search_code`, `get_file_skeleton`, `index_file`, `get_librarian_status`

### ORDEN 5: Scout Evolution Audit
- [ ] `scout.py` implementa GitHub API harvester
- [ ] Análisis de "Gold Standard" repositories
- [ ] Health scoring de repositorios
- [ ] `evolution.py` implementa Fitness Scorecard
- [ ] Métricas: Legibilidad, Escalabilidad, Error Handling, Acoplamiento
- [ ] Tools: `search_github_repos`, `benchmark_with_github`, `get_scout_status`, `get_evolution_status`

### ORDEN 6: Mechanic Vision Runtime
- [ ] `mechanic.py` implementa Docker sandbox
- [ ] Límites: 512MB RAM, 50% CPU, 30s timeout
- [ ] Network disabled, auto_remove containers
- [ ] `vision.py` implementa grafos de dependencia
- [ ] NetworkX + matplotlib para visualización
- [ ] Detección de ciclos (marcados en ROJO)
- [ ] Tools: `test_code_securely`, `visualize_architecture`, `get_mechanic_status`, `get_vision_status`

### ORDEN 7: HUD Command Bridge
- [ ] `hud_manager.py` implementa dashboard manager
- [ ] Throttling de 1 segundo
- [ ] Human-in-the-loop signals
- [ ] `main.py` orquesta todos los threads
- [ ] Graceful shutdown implementado
- [ ] Tools: `get_full_status`, `set_mission_goal`, `export_session`

---

## 📊 MÉTRICAS ESPERADAS

| Métrica | Valor Esperado |
|---------|----------------|
| Total Tools MCP | 24 |
| Módulos Python | 10 |
| Líneas de código (aprox) | 5000+ |
| Cobertura de features | 100% |

---

## 🧪 COMANDOS DE VERIFICACIÓN

```bash
# Verificar que el servidor inicia sin errores
uv run python -c "from mcp_server import mcp; print(f'Tools: {len(mcp._tool_manager._tools)}')"

# Verificar conteo de tools (debe ser 24)
uv run python -c "from mcp_server import mcp; assert len(mcp._tool_manager._tools) == 24"

# Verificar módulos importables
uv run python -c "import sentinel, router, librarian, scout, evolution, mechanic, vision, hud_manager; print('All modules OK')"

# Ejecutar probe de conectividad
uv run probe.py
```

---

## 📁 ESTRUCTURA REQUERIDA

```
Ultragent/
├── main.py              ✓ Entry point
├── mcp_server.py        ✓ MCP Server (24 tools)
├── sentinel.py          ✓ Filesystem watcher
├── router.py            ✓ 4-tier LLM economy
├── librarian.py         ✓ Code indexer
├── scout.py             ✓ GitHub harvester
├── evolution.py         ✓ Fitness scoring
├── mechanic.py          ✓ Docker sandbox
├── vision.py            ✓ Dependency graphs
├── hud_manager.py       ✓ Dashboard
├── probe.py             ✓ API diagnostics
├── pyproject.toml       ✓ Dependencies
├── .env.example         ✓ API keys template
├── README.md            ✓ Documentation
└── .ai/
    ├── memory.md        ✓ Architectural decisions
    ├── HUD.md           ✓ Control panel
    └── logs/            ✓ Module logs
```

---

## 🎯 OBJETIVOS DEL SISTEMA

1. **Economía de Tokens:** Distribuir tareas al tier más económico que pueda completarlas
2. **Observabilidad:** HUD.md como única fuente de verdad del estado del sistema
3. **Seguridad:** Ejecución de código en sandbox Docker aislado
4. **Calidad:** Auditoría automática contra Gold Standards de GitHub
5. **Arquitectura Visual:** Grafos de dependencia con detección de ciclos
6. **Human-in-the-Loop:** Señales claras cuando se requiere decisión humana

---

## 📝 NOTAS PARA EL AUDITOR IA

1. Todo el código está en Python 3.12+
2. Dependencias manejadas con `uv` (pyproject.toml)
3. Patrón Singleton usado en todos los módulos
4. Logging profesional con filtro de secretos
5. Graceful degradation si un módulo falla
6. `.ai/memory.md` contiene historial de decisiones arquitectónicas

---

*Documento generado para auditoría por agentes IA - ULTRAGENT v2.0*
