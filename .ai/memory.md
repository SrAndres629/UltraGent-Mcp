# 🧠 ULTRAGENT - MEMORIA EPISÓDICA
> *"El código no es texto; es infraestructura."*

---

## 📅 2026-02-04 | GÉNESIS - Protocolos de Asimilación

### Decisión Arquitectónica #001: Identidad Agnóstica
**Contexto:** Primera sesión de configuración de Ultragent.
**Decisión:** La identidad del sistema reside en `.ai/`, no en el modelo subyacente.
**Razón:** Permite continuidad entre sesiones y migración entre modelos (Gemini/Claude/GPT).

### Decisión Arquitectónica #002: Arquitectura de 7 Lóbulos
**Contexto:** Necesidad de evitar degradación cognitiva en tareas complejas.
**Decisión:** Separación de Preocupaciones (SoC) en módulos especializados.

| Lóbulo | Función | Herramienta Principal |
|--------|---------|----------------------|
| 🛡️ PRISMA | Gestión de Contexto | Fragmentación inteligente |
| 🧠 CORTEX | Persistencia | SQLite + HUD |
| 📚 LIBRARIAN | Conocimiento/RAG | Tree-sitter |
| 🕵️ SCOUT | Investigación | GitHub API / Web Search |
| 🧬 EVOLUTION | Auditoría Genética | Comparación vs Gold Standard |
| 👁️ VISION | Arquitectura Visual | NetworkX / Vision Models |
| 🔧 MECHANIC | Ejecución | Docker / uv |

**Razón:** Cada módulo tiene responsabilidad única, evitando saturación de contexto.

### Decisión Arquitectónica #003: Omni-Router (Orquestación Multi-Modelo)
**Contexto:** Objetivo de maximizar eficiencia η = Complejidad / (Costo × Latencia)
**Decisión:** Implementar enrutador de 4 tiers especializados:

| Tier | Proveedor | Uso | Justificación |
|------|-----------|-----|---------------|
| 🔴 VISUAL-AGÉNTICO | Kimi K2.5 | Vision-to-Code, diagramas | Gold standard multimodal |
| 🛠️ CODING | DeepSeek V3 / Qwen 2.5 | Backend, algoritmos | Balance precisión/costo |
| ⚡ SPEED | Groq / SambaNova / Cerebras | Tests, boilerplate | >300 t/s velocidad |
| 💎 STRATEGIC | Gemini Pro / Claude 3.5 | Arquitectura senior | Cerebro central |

**Razón:** Preservar tokens "Senior" para supervisión. Delegar tareas mecánicas al enjambre gratuito.
**Resiliencia:** Circuit Breaker con failover automático (<500ms) ante errores 429/5xx.

---

## 🔗 Hardware Asignado
- **CPU (i9):** Orquestación general, Docker
- **GPU (RTX 3060):** Embeddings, VISION

### Decisión Arquitectónica #004: Sentinel (Sistema Nervioso Reactivo)
**Contexto:** Necesidad de eliminar pasividad y responder a cambios del filesystem.
**Decisión:** Implementar Event-Driven Architecture con `watchdog`:

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO SENTINEL                           │
├─────────────────────────────────────────────────────────────┤
│  [Filesystem] ──on_modified──▶ [SENTINEL]                   │
│                                    │                        │
│                              Debounce (Δt=2s)               │
│                                    │                        │
│                          ┌─────────▼─────────┐              │
│                          │ LIBRARIAN         │              │
│                          │ (Tree-sitter)     │              │
│                          └─────────┬─────────┘              │
│                       Sintaxis OK? │                        │
│                    ┌───────────────┼───────────────┐        │
│                    ▼               ▼               ▼        │
│               [ERROR]          [REVIEW_READY]  [HUD.md]     │
│                  │                  │                       │
│                  ▼                  ▼                       │
│             MECHANIC           EVOLUTION                    │
│            (auto-fix)         (Auditoría)                   │
└─────────────────────────────────────────────────────────────┘
```

**Componentes:**
- **Watchdog:** Monitoreo 24/7 de workspace
- **Debounce:** `t_ready = t_last_change + 2s` (evita archivos incompletos)
- **Tree-sitter:** Validación sintáctica instantánea
- **HUD Feedback:** Reporte automático de cambios al panel

**Razón:** Convertir al sistema en reactivo. El Arquitecto solo es interrumpido para auditoría, no para correcciones triviales.

---

## 📅 2026-02-04 | ORDEN 1 - Infrastructure Setup

### Decisión Arquitectónica #005: Servidor MCP Base Desplegado
**Contexto:** Necesidad de materializar el sistema conceptual en código ejecutable.
**Decisión:** Implementar servidor FastMCP con 3 tools iniciales (sync_status, get_memory, get_hud).

**Stack implementado:**
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Package Manager | uv (Astral) | Resolución determinista, 10-100x más rápido |
| MCP Framework | fastmcp | Decoradores Pythonic, SSE nativo |
| Secrets | python-dotenv | Zero-config, estándar de industria |
| File Watcher | watchdog | Preparación para Sentinel |
| Database | sqlite3 + WAL | Concurrencia mejorada |

**Mitigaciones implementadas:**
1. SecretFilter en logging (redacta API keys)
2. WAL mode + busy_timeout para SQLite
3. Estructura flat con hatch.build para compatibilidad

---

## 📅 2026-02-04 | ORDEN 2 - Sentinel Reactive Core

### Decisión Arquitectónica #006: Sentinel Implementado
**Contexto:** Eliminar pasividad operativa con EDA.
**Decisión:** Watchdog + debounce 2s + signals.json.

Componentes: EventHandler, Debounce Timer, Exclusion Filters, HUD Updater, signals.json.
Mitigaciones: Race conditions (debounce), Event flooding (rate limiting), Recursive monitoring (filtros).

---

## 📅 2026-02-04 | ORDEN 3 - Omni-Router Economy

### Decisión Arquitectónica #007: Omni-Router Implementado
**Contexto:** Arbitraje de APIs con failover y gestión de tokens.
**Decisión:** 4 Tiers (Visual/Coding/Speed/Strategic) + Circuit Breaker + BudgetGuard.

Proveedores: Groq, SiliconFlow, NVIDIA NIM, Gemini. Failover <500ms, Exponential Backoff.
Features: Swarms para procesamiento paralelo, clasificación automática de tareas.

---

## 📅 2026-02-04 | ORDEN 4 - Librarian Knowledge Layer

### Decisión Arquitectónica #008: Librarian Implementado
**Contexto:** Memoria profunda con análisis AST y búsqueda semántica.
**Decisión:** Tree-sitter (Skeletonization) + ChromaDB + sentence-transformers.

Lenguajes: Python, JavaScript, TypeScript. Embeddings: all-MiniLM-L6-v2.
Features: Carga dinámica de gramáticas, hierarchical embedding, cross-referencing.

---

## 📅 2026-02-04 | ORDEN 5 - Scout Evolution Audit

### Decisión Arquitectónica #009: Scout/Evolution Implementados
**Contexto:** Investigación externa y crítica arquitectónica despiadada.
**Decisión:** GitHub API harvesting + Fitness Scorecard + prompts de crítica severa.

Features: Gold Standard detection (500+ stars, tests, typing), loop protection (max 3, 15% mejora).
Métricas: Legibilidad/Escalabilidad/ErrorHandling/Acoplamiento (25% c/u).

---

## 📅 2026-02-04 | ORDEN 6 - Mechanic Vision Runtime

### Decisión Arquitectónica #010: Mechanic/Vision Implementados
**Contexto:** Ejecución segura y visualización arquitectónica.
**Decisión:** Docker SDK + NetworkX + matplotlib para sandbox y grafos.

Features: auto_remove containers, 512MB mem, 50% CPU, timeout 30s, cycles en ROJO.
Métricas: Network disabled, read_only volumes, sin env vars del host.

---

## 📅 2026-02-04 | ORDEN 7 - HUD Command Bridge

### Decisión Arquitectónica #011: HUD Manager + main.py
**Contexto:** Observabilidad unificada y orquestación de arranque.
**Decisión:** Throttled HUD refresh (1s), threads para Sentinel/HUD, graceful shutdown.

Features: Human-in-the-loop signals, export_session ZIP, 24 tools totales.

---

## 🎉 ULTRAGENT v2.0 - SISTEMA COMPLETAMENTE OPERATIVO

**Fecha de finalización:** 2026-02-04T20:15:00-04:00

**Módulos implementados:** 8
**Tools MCP totales:** 24
**Líneas de código:** ~5000+

---
