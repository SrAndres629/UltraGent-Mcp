# ULTRAGENT v2.0

🤖 **Hybrid Autonomous Engineering System**

Sistema de Ingeniería Híbrida Autónoma basado en MCP (Model Context Protocol).

## ✨ Features

- **24 MCP Tools** organizadas en 8 módulos
- **4-Tier LLM Economy** (Strategic, Coding, Speed, Scout)
- **Omni-Router** con Circuit Breaker y failover automático
- **Sentinel** para monitoreo reactivo del filesystem
- **Librarian** con Tree-sitter y ChromaDB para indexación de código
- **Scout/Evolution** para benchmarking contra Gold Standards de GitHub
- **Mechanic** para ejecución segura en Docker sandbox
- **Vision** para visualización de arquitectura con NetworkX
- **HUD Dashboard** para observabilidad unificada

## 🚀 Quick Start

```bash
# Clonar repositorio
git clone https://github.com/SrAndres629/UltraGent-Mcp.git
cd UltraGent-Mcp

# Instalar dependencias con uv
uv sync

# Configurar API keys
cp .env.example .env
# Editar .env con tus keys

# Ejecutar servidor MCP
uv run main.py

# O solo el servidor MCP
uv run mcp_server.py
```

## 📂 Structure

```
Ultragent/
├── main.py              # Entry point orquestado
├── mcp_server.py        # MCP Server v2.0 (24 tools)
├── sentinel.py          # Filesystem watcher
├── router.py            # 4-tier LLM economy
├── librarian.py         # Code indexer (Tree-sitter + ChromaDB)
├── scout.py             # GitHub API harvester
├── evolution.py         # Fitness Scorecard
├── mechanic.py          # Docker sandbox
├── vision.py            # Dependency graph (NetworkX)
├── hud_manager.py       # Dashboard manager
├── probe.py             # API connectivity diagnostic
└── .ai/
    ├── HUD.md           # Control panel
    ├── memory.md        # Architectural decisions
    └── logs/            # Module logs
```

## 🛠️ MCP Tools

| Module | Tools |
|--------|-------|
| Core | `sync_status`, `get_memory`, `get_hud` |
| Sentinel | `get_sentinel_status`, `clear_sentinel_signals` |
| Router | `route_task`, `ask_swarm`, `get_router_status`, `get_token_usage` |
| Librarian | `search_code`, `get_file_skeleton`, `index_file`, `get_librarian_status` |
| Scout | `search_github_repos`, `get_scout_status` |
| Evolution | `benchmark_with_github`, `get_evolution_status` |
| Mechanic | `test_code_securely`, `get_mechanic_status` |
| Vision | `visualize_architecture`, `get_vision_status` |
| HUD | `get_full_status`, `set_mission_goal`, `export_session` |

## 🔑 API Keys Required

| Key | Purpose | Get it at |
|-----|---------|-----------|
| `GROQ_API_KEY` | Speed tier | [console.groq.com](https://console.groq.com) |
| `SILICONFLOW_API_KEY` | Coding tier | [siliconflow.cn](https://siliconflow.cn) |
| `GITHUB_TOKEN` | Scout tier | GitHub Settings |
| `GEMINI_API_KEY` | Strategic tier | [aistudio.google.com](https://aistudio.google.com/apikey) |

## 📝 License

MIT License
