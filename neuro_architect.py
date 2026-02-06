"""
ULTRAGENT NEURO-ARCHITECT v0.1 (Hyper-V Core)
=============================================
Sistema Nervioso Central para agentes IA.

Responsabilidades:
1. Mantener un NeuroGraph vivo (Grafo de dependencias + Estado)
2. Inyectar telemetría en tiempo real (Logs, Variables)
3. Proveer herramientas de razonamiento espacial (Impact Analysis, Flow Tracing)
4. Exportar estado para visualización 3D (WebGL)
"""

import ast
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from vision import get_vision, DependencyNode, DependencyEdge

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
NEURO_DIR = AI_DIR / "neuro"
NEURO_DIR.mkdir(parents=True, exist_ok=True)

NEURO_STATE_FILE = NEURO_DIR / "neuro_state.json"

logger = logging.getLogger("ultragent.neuro")

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS DE DATOS NEURONALES
# ═══════════════════════════════════════════════════════════════════════════════

class SynapseType(str, Enum):
    IMPORT = "import"           # Dependencia estática
    CALL = "call"               # Llamada de función
    DATA_FLOW = "data_flow"     # Flujo de datos (return/yield)
    INHERITANCE = "inheritance" # Herencia de clase

@dataclass
class NeuronState:
    """Estado dinámico de un nodo (neurona/archivo)."""
    last_active: Optional[datetime] = None
    activation_level: float = 0.0  # 0.0 a 1.0 (brillo en 3D)
    error_rate: float = 0.0
    active_variables: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list) # Atomic Viz
    fix_attempts: int = 0  # To avoid infinite loops in auto-healing

    def to_dict(self) -> dict:
        return {
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "activation_level": self.activation_level,
            "error_rate": self.error_rate,
            "active_variables": self.active_variables,
            "logs": self.logs[-5:],  # Keep last 5 logs
            "fix_attempts": self.fix_attempts
        }

@dataclass
class ImpactPrediction:
    """Predicción de impacto de un cambio."""
    target_node: str
    direct_impact: List[str]
    ripple_effect: List[str]  # Efectos secundarios (2do grado)
    risk_score: float         # 0.0 a 100.0
    breaking_paths: List[str] # Rutas críticas afectadas

    @property
    def affected_nodes(self) -> List[str]:
        """Unifica impactos directos e indirectos."""
        return list(set(self.direct_impact + self.ripple_effect))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["affected_nodes"] = self.affected_nodes
        return data

# ═══════════════════════════════════════════════════════════════════════════════
# NEURO ARCHITECT
# ═══════════════════════════════════════════════════════════════════════════════

class NeuroArchitect:
    """
    Arquitecto del Sistema Nervioso.
    Mantiene el grafo vivo y responde consultas de otros agentes.
    """

    def __init__(self, project_root: Optional[str] = None):
        self._lock = Lock()
        self._graph: nx.DiGraph = nx.DiGraph()
        self._states: Dict[str, NeuronState] = {}
        self._project_root = project_root
        self._vision = get_vision(project_root)
        
        # Inicialización
        self._initialize_cortex()
        logger.info(f"NeuroArchitect (Hyper-V) online. Project: {project_root or 'default'}")

    def _initialize_cortex(self):
        """Inicializa el grafo base usando Vision y enriquece con datos."""
        try:
            logger.info("Building initial neural web...")
            nodes, edges = self._vision.scan_project()
            
            with self._lock:
                self._graph = self._vision.build_graph(nodes, edges, hide_stdlib=False)
                
                # Inicializar estados
                for node in self._graph.nodes():
                    self._states[node] = NeuronState()
                
                # Enriquecer con Data Flow (Tipos de datos) - Deshabilitado por defecto para velocidad
                # self._enrich_graph_with_data_flow()
                    
            logger.info(f"Neural web built: {len(self._graph.nodes())} neurons connected.")
        except Exception as e:
            logger.error(f"Failed to initialize cortex: {e}")

    def _enrich_graph_with_data_flow(self):
        """Usa Librarian para descubrir qué datos viajan por las sinapsis."""
        try:
            from librarian import get_librarian
            librarian = get_librarian()
            
            # Mapeo rápido de nodo -> file_path
            # Asumimos que vision.py etiqueta los nodos con paths si es posible
            # o intentamos inferir.
            
            updates = []
            
            for u, v, data in self._graph.edges(data=True):
                # Si 'u' llama a 'v', queremos saber qué retorna 'v' (input para u)
                # y qué argumentos toma 'v' (output de u)
                
                # Intentamos obtener esqueleto del destino 'v'
                # Supongamos que v es "modulo.Clase.metodo" o "archivo.py"
                
                # Búsqueda heurística en Librarian
                results = librarian.semantic_search(
                    query=f"function signature for {v}", 
                    n_results=1
                )
                
                if results and results[0]['relevance'] > 0.8:
                    skeleton = results[0]
                    signature = skeleton.get('signature', '')
                    
                    # Parsear firma simple (muy básico)
                    # def foo(a: int) -> str
                    ret_type = "Any"
                    if "->" in signature:
                        ret_type = signature.split("->")[1].strip().split(":")[0]
                    
                    args = "()"
                    if "(" in signature and ")" in signature:
                        args = signature[signature.find("("):signature.find(")")+1]

                    updates.append((u, v, {"payload": f"{args} -> {ret_type}"}))

            # Aplicar actualizaciones
            for u, v, attrs in updates:
                self._graph[u][v].update(attrs)
                
        except Exception as e:
            logger.warning(f"Data flow enrichment partial failure: {e}")

    def get_compressed_brain_state(self) -> str:
        """
        Retorna el estado del cerebro en formato minificado y optimizado para tokens.
        Diseñado para ser consumido por LLMs (Gemini/Claude).
        
        Format (JSON-like compact):
        N:{id:t,al,er} (Nodes: type, activation, error)
        E:{s,t,tp} (Edges: source, target, type)
        """
        nodes_min = []
        for n, attrs in self._graph.nodes(data=True):
            state = self._states.get(n, NeuronState())
            # Condensar tipo: file->f, function->fn, class->c
            ntype = attrs.get("node_type", "u")[0] 
            
            # Solo incluir nodos activos o con error para ahorrar más?
            # Por ahora todo, pero compacto.
            n_data = f"{n}:{ntype}"
            if state.activation_level > 0: n_data += f":{state.activation_level:.1f}"
            if state.error_rate > 0: n_data += f":E{state.error_rate:.1f}"
            nodes_min.append(n_data)

        edges_min = []
        for u, v, attrs in self._graph.edges(data=True):
            etype = attrs.get("type", "u")[0] # c=call, i=import
            edges_min.append(f"{u}>{v}:{etype}")

        return json.dumps({
            "T": datetime.now().isoformat(), # Timestamp
            "N": nodes_min, # Nodes
            "E": edges_min, # Edges
        }, separators=(',', ':'))

    def ingest_telemetry(self, node_name: str, event_type: str, payload: Dict[str, Any]):
        """
        Inyecta telemetría en tiempo real al sistema nervioso.
        
        Args:
            node_name: Nombre del nodo (archivo/módulo)
            event_type: 'execution', 'error', 'variable_update'
            payload: Datos del evento
        """
        with self._lock:
            if node_name not in self._states:
                # Auto-register new nodes dynamically
                if node_name not in self._graph:
                    self._graph.add_node(node_name, node_type="dynamic")
                self._states[node_name] = NeuronState()

            state = self._states[node_name]
            state.last_active = datetime.now()
            
            if event_type == "execution":
                state.activation_level = min(1.0, state.activation_level + 0.2)
                # Decay logic would go somewhere else or on tick
                
            elif event_type == "error":
                state.error_rate = min(1.0, state.error_rate + 0.1)
                log_msg = f"ERROR: {payload.get('message', 'Unknown error')}"
                state.logs.append(log_msg)
                
            elif event_type == "variable_update":
                state.active_variables.update(payload)

            elif event_type == "debt_scan":
                # Actualizar estado de salud basado en escaneo de deuda
                score = payload.get("score", 100)
                issue_count = payload.get("issue_count", 0)
                
                # Mapear score (0-100) a error_rate (0.0-1.0) inverso
                # Score 100 -> Error 0.0
                # Score 0   -> Error 1.0
                new_error_rate = 1.0 - (score / 100.0)
                state.error_rate = new_error_rate
                
                if issue_count > 0:
                    state.logs.append(f"DEBT SCAN: {issue_count} issues found (Score: {score})")
                    # Añadir detalles al log
                    for issue in payload.get("issues", [])[:3]: # Top 3
                        state.logs.append(f"[{issue['type']}] L{issue['line']}")
            
            # Persistir estado periódicamente? 
            # Por ahora lo dejamos en memoria para velocidad.

    def get_next_focus(self) -> Dict[str, Any]:
        """
        Determina cuál es el siguiente nodo más crítico para atacar.
        Basado en: Error Rate * Risk Score / (Attempts + 1)
        """
        candidates = []
        
        # Pre-calcular centralidad si no está cacheada (o usar in-degree simple)
        try:
            centrality = nx.degree_centrality(self._graph)
        except:
            centrality = {}

        for n in self._graph.nodes():
            state = self._states.get(n, NeuronState())
            
            # Solo considerar nodos con algún problema
            if state.error_rate > 0.1:
                # Metrics
                risk = centrality.get(n, 0.0) * 100 # 0-100 approx
                health = state.error_rate # 0.0-1.0
                
                # Formula de Prioridad:
                # (Daño * Impacto) / (Intentos + 1) -> Penaliza reintentos fallidos
                priority = (health * (1 + risk)) / (state.fix_attempts + 1)
                
                candidates.append({
                    "node": n,
                    "priority": priority,
                    "reason": f"Error: {health:.2f}, Risk: {risk:.2f}, Attempts: {state.fix_attempts}"
                })
        
        # Ordenar por prioridad descendente
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        
        if not candidates:
            return {"target": None, "message": "System is healthy."}
            
        target = candidates[0]
        return {
            "target": target["node"],
            "priority": target["priority"],
            "reason": target["reason"],
            "candidates_count": len(candidates)
        }

    def analyze_impact(self, target_node: str) -> ImpactPrediction:
        """
        Predice qué parte del sistema se verá afectada si se modifica 'target_node'.
        Herramienta clave para agentes antes de escribir código.
        """
        if target_node not in self._graph:
            return ImpactPrediction(target_node, [], [], 0.0, [])

        # 1. Impacto Directo (Incoming edges: quién depende de mí)
        direct_dependents = list(self._graph.predecessors(target_node))
        
        # 2. Ripple Effect (BFS/DFS downstream inversa - quién depende de quién depende de mí)
        ripple_dependents = set()
        for bio in direct_dependents:
            try:
                # Ancestors en grafo de dependencias = dependents reales
                ancestors = nx.ancestors(self._graph, bio)
                ripple_dependents.update(ancestors)
            except Exception:
                pass
        
        # Filtrar directos del ripple
        ripple_list = [n for n in ripple_dependents if n not in direct_dependents and n != target_node]
        
        # 3. Risk Score Calculation
        # Basado en criticalidad (PageRank o Degree Centrality)
        try:
            centrality = nx.degree_centrality(self._graph)
            node_score = centrality.get(target_node, 0.0) * 1000 # Escalar
        except:
            node_score = len(direct_dependents) * 10

        risk_score = min(100.0, node_score + (len(ripple_list) * 2))

        return ImpactPrediction(
            target_node=target_node,
            direct_impact=direct_dependents,
            ripple_effect=ripple_list[:20], # Top 20
            risk_score=risk_score,
            breaking_paths=[] # TODO: Path finding logic
        )

    def trace_flow(self, start_node: str, end_node: str) -> Dict[str, Any]:
        """
        Traza el camino de conexión entre dos nodos.
        Useful for understanding how data travels.
        """
        try:
            path = nx.shortest_path(self._graph, source=start_node, target=end_node)
            return {
                "exists": True,
                "path": path,
                "length": len(path),
                "steps": [
                    {
                        "source": u,
                        "target": v,
                        "type": self._graph[u][v].get("edge_type", "unknown")
                    }
                    for u, v in zip(path[:-1], path[1:])
                ]
            }
        except nx.NetworkXNoPath:
            return {"exists": False, "error": "No path found"}
        except nx.NodeNotFound as e:
            return {"exists": False, "error": f"Node not found: {e}"}

    def get_brain_state(self) -> Dict[str, Any]:
        """
        Retorna el estado completo del cerebro para renderizado o análisis.
        """
        nodes_data = []
        for n, attrs in self._graph.nodes(data=True):
            state = self._states.get(n, NeuronState())
            
            # Atomic Viz: Recuperar memorias vinculadas al nodo desde el Cortex
            try:
                from cortex import get_cortex
                related = get_cortex().get_related_memories(n)
                state.memories = [
                    {"content": m.content, "tags": m.tags, "importance": m.importance} for m in related
                ]
            except Exception as e:
                logger.warning(f"Error vinculando memorias a {n}: {e}")

            nodes_data.append({
                "id": n,
                "type": attrs.get("node_type", "unknown"),
                "state": state.to_dict(),
                "metrics": {
                    "in_degree": self._graph.in_degree(n) if n in self._graph else 0,
                    "out_degree": self._graph.out_degree(n) if n in self._graph else 0,
                }
            })

        edges_data = []
        for u, v, attrs in self._graph.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "type": attrs.get("edge_type", "unknown")
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "nodes": nodes_data,
            "links": edges_data,
            "neuron_count": len(nodes_data),
            "synapse_count": len(edges_data)
        }

    def export_neuro_map(self) -> Path:
        """Exporta el estado cerebral a JSON y genera el HTML visual."""
        data = self.get_brain_state()
        
        # 1. Guardar JSON raw
        NEURO_STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        # 2. Generar HTML con visualización 3D
        html_content = self._generate_html_template(data)
        html_path = NEURO_DIR / "neuro_map.html"
        html_path.write_text(html_content, encoding="utf-8")
        
        return html_path

    def _generate_html_template(self, data: Dict[str, Any]) -> str:
        """Genera el dashboard HTML/WebGL autónomo."""
        json_payload = json.dumps(data)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ultragent Neuro-Vision (Hyper-V)</title>
    <script src="//unpkg.com/3d-force-graph"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{ margin: 0; background-color: #050505; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; overflow: hidden; }}
        #3d-graph {{ z-index: 1; position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
        
        /* HUD Overlay */
        #hud {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            background: rgba(10, 10, 15, 0.85);
            backdrop-filter: blur(12px);
            padding: 20px;
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 8px;
            width: 300px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }}
        
        h1 {{ font-size: 16px; margin: 0 0 10px 0; color: #64ffda; text-transform: uppercase; letter-spacing: 1px; }}
        .metric {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }}
        .metric-value {{ font-weight: bold; color: #fff; }}
        
        /* Node Tooltip */
        .node-tooltip {{
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #64ffda;
            padding: 10px;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
        }}
        
        /* Controls */
        #controls {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            z-index: 10;
            display: flex;
            gap: 10px;
        }}
        
        button {{
            background: rgba(100, 255, 218, 0.1);
            border: 1px solid #64ffda;
            color: #64ffda;
            padding: 8px 16px;
            cursor: pointer;
            font-family: inherit;
            font-size: 12px;
            transition: all 0.2s;
        }}
        
        button:hover {{ background: rgba(100, 255, 218, 0.2); box-shadow: 0 0 10px rgba(100, 255, 218, 0.2); }}

        /* Analysis Panel */
        #analysis-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 350px;
            background: rgba(10, 10, 15, 0.9);
            border-left: 1px solid rgba(100, 255, 218, 0.3);
            height: calc(100% - 40px);
            z-index: 5;
            padding: 20px;
            transform: translateX(400px);
            transition: transform 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
            overflow-y: auto;
        }}
        
        #analysis-panel.open {{ transform: translateX(0); }}
        
        .log-entry {{ 
            font-size: 10px; 
            border-left: 2px solid #333; 
            padding-left: 8px; 
            margin-bottom: 6px; 
            opacity: 0.7; 
        }}
        .log-entry.error {{ border-color: #ff5555; color: #ffaaaa; }}
        
    </style>
</head>
<body>
    <div id="3d-graph"></div>
    
    <div id="hud">
        <h1>Ultragent Hyper-V</h1>
        <div class="metric"><span>Neurons (Nodes)</span><span class="metric-value" id="node-count">0</span></div>
        <div class="metric"><span>Synapses (Links)</span><span class="metric-value" id="edge-count">0</span></div>
        <div class="metric"><span>System Status</span><span class="metric-value" style="color: #64ffda">ONLINE</span></div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
        <div id="selected-node-info">
            <div style="font-size: 10px; color: #888;">HOVER OVER A NODE</div>
        </div>
    </div>
    
    <div id="controls">
        <button onclick="zoomToFit()">RESET VIEW</button>
        <button onclick="toggleRotation()">AUTO ROTATE</button>
        <button onclick="refreshData()">REFRESH DATA</button>
        <button onclick="toggleAnalysis()">ANALYSIS PANEL</button>
    </div>
    
    <div id="analysis-panel">
        <h1>Deep Analysis Trace</h1>
        <div id="analysis-content">
            <p style="font-size: 11px; color: #aaa;">Select a node to view impact analysis and logs.</p>
        </div>
    </div>

    <script>
        // Data Injection
        const initData = {json_payload};
        
        // Config
        const CONFIG = {{
            nodeRelSize: 4,
            nodeResolution: 16,
            linkWidth: 1.5,
            particleWidth: 2,
        }};

        // State
        let isRotating = true;
        
        // Initialize Graph
        const Graph = ForceGraph3D()
            (document.getElementById('3d-graph'))
            .graphData(initData)
            .nodeLabel('id')
            .nodeColor(node => {{
                if (node.state && node.state.error_rate > 0) return '#ff5555'; // Error -> Red
                if (node.type === 'file') return '#64ffda'; // File -> Cyan
                if (node.type === 'module') return '#bd93f9'; // Module -> Purple
                if (node.type === 'class') return '#ff79c6'; // Class -> Pink
                return '#8be9fd';
            }})
            .nodeThreeObject(node => {{
                // Custom geometry logic could go here
                // For now standard spheres but mapped to brightness
                return false; 
            }})
            .nodeVal(node => {{
                // Size based on importance or degree
                const val = (node.metrics ? node.metrics.in_degree : 1) * 0.5 + 2;
                return val;
            }})
            .nodeOpacity(0.9)
            .linkDirectionalParticles(2) // Flow particles
            .linkDirectionalParticleSpeed(d => 0.005)
            .linkWidth(CONFIG.linkWidth)
            .linkColor(() => 'rgba(100, 255, 218, 0.2)')
            .onNodeHover(node => {{
                const infoDiv = document.getElementById('selected-node-info');
                document.body.style.cursor = node ? 'pointer' : null;
                
                if (node) {{
                    let logsHtml = '';
                    if (node.state && node.state.logs) {{
                        logsHtml = node.state.logs.map(l => `<div class="log-entry">${{l}}</div>`).join('');
                    }}
                    
                    infoDiv.innerHTML = `
                        <div style="font-weight: bold; color: #fff; margin-bottom: 5px;">${{node.id}}</div>
                        <div style="font-size: 11px; color: #bd93f9;">${{node.type}}</div>
                        <div style="margin-top: 10px; font-size: 10px;">ACTIVITY: ${{ (node.state.activation_level * 100).toFixed(0) }}%</div>
                        <div style="margin-top: 5px; font-size: 10px;">VARIABLES: ${{ Object.keys(node.state.active_variables || {{}}).length }}</div>
                        <div style="margin-top: 10px;">${{logsHtml}}</div>
                    `;
                }} else {{
                    infoDiv.innerHTML = '<div style="font-size: 10px; color: #888;">HOVER OVER A NODE</div>';
                }}
            }})
            .onNodeClick(node => {{
                // Focus camera on node
                const distance = 40;
                const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

                Graph.cameraPosition(
                    {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }}, // new position
                    node, // lookAt ({{ x, y, z }})
                    3000  // ms transition duration
                );
                
                populateAnalysis(node);
            }});

        // HUD Updates
        document.getElementById('node-count').innerText = initData.nodes.length;
        document.getElementById('edge-count').innerText = initData.links.length;

        // Auto Rotate
        let angle = 0;
        setInterval(() => {{
            if (isRotating) {{
                Graph.cameraPosition({{
                    x: 200 * Math.sin(angle),
                    z: 200 * Math.cos(angle)
                }});
                angle += 0.003;
            }}
        }}, 10);

        // Functions
        function zoomToFit() {{
            Graph.zoomToFit(400);
        }}
        
        function toggleRotation() {{
            isRotating = !isRotating;
        }}
        
        function toggleAnalysis() {{
            document.getElementById('analysis-panel').classList.toggle('open');
        }}
        
        function populateAnalysis(node) {{
            const panel = document.getElementById('analysis-content');
            document.getElementById('analysis-panel').classList.add('open');
            
            let varsHtml = '';
            if (node.state && node.state.active_variables) {{
                 varsHtml = Object.entries(node.state.active_variables)
                    .map(([k, v]) => `<div><span style="color:#64ffda">${{k}}:</span> ${{v}}</div>`)
                    .join('');
            }}
            
            let memsHtml = '';
            if (node.state && node.state.memories && node.state.memories.length > 0) {{
                memsHtml = node.state.memories
                    .map(m => `
                        <div style="background:#1a1a1a; padding:8px; border-left:2px solid #64ffda; margin-bottom:5px; font-size:11px;">
                            <div style="color:#888; font-size:9px; margin-bottom:2px;">IMPORTANCE: ${{m.importance}}</div>
                            ${{m.content}}
                        </div>
                    `).join('');
            }} else {{
                memsHtml = '<em style="color:#555">No related memory atoms found.</em>';
            }}
            
            panel.innerHTML = `
                <h3 style="color:#64ffda; border-bottom:1px solid #333; padding-bottom:10px; margin-bottom:15px;">
                    <span style="color:#888; font-size:0.8em;">NODE:</span> ${{node.id}}
                </h3>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color:#888; font-size:10px; text-transform:uppercase; letter-spacing:1px;">Atomic Memories (Cortex)</h4>
                    <div style="margin-top:10px;">
                        ${{memsHtml}}
                    </div>
                </div>

                <div style="margin-bottom: 20px;">
                    <h4 style="color:#888; font-size:10px; text-transform:uppercase; letter-spacing:1px;">Active Variables</h4>
                    <div style="font-family: monospace; font-size: 11px; background: #111; padding: 10px; margin-top:5px; border-radius:4px;">
                        ${{varsHtml || '<em style="color:#555">None</em>'}}
                    </div>
                </div>
                
                <div>
                    <h4 style="color:#888; font-size:10px; text-transform:uppercase; letter-spacing:1px;">Impact Prediction</h4>
                    <button style="width:100%; padding:10px; margin-top:10px; background:#64ffda; color:#0a192f; border:none; font-weight:bold; cursor:pointer;" 
                            onclick="alert('Triggering Impact Analysis via MCP...')">RUN SIMULATION</button>
                    <div style="font-size: 10px; color: #888; margin-top: 5px;">
                        Use Ultragent CLI to run <code>analyze_impact('${{node.id}}')</code>
                    </div>
                </div>
            `;
        }}
        
        function refreshData() {{
            // In a real implementation this would fetch /neuro_state.json
            console.log("Refreshing data...");
            location.reload();
        }}
        
    </script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_neuro_instance: Optional[NeuroArchitect] = None
_neuro_lock = Lock()
_current_neuro_project: Optional[str] = None

def get_neuro_architect(project_root: Optional[str] = None) -> NeuroArchitect:
    """
    Obtiene la instancia singleton del NeuroArchitect.
    
    Args:
        project_root: Optional path to target project. If provided and different
                      from current, a new NeuroArchitect instance is created.
    """
    global _neuro_instance, _current_neuro_project
    with _neuro_lock:
        if _neuro_instance is None or _current_neuro_project != project_root:
            _neuro_instance = NeuroArchitect(project_root=project_root)
            _current_neuro_project = project_root
        return _neuro_instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neuro = get_neuro_architect()
    
    # Test simulation
    print("Simulating activity...")
    neuro.ingest_telemetry("main", "execution", {})
    neuro.ingest_telemetry("scout", "error", {"message": "Rate limit exceeded"})
    
    print("\nImpact Analysis for 'scout':")
    impact = neuro.analyze_impact("scout")
    print(f"Risk Score: {impact.risk_score}")
    print(f"Direct Impact: {impact.direct_impact}")
    
    print("\nExporting NeuroMap...")
    path = neuro.export_neuro_map()
    print(f"Saved to {path}")
