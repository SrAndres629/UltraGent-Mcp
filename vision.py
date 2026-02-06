"""
ULTRAGENT VISION v0.1
=====================
Módulo de visualización arquitectónica con NetworkX.

Implementa:
- Análisis de dependencias como grafo dirigido G = (V, E)
- Detección de ciclos (marcados en ROJO)
- Clustering por paquete/módulo
- Generación de architecture_map.png
- Filtros para grafos legibles
"""

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional, List, Dict
from cortex import get_cortex

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import networkx as nx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
REPORTS_DIR = AI_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de visualización
GRAPH_CONFIG = {
    "max_nodes": 100,
    "hide_stdlib": True,
    "hide_tests": False,
    "figsize": (16, 12),
    "dpi": 150,
    "node_size": 2000,
    "font_size": 8,
}

# Paquetes de stdlib a ignorar
STDLIB_PACKAGES = {
    "os", "sys", "re", "json", "typing", "pathlib", "datetime",
    "collections", "itertools", "functools", "logging", "threading",
    "asyncio", "uuid", "hashlib", "base64", "tempfile", "dataclasses",
    "enum", "abc", "copy", "time", "math", "random", "io", "contextlib",
}

# Logger
logger = logging.getLogger("ultragent.vision")


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DependencyNode:
    """Nodo en el grafo de dependencias."""
    
    name: str
    node_type: str  # file, class, function, package
    file_path: Optional[str] = None
    line_count: int = 0
    is_external: bool = False
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "file_path": self.file_path,
            "line_count": self.line_count,
            "is_external": self.is_external,
            "tags": self.tags,
            "importance": self.importance,
            "metadata": self.metadata,
        }


@dataclass
class DependencyEdge:
    """Arista en el grafo de dependencias."""
    
    source: str
    target: str
    edge_type: str  # import, extends, uses
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
        }


@dataclass
class ArchitectureReport:
    """Reporte de arquitectura."""
    
    timestamp: datetime
    nodes: list[DependencyNode]
    edges: list[DependencyEdge]
    cycles: list[list[str]]
    hotspots: list[str]  # Nodos con alto in-degree
    graph_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "cycle_count": len(self.cycles),
            "cycles": self.cycles,
            "hotspots": self.hotspots,
            "graph_path": self.graph_path,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class ImportAnalyzer:
    """Analiza imports de archivos Python."""
    
    @staticmethod
    def extract_imports(file_path: Path) -> list[tuple[str, str]]:
        """
        Extrae imports de un archivo Python.
        
        Returns:
            list[tuple[source_file, imported_module]]
        """
        imports = []
        source_name = file_path.stem
        
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        imports.append((source_name, module))
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        imports.append((source_name, module))
                        
        except SyntaxError as e:
            logger.warning(f"Syntax error en {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error analizando {file_path}: {e}")
        
        return imports


# ═══════════════════════════════════════════════════════════════════════════════
# VISION ARCHITECT
# ═══════════════════════════════════════════════════════════════════════════════

class VisionArchitect:
    """
    Arquitecto de visualización de dependencias.
    
    Genera grafos de dependencias del proyecto y detecta
    ciclos y hotspots.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self._project_root = project_root or PROJECT_ROOT
        self._lock = Lock()
        
        # Estadísticas
        self._stats = {
            "scans": 0,
            "graphs_generated": 0,
            "cycles_found": 0,
        }
        
        self._cortex = get_cortex()
        logger.info(f"VisionArchitect inicializado: {self._project_root}")
    
    def scan_project(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> tuple[list[DependencyNode], list[DependencyEdge]]:
        """
        Escanea el proyecto y extrae nodos y aristas.
        
        Args:
            include_patterns: Patrones glob a incluir (default: *.py)
            exclude_patterns: Patrones a excluir
            
        Returns:
            tuple[nodes, edges]
        """
        include_patterns = include_patterns or ["*.py"]
        exclude_patterns = exclude_patterns or [
            "*.pyc", "__pycache__", ".venv", "venv", "node_modules",
            ".git", ".ai", "test_*", "*_test.py",
        ]
        
        nodes: dict[str, DependencyNode] = {}
        edges: list[DependencyEdge] = []
        
        # Buscar archivos Python
        for pattern in include_patterns:
            for file_path in self._project_root.rglob(pattern):
                # Aplicar exclusiones
                skip = False
                for exclude in exclude_patterns:
                    if file_path.match(exclude) or any(
                        p in str(file_path) for p in exclude_patterns
                    ):
                        skip = True
                        break
                
                if skip:
                    continue
                
                # Crear nodo para el archivo
                node_name = file_path.stem
                if node_name not in nodes:
                    line_count = len(file_path.read_text(encoding="utf-8").splitlines())
                    nodes[node_name] = DependencyNode(
                        name=node_name,
                        node_type="file",
                        file_path=str(file_path),
                        line_count=line_count,
                    )
                
                # Extraer imports
                imports = ImportAnalyzer.extract_imports(file_path)
                for source, target in imports:
                    # Crear nodo para dependencia si no existe
                    if target not in nodes:
                        is_stdlib = target in STDLIB_PACKAGES
                        nodes[target] = DependencyNode(
                            name=target,
                            node_type="package" if is_stdlib else "module",
                            is_external=True,
                        )
                    
                    # Crear arista
                    edges.append(DependencyEdge(
                        source=source,
                        target=target,
                        edge_type="import",
                    ))
        
        # Inyectar MEMORIAS del Cortex
        try:
            memories = self._cortex.get_all_memories()
            for m in memories:
                node_id = f"mem_{m.id}"
                nodes[node_id] = DependencyNode(
                    name=f"Memory #{m.id}",
                    node_type="memory",
                    tags=m.tags,
                    importance=m.importance,
                    metadata={"content": m.content}
                )
                
                # Intentar vincular memoria a archivos por tags
                for tag in m.tags:
                    for name, node in nodes.items():
                        if node.node_type == "file" and tag.lower() in name.lower():
                            edges.append(DependencyEdge(
                                source=node_id,
                                target=name,
                                edge_type="relates_to"
                            ))
        except Exception as e:
            logger.warning(f"Error cargando memorias en Vision: {e}")
        
        with self._lock:
            self._stats["scans"] += 1
        
        logger.info(f"Escaneado: {len(nodes)} nodos, {len(edges)} aristas")
        
        return list(nodes.values()), edges
    
    def build_graph(
        self,
        nodes: list[DependencyNode],
        edges: list[DependencyEdge],
        hide_stdlib: bool = True,
    ) -> nx.DiGraph:
        """Construye grafo NetworkX desde nodos y aristas."""
        G = nx.DiGraph()
        
        # Filtrar stdlib si es necesario
        if hide_stdlib:
            stdlib_names = {n.name for n in nodes if n.name in STDLIB_PACKAGES}
            nodes = [n for n in nodes if n.name not in stdlib_names]
            edges = [
                e for e in edges
                if e.source not in stdlib_names and e.target not in stdlib_names
            ]
        
        # Añadir nodos
        for node in nodes:
            G.add_node(
                node.name,
                node_type=node.node_type,
                file_path=node.file_path,
                line_count=node.line_count,
                is_external=node.is_external,
            )
        
        # Añadir aristas
        for edge in edges:
            if G.has_node(edge.source) and G.has_node(edge.target):
                G.add_edge(edge.source, edge.target, edge_type=edge.edge_type)
        
        return G
    
    def find_cycles(self, G: nx.DiGraph) -> list[list[str]]:
        """Encuentra ciclos en el grafo."""
        cycles = []
        try:
            cycles_gen = nx.simple_cycles(G)
            for cycle in cycles_gen:
                if len(cycle) > 1:  # Solo ciclos reales
                    cycles.append(cycle)
                    if len(cycles) >= 10:  # Limitar
                        break
        except nx.NetworkXNoCycle:
            pass
        except Exception as e:
            logger.warning(f"Error buscando ciclos: {e}")
        
        with self._lock:
            self._stats["cycles_found"] += len(cycles)
        
        return cycles
    
    def find_hotspots(
        self,
        G: nx.DiGraph,
        top_n: int = 5,
    ) -> list[str]:
        """Encuentra nodos con alto in-degree (muchas dependencias)."""
        in_degrees = dict(G.in_degree())
        sorted_nodes = sorted(
            in_degrees.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [node for node, degree in sorted_nodes[:top_n] if degree > 0]
    
    def generate_graph_image(
        self,
        G: nx.DiGraph,
        cycles: list[list[str]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Genera imagen PNG del grafo.
        
        Los ciclos se marcan en ROJO.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"architecture_map_{timestamp}.png"
        
        # Crear figura
        fig, ax = plt.subplots(
            figsize=GRAPH_CONFIG["figsize"],
            dpi=GRAPH_CONFIG["dpi"],
        )
        
        # Layout
        if len(G.nodes()) > 0:
            try:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            except Exception:
                pos = nx.circular_layout(G)
        else:
            pos = {}
        
        # Colores de nodos
        node_colors = []
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        
        for node in G.nodes():
            data = G.nodes[node]
            if node in cycle_nodes:
                node_colors.append("#FF4444")  # ROJO para ciclos
            elif data.get("is_external"):
                node_colors.append("#888888")  # Gris para externos
            elif data.get("node_type") == "file":
                node_colors.append("#4CAF50")  # Verde para archivos
            elif data.get("node_type") == "memory":
                node_colors.append("#9C27B0")  # Púrpura para memorias
            else:
                node_colors.append("#2196F3")  # Azul para otros
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=GRAPH_CONFIG["node_size"],
            ax=ax,
            alpha=0.9,
        )
        
        # Colores de aristas
        edge_colors = []
        for u, v in G.edges():
            if u in cycle_nodes and v in cycle_nodes:
                edge_colors.append("#FF0000")  # Rojo para ciclos
            else:
                edge_colors.append("#CCCCCC")
        
        # Dibujar aristas
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=15,
            ax=ax,
            alpha=0.6,
        )
        
        # Etiquetas
        nx.draw_networkx_labels(
            G, pos,
            font_size=GRAPH_CONFIG["font_size"],
            font_weight="bold",
            ax=ax,
        )
        
        # Título
        ax.set_title(
            f"Ultragent Architecture Map\n"
            f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())} | Cycles: {len(cycles)}",
            fontsize=12,
            fontweight="bold",
        )
        
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=10, label='Files'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9C27B0', markersize=10, label='Memories'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=10, label='External'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', markersize=10, label='Cycles'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Guardar
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        with self._lock:
            self._stats["graphs_generated"] += 1
        
        logger.info(f"Grafo generado: {output_path}")
        
        return output_path
    
    def generate_dependency_graph(
        self,
        output_path: Optional[str] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ArchitectureReport:
        """
        Genera grafo completo de dependencias del proyecto.
        
        Returns:
            ArchitectureReport con análisis y path a la imagen
        """
        # Escanear proyecto
        nodes, edges = self.scan_project(
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        
        # Construir grafo
        G = self.build_graph(
            nodes, edges,
            hide_stdlib=GRAPH_CONFIG["hide_stdlib"],
        )
        
        # Análisis
        cycles = self.find_cycles(G)
        hotspots = self.find_hotspots(G)
        
        # Generar imagen
        graph_path = self.generate_graph_image(
            G, cycles,
            output_path=Path(output_path) if output_path else None,
        )
        
        # Crear reporte
        report = ArchitectureReport(
            timestamp=datetime.now(),
            nodes=nodes,
            edges=edges,
            cycles=cycles,
            hotspots=hotspots,
            graph_path=str(graph_path),
        )
        
        logger.info(
            f"Reporte generado: {len(nodes)} nodos, "
            f"{len(edges)} aristas, {len(cycles)} ciclos"
        )
        
        return report
    
    async def generate_mermaid_graph(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> str:
        """
        Genera una representación en formato Mermaid.js del grafo de dependencias.
        
        Returns:
            str: Contenido en formato Mermaid.
        """
        nodes, edges = self.scan_project(include_patterns, exclude_patterns)
        
        mermaid_lines = ["graph TD"]
        
        # 1. Definir Nodos con estilos
        for node in nodes:
            # Escapar nombres para Mermaid
            clean_name = str(node.name).replace(".", "_").replace("-", "_")
            if node.node_type == "file":
                mermaid_lines.append(f'    {clean_name}["{node.name} (File)"]')
            elif node.node_type == "package":
                mermaid_lines.append(f'    {clean_name}["{node.name} (Package)"]:::pkg')
            elif node.node_type == "memory":
                mermaid_lines.append(f'    {clean_name}["{node.name} (Memory)"]:::mem')
            else:
                mermaid_lines.append(f'    {clean_name}["{node.name}"]')

        # 2. Definir Aristas
        for edge in edges:
            source = str(edge.source).replace(".", "_").replace("-", "_")
            target = str(edge.target).replace(".", "_").replace("-", "_")
            
            # Flecha según el tipo de relación
            if edge.edge_type == "import":
                mermaid_lines.append(f"    {source} --> {target}")
            elif edge.edge_type == "relates_to":
                mermaid_lines.append(f"    {source} -.-> {target}")
            else:
                mermaid_lines.append(f"    {source} --- {target}")

        # 3. Estilos
        mermaid_lines.append("    classDef pkg fill:#f96,stroke:#333,stroke-width:2px;")
        mermaid_lines.append("    classDef mem fill:#bbf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;")
        
        return "\n".join(mermaid_lines)

    def get_status(self) -> dict:
        """Retorna estado del Vision."""
        return {
            "project_root": str(self._project_root),
            "reports_dir": str(REPORTS_DIR),
            "stats": dict(self._stats),
            "config": GRAPH_CONFIG,
        }

    def get_cycles_report(self) -> dict:
        """Retorna un reporte detallado de ciclos detectados."""
        nodes, edges = self.scan_project()
        G = self.build_graph(nodes, edges)
        cycles = self.find_cycles(G)
        return {
            "cycle_count": len(cycles),
            "cycles": cycles,
            "impacted_nodes": list(set([n for c in cycles for n in c]))
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_vision_instance: Optional[VisionArchitect] = None
_vision_lock = Lock()
_current_project_root: Optional[Path] = None


def get_vision(project_root: Optional[str] = None) -> VisionArchitect:
    """
    Obtiene la instancia singleton del Vision.
    
    Args:
        project_root: Optional path to target project. If provided and different
                      from current, a new VisionArchitect instance is created.
    """
    global _vision_instance, _current_project_root
    
    target_root = Path(project_root) if project_root else PROJECT_ROOT
    
    with _vision_lock:
        # If project_root changed, recreate the instance
        if _vision_instance is None or _current_project_root != target_root:
            _vision_instance = VisionArchitect(project_root=target_root)
            _current_project_root = target_root
        return _vision_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    print("=" * 60)
    print("ULTRAGENT VISION v0.1 - Test")
    print("=" * 60)
    
    vision = get_vision()
    print(f"Status: {vision.get_status()}")
    
    print("\nGenerando grafo de dependencias...")
    report = vision.generate_dependency_graph()
    
    print(f"\nResultado:")
    print(f"  Nodos: {len(report.nodes)}")
    print(f"  Aristas: {len(report.edges)}")
    print(f"  Ciclos: {len(report.cycles)}")
    print(f"  Hotspots: {report.hotspots}")
    print(f"  Imagen: {report.graph_path}")
