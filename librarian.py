"""
ULTRAGENT LIBRARIAN v0.1
========================
Módulo de memoria profunda con Tree-sitter (Skeletonization) y ChromaDB (RAG Local).

Implementa:
- Análisis AST multi-lenguaje con graméticas dinámicas
- Extracción de esqueletos (firmas + docstrings) sin implementación
- Embeddings GPU (sentence-transformers) para búsqueda semántica
- Persistencia vectorial en .ai/chroma/
- Cross-referencing y context precision
"""

import hashlib
import importlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer # Lazy loaded

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
CHROMA_DIR = AI_DIR / "chroma"

# Modelo de embeddings (80MB, eficiente para RTX 3060 6GB)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Configuración de colecciones
SKELETON_COLLECTION = "code_skeletons"
FULLCODE_COLLECTION = "full_code"

# Logger
logger = logging.getLogger("ultragent.librarian")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAMÁTICAS DE LENGUAJES
# ═══════════════════════════════════════════════════════════════════════════════

# Mapeo extensión -> (módulo_tree_sitter, nombre_lenguaje)
LANGUAGE_GRAMMARS: dict[str, tuple[str, str]] = {
    ".py": ("tree_sitter_python", "python"),
    ".js": ("tree_sitter_javascript", "javascript"),
    ".jsx": ("tree_sitter_javascript", "javascript"),
    ".ts": ("tree_sitter_typescript", "typescript"),
    ".tsx": ("tree_sitter_typescript", "tsx"),
}

# Nodos AST a extraer por lenguaje
EXTRACTION_NODES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeSkeleton:
    """Representa el esqueleto de un elemento de código."""
    
    name: str
    node_type: str  # function, class, method, interface
    signature: str  # Firma completa
    docstring: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    language: str
    file_hash: str
    parent: Optional[str] = None  # Para métodos: nombre de la clase
    
    def to_embedding_text(self) -> str:
        """Genera texto para embedding."""
        parts = [
            f"{self.node_type}: {self.name}",
            f"Signature: {self.signature}",
        ]
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        if self.parent:
            parts.append(f"Parent: {self.parent}")
        return "\n".join(parts)
    
    def to_metadata(self) -> dict:
        """Genera metadata para ChromaDB."""
        return {
            "name": self.name,
            "node_type": self.node_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "file_hash": self.file_hash,
            "parent": self.parent or "",
            "indexed_at": datetime.now().isoformat(),
        }


class UnsupportedLanguageError(Exception):
    """Lenguaje no soportado."""
    pass


class ParsingError(Exception):
    """Error al parsear el código."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SKELETON EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SkeletonExtractor:
    """
    Extrae esqueletos de código usando Tree-sitter.
    
    Soporta carga dinámica de gramáticas para múltiples lenguajes.
    """
    
    def __init__(self):
        self._parsers: dict[str, Any] = {}
        self._lock = Lock()
    
    def _get_parser(self, extension: str) -> tuple[Any, str]:
        """
        Obtiene o crea un parser para la extensión dada.
        
        Returns:
            tuple[Parser, language_name]
        """
        from tree_sitter import Parser
        
        grammar_info = LANGUAGE_GRAMMARS.get(extension)
        if not grammar_info:
            raise UnsupportedLanguageError(f"Extensión no soportada: {extension}")
        
        module_name, lang_name = grammar_info
        
        with self._lock:
            if lang_name not in self._parsers:
                try:
                    grammar_module = importlib.import_module(module_name)
                    parser = Parser(grammar_module.language())
                    self._parsers[lang_name] = parser
                    logger.info(f"Parser cargado: {lang_name}")
                except ImportError as e:
                    raise UnsupportedLanguageError(
                        f"Gramática no instalada: {module_name}. "
                        f"Ejecutar: uv add {module_name}"
                    ) from e
            
            return self._parsers[lang_name], lang_name
    
    def _extract_docstring(self, node: Any, source: bytes, lang: str) -> Optional[str]:
        """Extrae docstring de un nodo."""
        if lang == "python":
            # Buscar expression_statement con string al inicio del body
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        if block_child.type == "expression_statement":
                            for expr_child in block_child.children:
                                if expr_child.type == "string":
                                    text = source[expr_child.start_byte:expr_child.end_byte].decode("utf-8")
                                    # Limpiar quotes
                                    text = text.strip("\"'")
                                    if text.startswith('""'):
                                        text = text[2:-2]
                                    return text.strip()
                        break
        
        elif lang in ("javascript", "typescript"):
            # Buscar comment antes del nodo
            # (simplificado - en producción usar reglas más complejas)
            pass
        
        return None
    
    def _extract_signature(self, node: Any, source: bytes, lang: str) -> str:
        """Extrae la firma de una función/clase sin el body."""
        if lang == "python":
            # Para Python: desde el inicio hasta los dos puntos (:)
            for child in node.children:
                if child.type == ":":
                    return source[node.start_byte:child.end_byte].decode("utf-8").strip()
            # Fallback: primera línea
            text = source[node.start_byte:node.end_byte].decode("utf-8")
            first_line = text.split("\n")[0]
            return first_line.strip()
        
        elif lang in ("javascript", "typescript"):
            # Para JS/TS: hasta la llave de apertura
            for child in node.children:
                if child.type == "statement_block" or child.type == "{":
                    return source[node.start_byte:child.start_byte].decode("utf-8").strip()
            # Fallback
            text = source[node.start_byte:node.end_byte].decode("utf-8")
            return text.split("{")[0].strip()
        
        return source[node.start_byte:node.end_byte].decode("utf-8").split("\n")[0]
    
    def _get_node_name(self, node: Any, source: bytes, lang: str) -> str:
        """Extrae el nombre de un nodo."""
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                return source[child.start_byte:child.end_byte].decode("utf-8")
            # Para decorated_definition, buscar en el hijo
            if child.type in EXTRACTION_NODES.get(lang, []):
                return self._get_node_name(child, source, lang)
        return "anonymous"
    
    def _walk_tree(
        self,
        node: Any,
        source: bytes,
        lang: str,
        file_path: str,
        file_hash: str,
        parent: Optional[str] = None,
    ) -> list[CodeSkeleton]:
        """Recorre el AST y extrae esqueletos."""
        skeletons = []
        target_nodes = EXTRACTION_NODES.get(lang, [])
        
        if node.type in target_nodes:
            name = self._get_node_name(node, source, lang)
            signature = self._extract_signature(node, source, lang)
            docstring = self._extract_docstring(node, source, lang)
            
            # Determinar tipo de nodo
            node_type = "function"
            if "class" in node.type:
                node_type = "class"
            elif "interface" in node.type:
                node_type = "interface"
            elif "method" in node.type:
                node_type = "method"
            elif "decorated" in node.type:
                # Buscar el tipo real
                for child in node.children:
                    if "function" in child.type:
                        node_type = "function"
                    elif "class" in child.type:
                        node_type = "class"
            
            skeleton = CodeSkeleton(
                name=name,
                node_type=node_type,
                signature=signature,
                docstring=docstring,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=lang,
                file_hash=file_hash,
                parent=parent,
            )
            skeletons.append(skeleton)
            
            # Para clases, buscar métodos hijos
            if node_type == "class":
                for child in node.children:
                    skeletons.extend(
                        self._walk_tree(child, source, lang, file_path, file_hash, name)
                    )
                return skeletons
        
        # Continuar recursión para otros nodos
        for child in node.children:
            skeletons.extend(
                self._walk_tree(child, source, lang, file_path, file_hash, parent)
            )
        
        return skeletons
    
    def extract(self, file_path: str) -> list[CodeSkeleton]:
        """
        Extrae esqueletos de un archivo de código.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            list[CodeSkeleton]: Lista de esqueletos extraídos
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            UnsupportedLanguageError: Si el lenguaje no está soportado
            ParsingError: Si hay error al parsear
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        extension = path.suffix.lower()
        parser, lang = self._get_parser(extension)
        
        try:
            source = path.read_bytes()
            file_hash = hashlib.md5(source).hexdigest()
            
            tree = parser.parse(source)
            
            if tree.root_node is None:
                raise ParsingError(f"No se pudo parsear: {file_path}")
            
            skeletons = self._walk_tree(
                tree.root_node,
                source,
                lang,
                str(path.absolute()),
                file_hash,
            )
            
            logger.info(f"Extraídos {len(skeletons)} esqueletos de {path.name}")
            return skeletons
            
        except UnicodeDecodeError as e:
            raise ParsingError(f"Error de encoding en {file_path}: {e}") from e
        except Exception as e:
            raise ParsingError(f"Error al parsear {file_path}: {e}") from e
    
    def format_skeleton(self, skeleton: CodeSkeleton) -> str:
        """Formatea un esqueleto para visualización."""
        lines = [f"{skeleton.node_type.upper()}: {skeleton.name}"]
        lines.append(f"  Signature: {skeleton.signature}")
        if skeleton.docstring:
            lines.append(f"  Docstring: {skeleton.docstring[:100]}...")
        lines.append(f"  Location: {Path(skeleton.file_path).name}:{skeleton.start_line}-{skeleton.end_line}")
        if skeleton.parent:
            lines.append(f"  Parent: {skeleton.parent}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CODE LIBRARIAN
# ═══════════════════════════════════════════════════════════════════════════════

class CodeLibrarian:
    """
    Biblioteca de código con búsqueda semántica.
    
    Combina Tree-sitter para análisis AST y ChromaDB para
    almacenamiento vectorial persistente.
    """
    
    def __init__(self, persist_dir: Optional[Path] = None):
        self._persist_dir = persist_dir or CHROMA_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Colecciones
        self._skeleton_collection = self._client.get_or_create_collection(
            name=SKELETON_COLLECTION,
            metadata={"description": "Code skeletons (signatures + docstrings)"},
        )
        
        self._fullcode_collection = self._client.get_or_create_collection(
            name=FULLCODE_COLLECTION,
            metadata={"description": "Full code blocks for detailed retrieval"},
        )
        
        # Componentes
        self._extractor = SkeletonExtractor()
        self._embedder: Any = None # Lazy loaded SentenceTransformer
        self._lock = Lock()
        
        # Estadísticas
        self._stats = {
            "files_indexed": 0,
            "skeletons_indexed": 0,
            "queries_executed": 0,
        }
        
        logger.info(f"CodeLibrarian inicializado en {self._persist_dir}")
    
    def _get_embedder(self) -> Any:
        """Lazy-load del modelo de embeddings con GPU."""
        if self._embedder is None:
            # Detectar dispositivo
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            
            logger.info(f"Cargando modelo de embeddings en {device}...")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                 logger.error("sentence_transformers not installed")
                 raise

            self._embedder = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device,
            )
            logger.info(f"Modelo cargado: {EMBEDDING_MODEL} ({device})")
        
        return self._embedder
    
    def _generate_doc_id(self, file_path: str, name: str, start_line: int) -> str:
        """Genera ID único para un documento."""
        key = f"{file_path}:{name}:{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def index_file(self, file_path: str) -> dict:
        """
        Indexa un archivo en la biblioteca.
        
        Extrae esqueletos y los almacena con embeddings.
        Si el archivo ya estaba indexado, elimina los vectores antiguos.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            dict: Resultado de la indexación
        """
        path = Path(file_path)
        
        try:
            # Verificar extensión soportada
            if path.suffix.lower() not in LANGUAGE_GRAMMARS:
                return {
                    "success": False,
                    "error": f"Lenguaje no soportado: {path.suffix}",
                    "file": str(path),
                }
            
            # Eliminar vectores anteriores de este archivo
            self._remove_file_vectors(str(path.absolute()))
            
            # Extraer esqueletos
            skeletons = self._extractor.extract(str(path))
            
            if not skeletons:
                return {
                    "success": True,
                    "message": "No se encontraron elementos para indexar",
                    "file": str(path),
                    "skeletons": 0,
                }
            
            # Generar embeddings
            embedder = self._get_embedder()
            texts = [s.to_embedding_text() for s in skeletons]
            embeddings = embedder.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            # Almacenar en ChromaDB
            ids = []
            metadatas = []
            documents = []
            
            for i, skeleton in enumerate(skeletons):
                doc_id = self._generate_doc_id(
                    skeleton.file_path,
                    skeleton.name,
                    skeleton.start_line,
                )
                ids.append(doc_id)
                metadatas.append(skeleton.to_metadata())
                documents.append(skeleton.signature)
            
            self._skeleton_collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents,
            )
            
            # Actualizar estadísticas
            with self._lock:
                self._stats["files_indexed"] += 1
                self._stats["skeletons_indexed"] += len(skeletons)
            
            logger.info(f"Indexado: {path.name} ({len(skeletons)} esqueletos)")
            
            return {
                "success": True,
                "file": str(path),
                "skeletons": len(skeletons),
                "elements": [s.name for s in skeletons],
            }
            
        except FileNotFoundError as e:
            logger.error(f"Archivo no encontrado: {file_path}")
            return {"success": False, "error": str(e), "file": str(path)}
        except UnsupportedLanguageError as e:
            logger.warning(f"Lenguaje no soportado: {e}")
            return {"success": False, "error": str(e), "file": str(path)}
        except Exception as e:
            logger.error(f"Error indexando {file_path}: {e}")
            return {"success": False, "error": str(e), "file": str(path)}
    
    def _remove_file_vectors(self, file_path: str) -> int:
        """Elimina vectores de un archivo específico."""
        try:
            # Buscar documentos con este file_path
            results = self._skeleton_collection.get(
                where={"file_path": file_path},
            )
            
            if results["ids"]:
                self._skeleton_collection.delete(ids=results["ids"])
                logger.debug(f"Eliminados {len(results['ids'])} vectores de {file_path}")
                return len(results["ids"])
            
            return 0
        except Exception as e:
            logger.error(f"Error eliminando vectores: {e}")
            return 0
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        language: Optional[str] = None,
        node_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Búsqueda semántica en la biblioteca de código.
        
        Args:
            query: Consulta en lenguaje natural
            n_results: Número máximo de resultados
            language: Filtrar por lenguaje (python, javascript, etc.)
            node_type: Filtrar por tipo (function, class, method)
            
        Returns:
            list[dict]: Resultados ordenados por relevancia
        """
        try:
            # Generar embedding de la query
            embedder = self._get_embedder()
            query_embedding = embedder.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            # Construir filtro
            where_filter = None
            if language or node_type:
                conditions = []
                if language:
                    conditions.append({"language": language})
                if node_type:
                    conditions.append({"node_type": node_type})
                
                if len(conditions) == 1:
                    where_filter = conditions[0]
                else:
                    where_filter = {"$and": conditions}
            
            # Ejecutar búsqueda
            results = self._skeleton_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )
            
            # Actualizar estadísticas
            with self._lock:
                self._stats["queries_executed"] += 1
            
            # Formatear resultados
            formatted = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else None
                    
                    formatted.append({
                        "id": doc_id,
                        "name": metadata.get("name"),
                        "node_type": metadata.get("node_type"),
                        "signature": results["documents"][0][i],
                        "file_path": metadata.get("file_path"),
                        "start_line": metadata.get("start_line"),
                        "end_line": metadata.get("end_line"),
                        "language": metadata.get("language"),
                        "parent": metadata.get("parent"),
                        "relevance": 1 - distance if distance else None,
                    })
            
            logger.info(f"Búsqueda '{query[:30]}...': {len(formatted)} resultados")
            return formatted
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    def get_file_skeleton(self, file_path: str) -> dict:
        """
        Obtiene el esqueleto completo de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            dict: Esqueleto formateado del archivo
        """
        try:
            skeletons = self._extractor.extract(file_path)
            
            formatted = {
                "file": file_path,
                "elements": [],
                "summary": {},
            }
            
            # Contar por tipo
            type_counts: dict[str, int] = {}
            
            for skeleton in skeletons:
                type_counts[skeleton.node_type] = type_counts.get(skeleton.node_type, 0) + 1
                
                formatted["elements"].append({
                    "name": skeleton.name,
                    "type": skeleton.node_type,
                    "signature": skeleton.signature,
                    "docstring": skeleton.docstring,
                    "line": skeleton.start_line,
                    "parent": skeleton.parent,
                })
            
            formatted["summary"] = type_counts
            return formatted
            
        except Exception as e:
            return {
                "file": file_path,
                "error": str(e),
                "elements": [],
            }
    
    def get_status(self) -> dict:
        """Retorna estado del Librarian."""
        try:
            skeleton_count = self._skeleton_collection.count()
        except Exception:
            skeleton_count = 0
        
        return {
            "persist_dir": str(self._persist_dir),
            "skeleton_count": skeleton_count,
            "stats": dict(self._stats),
            "supported_languages": list(LANGUAGE_GRAMMARS.keys()),
            "embedder_loaded": self._embedder is not None,
        }
    
    def index_directory(
        self,
        directory: str,
        extensions: Optional[list[str]] = None,
        recursive: bool = True,
    ) -> dict:
        """
        Indexa todos los archivos de un directorio.
        
        Args:
            directory: Ruta al directorio
            extensions: Extensiones a indexar (default: todas soportadas)
            recursive: Buscar recursivamente
            
        Returns:
            dict: Resumen de indexación
        """
        path = Path(directory)
        if not path.exists():
            return {"success": False, "error": f"Directorio no encontrado: {directory}"}
        
        target_extensions = set(extensions or LANGUAGE_GRAMMARS.keys())
        
        files = []
        if recursive:
            for ext in target_extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in target_extensions:
                files.extend(path.glob(f"*{ext}"))
        
        # Filtrar archivos en directorios ignorados
        ignored_patterns = ["node_modules", "__pycache__", ".git", ".venv", "venv"]
        files = [
            f for f in files
            if not any(p in str(f) for p in ignored_patterns)
        ]
        
        results = {
            "success": True,
            "directory": str(path),
            "files_found": len(files),
            "files_indexed": 0,
            "total_skeletons": 0,
            "errors": [],
        }
        
        for file in files:
            result = self.index_file(str(file))
            if result.get("success"):
                results["files_indexed"] += 1
                results["total_skeletons"] += result.get("skeletons", 0)
            else:
                results["errors"].append({
                    "file": str(file),
                    "error": result.get("error"),
                })
        
        logger.info(
            f"Indexado directorio: {results['files_indexed']}/{results['files_found']} archivos, "
            f"{results['total_skeletons']} esqueletos"
        )
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_librarian_instance: Optional[CodeLibrarian] = None
_librarian_lock = Lock()


def get_librarian() -> CodeLibrarian:
    """Obtiene la instancia singleton del Librarian."""
    global _librarian_instance
    with _librarian_lock:
        if _librarian_instance is None:
            _librarian_instance = CodeLibrarian()
        return _librarian_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARA TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    print("=" * 60)
    print("ULTRAGENT LIBRARIAN v0.1 - Test")
    print("=" * 60)
    
    librarian = get_librarian()
    print(f"Status: {librarian.get_status()}")
    
    # Test con archivo propio
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nIndexando: {file_path}")
        result = librarian.index_file(file_path)
        print(f"Resultado: {result}")
        
        print(f"\nEsqueleto:")
        skeleton = librarian.get_file_skeleton(file_path)
        for elem in skeleton.get("elements", []):
            print(f"  {elem['type']}: {elem['name']} (L{elem['line']})")
    else:
        print("\nUso: python librarian.py <archivo.py>")
