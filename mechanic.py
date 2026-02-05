"""
ULTRAGENT MECHANIC v0.1
=======================
Módulo de ejecución segura de código en Docker sandbox.

Implementa:
- Contenedores efímeros con auto-destrucción
- Límites de recursos (CPU, RAM, timeout)
- Aislamiento de red y filesystem
- Captura de STDOUT/STDERR
- Logs persistentes en .ai/logs/mechanic/
"""

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
AI_DIR = PROJECT_ROOT / os.getenv("AI_CORE_DIR", ".ai")
OPENHANDS_WORKSPACE = os.getenv("OPENHANDS_WORKSPACE")
WORKSPACE_DIR = Path(OPENHANDS_WORKSPACE) if OPENHANDS_WORKSPACE else AI_DIR / "workspace"
LOGS_DIR = AI_DIR / "logs" / "mechanic"

# Crear directorios
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de contenedores
CONTAINER_CONFIG = {
    "image": "nikolaik/python-nodejs:python3.12-nodejs22",
    "mem_limit": "512m",
    "memswap_limit": "512m",
    "cpu_period": 100000,
    "cpu_quota": 50000,  # 50% de 1 CPU
    "network_disabled": True,
    "auto_remove": True,
    "read_only": False,
    "security_opt": ["no-new-privileges"],
}

DEFAULT_TIMEOUT = 30  # segundos
MAX_TIMEOUT = 120  # máximo permitido

# Logger
logger = logging.getLogger("ultragent.mechanic")


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Resultado de una ejecución en sandbox."""
    
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    container_id: str
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "container_id": self.container_id,
            "error": self.error,
        }


class DockerNotAvailableError(Exception):
    """Docker no está disponible."""
    pass


class SandboxTimeoutError(Exception):
    """Timeout en la ejecución."""
    pass


class SandboxSecurityError(Exception):
    """Violación de seguridad."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANIC EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MechanicExecutor:
    """
    Ejecutor de código seguro en contenedores Docker.
    
    Proporciona aislamiento completo para ejecutar código
    sin riesgo para el sistema host.
    """
    
    def __init__(self):
        self._lock = Lock()
        self._client: Optional[docker.DockerClient] = None
        
        # Estadísticas
        self._stats = {
            "executions": 0,
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "total_time": 0.0,
        }
        
        # Intentar conectar a Docker
        try:
            self._client = docker.from_env()
            self._client.ping()
            self._is_available = True
            self.engine_mode = "HYBRID (Legacy + OpenHands)"
            logger.info(f"MechanicExecutor conectado a Docker (Modo: {self.engine_mode})")
        except Exception as e:
            logger.warning(f"Docker no disponible: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        """Indica si Docker está listo para usarse."""
        return self._is_available
    
    def _ensure_image(self, image: str = CONTAINER_CONFIG["image"]) -> bool:
        """Asegura que la imagen esté disponible localmente."""
        if not self._client:
            return False
        
        try:
            self._client.images.get(image)
            return True
        except ImageNotFound:
            logger.info(f"Descargando imagen: {image}")
            try:
                self._client.images.pull(image)
                return True
            except Exception as e:
                logger.error(f"Error descargando imagen: {e}")
                return False
    
    def _validate_script(self, script: str) -> None:
        """Valida que el script no contenga operaciones peligrosas."""
        dangerous_patterns = [
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "__import__('os')",
            "eval(",
            "exec(",
            "open('/etc",
            "open('/root",
            "shutil.rmtree",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in script:
                raise SandboxSecurityError(
                    f"Patrón peligroso detectado: {pattern}"
                )
    
    def _save_log(
        self,
        result: ExecutionResult,
        script: str,
        requirements: Optional[list[str]] = None,
    ) -> Path:
        """Guarda log de ejecución."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"exec_{timestamp}_{result.container_id[:8]}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "container_id": result.container_id,
            "result": result.to_dict(),
            "script_preview": script[:500],
            "requirements": requirements or [],
        }
        
        log_file.write_text(
            json.dumps(log_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        return log_file
    
    def run_in_sandbox(
        self,
        script: str,
        requirements: Optional[list[str]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        working_dir: Optional[str] = None,
        allow_network: bool = False,
    ) -> ExecutionResult:
        """
        Ejecuta un script Python en un contenedor aislado.
        
        Args:
            script: Código Python a ejecutar
            requirements: Lista de paquetes pip a instalar
            timeout: Tiempo máximo de ejecución en segundos
            working_dir: Directorio de trabajo (debe estar en .ai/workspace/)
            allow_network: Permitir acceso a red (para cloning/install)
            
        Returns:
            ExecutionResult con stdout, stderr, exit_code
        """
        if not self.is_available:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                execution_time=0,
                container_id="none",
                error="Docker no disponible",
            )
        
        # Validar timeout
        timeout = min(max(1, timeout), MAX_TIMEOUT)
        
        # Validar script
        try:
            self._validate_script(script)
        except SandboxSecurityError as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0,
                container_id="security_blocked",
                error=str(e),
            )
        
        # Asegurar imagen
        if not self._ensure_image():
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="No se pudo obtener la imagen Docker",
                execution_time=0,
                container_id="image_error",
                error="Image pull failed",
            )
        
        # Generar ID único
        container_id = f"ultragent_{uuid.uuid4().hex[:12]}"
        
        # Crear script wrapper
        setup_commands = []
        if requirements:
            pip_install = " ".join(requirements)
            setup_commands.append(f"pip install -q {pip_install}")
        
        wrapper_script = f"""
import sys
import traceback

# Setup
{''.join(f'import subprocess; subprocess.run(["pip", "install", "-q", "{req}"], check=True);' for req in (requirements or []))}

# User script
try:
{chr(10).join('    ' + line for line in script.split(chr(10)))}
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(wrapper_script)
            temp_script = f.name
        
        start_time = datetime.now()
        
        try:
            # Configurar volúmenes
            volumes = {}
            if working_dir and Path(working_dir).exists():
                # Validar que esté en workspace o OpenHands
                wd_path = Path(working_dir).resolve()
                ws_path = WORKSPACE_DIR.resolve()
                
                # Permitir si está en workspace definido
                if str(wd_path).startswith(str(ws_path)) or (OPENHANDS_WORKSPACE and str(wd_path).startswith(os.path.abspath(OPENHANDS_WORKSPACE))):
                    volumes[str(wd_path)] = {
                        "bind": "/workspace",
                        "mode": "rw",  # Changed to RW for OpenHands interaction
                    }
            
            # Montar script
            volumes[temp_script] = {
                "bind": "/script.py",
                "mode": "ro",
            }
            
            # Ejecutar contenedor
            container = self._client.containers.run(
                image=CONTAINER_CONFIG["image"],
                command=["python", "/script.py"],
                name=container_id,
                volumes=volumes,
                mem_limit=CONTAINER_CONFIG["mem_limit"],
                memswap_limit=CONTAINER_CONFIG["memswap_limit"],
                cpu_period=CONTAINER_CONFIG["cpu_period"],
                cpu_quota=CONTAINER_CONFIG["cpu_quota"],
                network_disabled=not allow_network,
                auto_remove=False,  # Necesitamos logs primero
                detach=True,
                environment={},  # Sin variables del host
            )
            
            # Esperar con timeout
            try:
                exit_result = container.wait(timeout=timeout)
                exit_code = exit_result.get("StatusCode", -1)
            except Exception:
                # Timeout - matar contenedor
                try:
                    container.kill()
                except Exception:
                    pass
                
                with self._lock:
                    self._stats["timeouts"] += 1
                
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timeout después de {timeout}s",
                    execution_time=timeout,
                    container_id=container_id,
                    error="Timeout",
                )
            
            # Obtener logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            
            # Limpiar contenedor
            try:
                container.remove(force=True)
            except Exception:
                pass
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Actualizar estadísticas
            with self._lock:
                self._stats["executions"] += 1
                self._stats["total_time"] += execution_time
                if exit_code == 0:
                    self._stats["successful"] += 1
                else:
                    self._stats["failed"] += 1
            
            result = ExecutionResult(
                success=(exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                container_id=container_id,
            )
            
            # Guardar log
            self._save_log(result, script, requirements)
            
            logger.info(
                f"Ejecución completada: {container_id[:8]} | "
                f"exit={exit_code} | time={execution_time:.2f}s"
            )
            
            return result
            
        except ContainerError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ExecutionResult(
                success=False,
                exit_code=e.exit_status,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                container_id=container_id,
                error=str(e),
            )
            
        except APIError as e:
            logger.error(f"Docker API error: {e}")
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0,
                container_id=container_id,
                error=f"Docker API error: {e}",
            )
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_script)
            except Exception:
                pass
    
    def run_script_file(
        self,
        script_path: str,
        requirements: Optional[list[str]] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ExecutionResult:
        """
        Ejecuta un archivo de script en sandbox.
        
        Args:
            script_path: Ruta al archivo .py
            requirements: Dependencias pip
            timeout: Timeout en segundos
            
        Returns:
            ExecutionResult
        """
        path = Path(script_path)
        if not path.exists():
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Archivo no encontrado: {script_path}",
                execution_time=0,
                container_id="file_not_found",
                error="File not found",
            )
        
        script = path.read_text(encoding="utf-8")
        return self.run_in_sandbox(
            script=script,
            requirements=requirements,
            timeout=timeout,
            working_dir=str(path.parent),
        )
    
    def clone_repo(self, repo_url: str, branch: Optional[str] = None) -> dict:
        """
        Clona un repositorio en un directorio efímero seguro.
        
        Usa un contenedor con acceso a red temporal para git clone.
        
        Args:
            repo_url: URL HTTPS del repositorio
            branch: Rama opcional
            
        Returns:
            dict: {success, local_path, error}
        """
        if not self.is_available:
            return {"success": False, "error": "Docker no disponible"}
            
        # Generar ID único para el clone path
        clone_id = uuid.uuid4().hex[:8]
        clone_dir = WORKSPACE_DIR / "clones" / clone_id
        clone_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Clonando {repo_url} en {clone_dir}...")
        
        # Script para clonar
        branch_arg = f"--branch {branch}" if branch else ""
        clone_script = f"""
import subprocess
import sys

try:
    cmd = ["git", "clone", "{repo_url}", ".", "--depth", "1", {branch_arg}]
    # Filtrar argumentos vacíos
    cmd = [c for c in cmd if c]
    
    print(f"Executing: {{' '.join(cmd)}}")
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
    print("Clone successful")
except Exception as e:
    print(f"Clone error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        # Ejecutar con red habilitada
        result = self.run_in_sandbox(
            script=clone_script,
            requirements=["gitpython"], # Asegurar que gitpython (o git cli) este disponible si usamos lib, pero aqui usamos subprocess git cli
            timeout=120,
            working_dir=str(clone_dir),
            allow_network=True
        )
        
        if result.success:
            return {
                "success": True,
                "local_path": str(clone_dir),
                "clone_id": clone_id
            }
        else:
            # Limpiar si falló
            import shutil
            try:
                shutil.rmtree(clone_dir)
            except:
                pass
            return {"success": False, "error": f"Clone failed: {result.stderr}"}

    def apply_external_pattern(self, code_block: str, target_file: str) -> dict:
        """
        Aplica un patrón de código externo (adaptado) al workspace.
        
        Args:
            code_block: Código Python a escribir
            target_file: Ruta relativa al workspace
            
        Returns:
            dict: Resultado de la operación
        """
        try:
            full_path = WORKSPACE_DIR / target_file
            
            # Asegurar que está en el workspace
            if not str(full_path.resolve()).startswith(str(WORKSPACE_DIR.resolve())):
                return {"success": False, "error": "Security: Target file outside workspace"}
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir directamente (asumiendo que Mechanic corre con permisos)
            # Si se requiere sudo/docker write, usar run_in_sandbox
            full_path.write_text(code_block, encoding="utf-8")
            
            return {
                "success": True, 
                "path": str(full_path),
                "bytes_written": len(code_block)
            }
        except Exception as e:
            logger.error(f"Failed to apply pattern: {e}")
            return {"success": False, "error": str(e)}

    async def run_agentic_session(
        self,
        task: str,
        workspace_path: str | None = None,
        agent_type: str = "CodeActAgent",
        max_iterations: int = 10,
    ) -> dict:
        """
        [NEW] Ejecuta una sesión agéntica completa usando el motor OpenHands.
        
        Capabilities:
        - Bucle de eventos (Event Stream)
        - Corrección de errores automática
        - Persistencia de contexto
        
        Args:
            task: La instrucción de alto nivel (ej: "Arregla los tests")
            workspace_path: Ruta al directorio de trabajo (default: .ai/workspace)
            agent_type: Tipo de agente OpenHands (default: CodeActAgent)
            max_iterations: Límite de pasos del agente
            
        Returns:
            dict: Estado final de la sesión
        """
        logger.info(f"Iniciando Sesión Agéntica (OpenHands): '{task}'")
        
        if not self.is_available:
            return {"success": False, "error": "Docker no disponible"}

        # Configuración de entorno Senior: Forzar detección de .NET y generar runtimeconfig estable
        dotnet_root = os.environ.get("DOTNET_ROOT")
        if not dotnet_root:
            potential_paths = [r"C:\Program Files\dotnet"]
            for p in potential_paths:
                if os.path.exists(p):
                    os.environ["DOTNET_ROOT"] = p
                    os.environ["PATH"] = f"{p};" + os.environ["PATH"]
                    dotnet_root = p
                    break

        # Generar runtimeconfig.json para forzar .NET 8 (LTS) y evitar conflictos con Previews (v10)
        config_path = AI_DIR / "ultragent.runtimeconfig.json"
        if not config_path.exists():
            config_data = {
                "runtimeOptions": {
                    "tfm": "net8.0",
                    "framework": {
                        "name": "Microsoft.NETCore.App",
                        "version": "8.0.0"
                    }
                }
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Generado {config_path} para estabilidad de .NET")

        try:
            # Inicialización manual de pythonnet antes de que OpenHands lo intente
            import clr_loader
            from pythonnet import load
            
            try:
                # Intentamos cargar el runtime estable específicamente
                runtime = clr_loader.get_coreclr(
                    runtime_config=str(config_path),
                    dotnet_root=dotnet_root
                )
                load(runtime)
                logger.info("Host de .NET (CoreCLR) inicializado exitosamente via net8.0 config")
            except Exception as e:
                logger.warning(f"Fallo en carga manual de .NET, reintentando carga estándar: {e}")
                # Si falla el manual, OpenHands intentará su propia carga (fallback)
            
            # Lazy imports para evitar dependencias duras
            from openhands.controller.agent_controller import AgentController
            from openhands.core.schema import AgentState
            from openhands.runtime.impl.docker.docker_runtime import DockerRuntime
            
            # Resolver workspace
            ws_path = Path(workspace_path) if workspace_path else WORKSPACE_DIR
            ws_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"OpenHands Workspace: {ws_path}")
            
            # Configuración Senior v1.2.0: Usar OpenHandsConfig para el Runtime
            from openhands.core.config import OpenHandsConfig
            from openhands.llm.llm_registry import LLMRegistry
            from openhands.events import EventStream
            from openhands.storage.local import LocalFileStore
            
            # 1. Preparar Almacenamiento, Configuración y Stream
            # Crear un subdirectorio para eventos dentro del workspace para evitar colisiones
            events_path = ws_path / "events"
            events_path.mkdir(parents=True, exist_ok=True)
            file_store = LocalFileStore(root=str(events_path))
            
            # WORKAROUND Senior: OpenHands v1.2.0 tiene un bug al buscar 'skills' en site-packages
            # Crearemos un directorio dummy para evitar WinError 3 durante la inicialización del runtime
            try:
                import openhands
                oh_path = Path(openhands.__file__).parent
                skills_dummy = oh_path.parent / "skills"
                if not skills_dummy.exists():
                    skills_dummy.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Creado directorio dummy de skills en: {skills_dummy}")
            except Exception as e:
                logger.warning(f"No se pudo crear el directorio dummy de skills: {e}")

            oh_config = OpenHandsConfig()
            # Forzar el uso de la imagen oficial estable para v1.2.0
            oh_config.sandbox.runtime_container_image = "ghcr.io/all-hands-ai/runtime:0.12-nikolaik"
            oh_config.sandbox.force_rebuild_runtime = False
            oh_config.sandbox.initialize_plugins = True
            oh_config.workspace_base = str(ws_path)
            oh_config.workspace_base = str(ws_path)
            
            # Inicializar Registro de LLM con Config
            llm_registry = LLMRegistry(config=oh_config)
            
            # EventStream en v1.2.0 requiere file_store
            event_stream = EventStream(sid="ultragent-session", file_store=file_store)
            
            # 2. Iniciar Runtime (El Cuerpo Dockerizado)
            runtime = DockerRuntime(
                config=oh_config,
                event_stream=event_stream,
                llm_registry=llm_registry,
            )
            await runtime.connect()
            
            # 3. Inicializar Agente (El Carácter)
            from openhands.controller.agent import Agent
            agent_cls = Agent.get_cls(agent_type)
            # El agente requiere su propia config y el registro de LLM
            agent_instance = agent_cls(config=oh_config.get_agent_config(agent_type), llm_registry=llm_registry)
            
            # 4. Inicializar Controlador (El Cerebro)
            from openhands.server.services.conversation_stats import ConversationStats
            conv_stats = ConversationStats()
            
            controller = AgentController(
                agent=agent_instance,
                event_stream=event_stream,
                conversation_stats=conv_stats,
                iteration_delta=max_iterations,
                sid="ultragent-session",
                file_store=file_store,
            )
            # El controlador se suscribe automáticamente al event_stream si no es un delegado
            
            # 3. Hook de Observabilidad (Conectar al HUD/Cortex)
            trajectory = []
            
            async def event_handler(event):
                # Capturar cada paso del agente
                if event.source == "agent":
                    # Acción del agente (Comando, Edición)
                    trajectory.append(f"🤖 ACT: {event.action}")
                    logger.info(f"[OH-Agent] {event.action}")
                elif event.source == "environment":
                    # Respuesta del entorno (Salida, Error)
                    # Truncar output largo
                    content = str(event.content)
                    preview = content[:100] + "..." if len(content) > 100 else content
                    trajectory.append(f"🌍 OBS: {preview}")
            
            controller.event_stream.subscribe(event_handler)
            
            # 4. Ejecutar Tarea
            logger.info("Delegando control al Agente...")
            end_state = await controller.start_task(task)
            
            # 5. Cerrar Runtime
            # await runtime.stop() # Mantener vivo? No, cerrar por ahora.
            
            success = (end_state.agent_state == AgentState.FINISHED)
            
            return {
                "success": success,
                "final_state": str(end_state.agent_state),
                "trajectory": trajectory,
                "metrics": {
                    "steps": len(trajectory),
                    "iterations_used": controller.state.iteration,
                }
            }
            
        except (ImportError, Exception) as e:
            # Capturamos ImportError y cualquier otro error de runtime (ej: DotNetMissingError)
            logger.error(f"OpenHands Engine no disponible: {e}")
            return {
                "success": False,
                "error": f"Agentic Mode Unavailable: {e}",
                "details": "Ensure Python 3.12+, .NET Runtime (Windows), and 'uv sync' are correct."
            }
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def heal_node(self, node_id: str, issue_description: str) -> dict:
        """
        [NEW] Intenta reparar un nodo usando Inteligencia Sistémica.
        Integración: Cortex (Memoria) + NeuroArchitect (Contexto) + OpenHands (Agente).
        """
        logger.info(f"Iniciando protocolo de curación para: {node_id}")
        
        try:
            # 1. Obtener Contexto Arquitectónico (NeuroArchitect)
            # Evitar import circular
            from neuro_architect import get_neuro_architect
            neuro = get_neuro_architect()
            
            # Usar 'bundle' para obtener todo lo necesario
            has_impact = neuro.analyze_impact(node_id)
            impact_info = f"Risk Score: {has_impact.risk_score:.1f}, Dependents: {len(has_impact.direct_impact)}"
            
            # 2. Consultar Memoria Episódica (Cortex)
            from cortex import get_cortex
            cortex = get_cortex()
            
            # Buscar si ya hemos arreglado algo similar
            memories = cortex.get_all_memories() # Idealmente search, pero por ahora filter simple o usar vector search si disponible
            # Usar Librarian search via Cortex si estuviera expuesto directo, pero simulamos
            relevant_memories = [m.content for m in memories if "fix" in m.content.lower() or "refactor" in m.content.lower()][:3]
            
            memory_context = "\n".join(relevant_memories) if relevant_memories else "No previous fixes found."

            # 3. Construir Prompt de Misión para el Agente
            prompt = f"""
MISSION: REFACTOR_CODE
TARGET: {node_id}
ISSUE: {issue_description}

CONTEXT:
{impact_info}

PREVIOUS KNOWLEDGE:
{memory_context}

INSTRUCTIONS:
1. Analyze the file {node_id}.
2. Fix the reported issue ensuring NO functionality is broken.
3. Run existing tests or create a small validation script.
4. If successful, exit with success.
"""

            # 4. Ejecutar Sesión Agéntica
            session_result = await self.run_agentic_session(
                task=prompt,
                max_iterations=10, # Keep it focused
                agent_type="CodeActAgent"
            )
            
            # 5. Memorizar Resultado (Cortex Loop)
            if session_result.get("success"):
                cortex.add_memory(
                    content=f"Fixed issue '{issue_description}' in {node_id} using automated agent.",
                    tags=["fix", "auto-healing", node_id],
                    importance=0.8
                )
                logger.info(f"Curación exitosa de {node_id}. Memorizada.")
            else:
                logger.warning(f"Fallo en curación de {node_id}.")
                
            return session_result

        except Exception as e:
            logger.error(f"Error en heal_node: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> dict:
        """Retorna estado del Mechanic."""
        return {
            "docker_available": self.is_available,
            "engine_mode": "HYBRID (Legacy + OpenHands)", # Actualizado
            "image": CONTAINER_CONFIG["image"],
            "limits": {
                "memory": CONTAINER_CONFIG["mem_limit"],
                "cpu_quota": f"{CONTAINER_CONFIG['cpu_quota']/CONTAINER_CONFIG['cpu_period']*100:.0f}%",
                "timeout_default": DEFAULT_TIMEOUT,
                "timeout_max": MAX_TIMEOUT,
            },
            "stats": dict(self._stats),
            "logs_dir": str(LOGS_DIR),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_mechanic_instance: Optional[MechanicExecutor] = None
_mechanic_lock = Lock()


def get_mechanic() -> MechanicExecutor:
    """Obtiene la instancia singleton del Mechanic."""
    global _mechanic_instance
    with _mechanic_lock:
        if _mechanic_instance is None:
            _mechanic_instance = MechanicExecutor()
        return _mechanic_instance


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
    print("ULTRAGENT MECHANIC v0.1 - Test")
    print("=" * 60)
    
    mechanic = get_mechanic()
    print(f"Status: {mechanic.get_status()}")
    
    if mechanic.is_available:
        print("\nEjecutando script de prueba...")
        result = mechanic.run_in_sandbox(
            script='print("Hello from sandbox!")\nprint(2 + 2)',
            timeout=30,
        )
        print(f"Success: {result.success}")
        print(f"STDOUT: {result.stdout}")
        print(f"Time: {result.execution_time:.2f}s")
    else:
        print("\n⚠️ Docker no disponible")
