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
WORKSPACE_DIR = AI_DIR / "workspace"
LOGS_DIR = AI_DIR / "logs" / "mechanic"

# Crear directorios
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de contenedores
CONTAINER_CONFIG = {
    "image": "python:3.12-slim",
    "mem_limit": "512m",
    "memswap_limit": "512m",
    "cpu_period": 100000,
    "cpu_quota": 50000,  # 50% de 1 CPU
    "network_disabled": True,
    "auto_remove": True,
    "read_only": False,  # Necesario para pip install
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
            logger.info("MechanicExecutor conectado a Docker")
        except Exception as e:
            logger.warning(f"Docker no disponible: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        """Verifica si Docker está disponible."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False
    
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
    ) -> ExecutionResult:
        """
        Ejecuta un script Python en un contenedor aislado.
        
        Args:
            script: Código Python a ejecutar
            requirements: Lista de paquetes pip a instalar
            timeout: Tiempo máximo de ejecución en segundos
            working_dir: Directorio de trabajo (debe estar en .ai/workspace/)
            
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
                # Validar que esté en workspace
                wd_path = Path(working_dir).resolve()
                ws_path = WORKSPACE_DIR.resolve()
                if str(wd_path).startswith(str(ws_path)):
                    volumes[str(wd_path)] = {
                        "bind": "/workspace",
                        "mode": "ro",
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
                network_disabled=CONTAINER_CONFIG["network_disabled"],
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
    
    def get_status(self) -> dict:
        """Retorna estado del Mechanic."""
        return {
            "docker_available": self.is_available,
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
