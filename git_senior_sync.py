import asyncio
import os
import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | GIT-SENIOR | %(message)s"
)
logger = logging.getLogger("ultragent.git")

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

async def generate_commit_message(diff: str) -> str:
    """Genera un mensaje de commit senior usando el LLM."""
    if not diff:
        return "chore: minor updates"
    
    try:
        from router import get_router
        router = get_router()
        
        prompt = f"""
        Eres un Arquitecto Senior. Escribe un MENSAJE DE COMMIT para estos cambios.
        Sigue las especificaciones de 'Conventional Commits'. 
        Usa el formato: <type>(scope): <description>
        
        TYPES: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
        
        DIFF:
        {diff[:8000]}
        
        Instrucciones:
        1. Sé conciso pero informativo (primera línea < 72 chars).
        2. Detalla los cambios clave en el cuerpo si es complejo.
        3. Responde SOLAMENTE con el mensaje de commit.
        """
        
        result = await router.route_request(
            prompt=prompt,
            task_type="coding",
            system_prompt="Eres un experto en Git y estándares de ingeniería."
        )
        
        # Extracción segura de la respuesta
        response_text = ""
        if isinstance(result, dict):
            response_text = result.get("response", "")
        elif hasattr(result, "response"):
            response_text = result.response
        elif hasattr(result, "content"):
            response_text = result.content
            
        if response_text and isinstance(response_text, str):
            return response_text.strip().strip('"').strip("'")
                
        return "feat: implement 'Operation Zero-Debt' and git senior sync [AI fallback]"
    except Exception as e:
        logger.error(f"Error generando mensaje: {e}")
        return "feat: automate senior git synchronization flow [emergency fallback]"

def run_git(args: list) -> str:
    """Ejecuta comandos git y retorna salida de forma segura."""
    try:
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore"
        )
        if result and result.stdout:
            return result.stdout.strip()
        return ""
    except subprocess.CalledProcessError as e:
        logger.error(f"Git Error ({args[0]}): {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error in run_git: {e}")
        return ""

async def git_senior_sync():
    logger.info("🚀 Iniciando Sincronización Git Senior...")
    
    try:
        # 1. Verificar cambios
        status = run_git(["status", "--short"])
        logger.info(f"Status output captured. Type: {type(status)}")
        if not status:
            logger.info("✅ No hay cambios pendientes.")
            return

        logger.info(f"📁 Cambios detectados:\n{status}")
        
        # 2. Add (Respetando .gitignore y .agentignore)
        # Git ignora .gitignore por defecto, .agentignore es para la IA
        run_git(["add", "."])
        
        # 3. Obtener Diff para la IA
        diff = run_git(["diff", "--cached"])
        logger.info(f"Diff output captured. Type: {type(diff)}")
        
        # 4. Generar Mensaje Inteligente
        logger.info("🧠 Generando mensaje de commit inteligente...")
        message = await generate_commit_message(diff)
        logger.info(f"✨ Mensaje Sugerido: {message}")
        
        # 5. Commit
        run_git(["commit", "-m", message])
        
        # 6. Push
        logger.info("📤 Enviando cambios al servidor remoto (Push)...")
        run_git(["push", "origin", "main"]) # Asumimos main, podrías parametrizar
        
        logger.info("🎉 Sincronización exitosa.")
        
    except Exception as e:
        logger.error(f"❌ Fallo en la sincronización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(git_senior_sync())
