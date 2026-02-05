"""
ULTRAGENT MECHANIC v2.1 (Native Multi-Provider)
===============================================
Motor de ejecución nativa con redundancia de LLMs.

Providers soportados:
1. Google Gemini (1M Context) - Priority 1
2. SiliconFlow (DeepSeek V3/R1) - Priority 2
3. Groq (Llama 3 70B) - Priority 3

Fallback automático si una clave falla.
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Configuración de Logging
logger = logging.getLogger("ultragent.mechanic")

# Cargar entorno
PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LocalExecutor:
    def run_command(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """Ejecuta un comando en shell con timeout."""
        try:
            logger.info(f"Exec: {command}")
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT)
            )
            return {
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "success": process.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Timeout expired", "success": False}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e), "success": False}

    def write_file(self, path: str, content: str) -> bool:
        """Escribe un archivo en disco."""
        try:
            full_path = PROJECT_ROOT / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"Wrote file: {path}")
            return True
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return False

    def read_file(self, path: str) -> str:
        """Lee un archivo del disco."""
        try:
            full_path = PROJECT_ROOT / path
            if not full_path.exists():
                return f"Error: File {path} not found."
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file {path}: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("No GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Gemini Rate Limit (429). Retrying in 5s... ({attempt+1}/3)")
                    time.sleep(5)
                else:
                    raise e
        raise RuntimeError("Gemini 429: Rate Limit Exceeded after retries.")

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

# ═══════════════════════════════════════════════════════════════════════════════
# NATIVE AGENT WITH FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

class NativeAgent:
    def __init__(self):
        self.executor = LocalExecutor()
        self.tools = {
            "run_command": self.executor.run_command,
            "read_file": self.executor.read_file,
            "write_file": self.executor.write_file,
        }
        # Initialize all providers in order
        self.providers = []
        self._init_providers()

    def _init_providers(self):
        """Inicializa proveedores disponibles (Swarm Redundancy)."""
        # 1. Gemini (Primary - Brain)
        try:
             import google.generativeai as genai
             key = os.getenv("GEMINI_API_KEY")
             if key:
                 self.providers.append(GeminiProvider())
                 logger.info("Provider added: Gemini (Primary)")
        except Exception as e: logger.warning(f"Skip Gemini: {e}")

        # 2. Groq (Secondary - Velocity Backup)
        try:
            key = os.getenv("GROQ_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key, 
                    base_url="https://api.groq.com/openai/v1",
                    model="llama-3.3-70b-versatile"
                ))
                logger.info("Provider added: Groq")
        except Exception as e: logger.warning(f"Skip Groq: {e}")

        # 3. SiliconFlow (Tertiary - DeepSeek Logic)
        try:
            key = os.getenv("SILICONFLOW_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key, 
                    base_url="https://api.siliconflow.cn/v1",
                    model="deepseek-ai/DeepSeek-V3"
                ))
                logger.info("Provider added: SiliconFlow")
        except Exception as e: logger.warning(f"Skip SiliconFlow: {e}")

        # 4. Cerebras (Quaternary - Instant Llama 3.3)
        try:
            key = os.getenv("CEREBRAS_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key,
                    base_url="https://api.cerebras.ai/v1",
                    model="llama-3.3-70b"
                ))
                logger.info("Provider added: Cerebras")
        except Exception as e: logger.warning(f"Skip Cerebras: {e}")

        # 5. SambaNova (Quintary - Massive Context)
        try:
            key = os.getenv("SAMBANOVA_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key,
                    base_url="https://api.sambanova.ai/v1",
                    model="Meta-Llama-3.1-70B-Instruct"
                ))
                logger.info("Provider added: SambaNova")
        except Exception as e: logger.warning(f"Skip SambaNova: {e}")

        # 6. NVIDIA NIM (Sextary - Specialized)
        try:
            key = os.getenv("NVIDIA_NIM_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model="meta/llama-3.1-70b-instruct"
                ))
                logger.info("Provider added: NVIDIA NIM")
        except Exception as e: logger.warning(f"Skip NVIDIA: {e}")

        # 7. OpenRouter (Last Resort - Aggregator)
        try:
            key = os.getenv("OPENROUTER_API_KEY")
            if key:
                self.providers.append(OpenAICompatibleProvider(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                    model="google/gemini-2.0-flash-001" # Uses free tier on OR
                ))
                logger.info("Provider added: OpenRouter")
        except Exception as e: logger.warning(f"Skip OpenRouter: {e}")

        if not self.providers:
             raise RuntimeError("No working LLM providers found.")

    def _get_working_provider(self):
         """Retorna el primer proveedor funcional (o el siguiente si falla)."""
         if not self.providers:
             raise RuntimeError("All providers failed.")
         return self.providers[0]

    def _rotate_provider(self):
        """Rota al siguiente proveedor en caso de fallo."""
        if self.providers:
             failed = self.providers.pop(0)
             logger.warning(f"Rotating provider: {type(failed).__name__} failed.")
        
        if not self.providers:
             raise RuntimeError("All providers exhausted.")
        
        logger.info(f"New provider active: {type(self.providers[0]).__name__}")

    async def run_task(self, task: str, max_steps: int = 10) -> str:
        logger.info(f"🚀 Iniciando Misión (Native Loop): {task}")
        history = [f"USER TASK: {task}"]
        
        system_instruction = """
        Eres Ultragent Mechanic. Responde SOLO en JSON válido.
        Formato: {"tool": "name", "args": {...}} o {"tool": "finish", "args": {"result": "..."}}
        Tools: run_command(command), read_file(path), write_file(path, content).
        """
        
        for step in range(max_steps):
            prompt = f"{system_instruction}\n\nHistory: {history[-3:]}\n\nTask: {task}\n\nAction (JSON):"
            
            # Retry loop for providers
            text = None
            while text is None:
                try:
                    provider = self._get_working_provider()
                    text = provider.generate(prompt)
                except Exception as e:
                    logger.error(f"Provider Error: {e}")
                    try:
                        self._rotate_provider()
                    except RuntimeError:
                         return "Fatal: All LLMs failed."
            
            # --- Rest of logic same as before ---
            logger.info(f"🤖 Agent ({type(provider).__name__}): {text[:100]}...")
            
            try:
                # Parse
                clean_text = text.replace("```json", "").replace("```", "").strip()
                if "{" not in clean_text: 
                     history.append(f"Error: Output was not JSON. Model said: {clean_text}")
                     continue
                     
                action = json.loads(clean_text[clean_text.find("{"):clean_text.rfind("}")+1])
                
                tool_name = action.get("tool")
                
                if tool_name == "finish":
                    return action.get("args", {}).get("result", "Done")
                
                if tool_name in self.tools:
                    func = self.tools[tool_name]
                    args = action.get("args", {})
                    logger.info(f"🛠️ Tool: {tool_name} {args}")
                    
                    result = func(**args)
                    obs = f"Observation: {result}"
                    history.append(f"Action: {tool_name}({args})\n{obs}")
                else:
                    history.append(f"Error: Tool {tool_name} not found.")
            
            except Exception as e:
                history.append(f"Error executing step: {e}")
                logger.error(f"Step Error: {e}")
                
        return "Max steps reached."

# Singleton
_mechanic_instance = None
def get_mechanic():
    global _mechanic_instance
    if _mechanic_instance is None:
        try:
             _mechanic_instance = NativeAgent()
        except Exception as e:
             logger.error(f"FATAL: No se pudo levantar NativeAgent: {e}")
             return None
    return _mechanic_instance
