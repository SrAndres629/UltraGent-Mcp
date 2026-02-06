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

    def edit_file(self, path: str, target: str, replacement: str) -> dict:
        """
        Edición quirúrgica de archivos (Search & Replace).
        Reemplaza la PRIMERA ocurrencia de 'target' con 'replacement'.
        """
        try:
            full_path = PROJECT_ROOT / path
            if not full_path.exists():
                return {"success": False, "error": f"File {path} not found"}
            
            content = full_path.read_text(encoding="utf-8")
            
            if target not in content:
                # Intento de relajación de whitespace (normalizar)
                return {"success": False, "error": "Target snippet not found in file"}
                
            # Verificar unicidad para seguridad (opcional, por ahora confiamos en el agente)
            if content.count(target) > 1:
                logger.warning(f"Target found multiple times in {path}, replacing first occurrence only.")
                
            new_content = content.replace(target, replacement, 1)
            full_path.write_text(new_content, encoding="utf-8")
            
            logger.info(f"Surgical edit applied to {path}")
            return {"success": True, "message": "Patch applied successfully"}
            
        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return {"success": False, "error": str(e)}

    def append_to_file(self, path: str, content: str) -> dict:
        """Añade contenido al final del archivo."""
        try:
            full_path = PROJECT_ROOT / path
            with open(full_path, "a", encoding="utf-8") as f:
                f.write("\n" + content)
            return {"success": True, "message": "Content appended"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

class LLMProvider:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("No GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    async def generate(self, prompt: str) -> str:
        import asyncio
        for attempt in range(3):
            try:
                # generate_content is blocking, ideally use generate_content_async
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Gemini Rate Limit (429). Retrying in 5s... ({attempt+1}/3)")
                    await asyncio.sleep(5)
                else:
                    raise e
        raise RuntimeError("Gemini 429: Rate Limit Exceeded after retries.")

class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    async def generate(self, prompt: str) -> str:
        # Note: In a production async environment, use AsyncOpenAI
        # For now, we wrap in thread to avoid blocking loop if using sync client
        import asyncio
        loop = asyncio.get_event_loop()
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
        response = await loop.run_in_executor(None, _call)
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

    @property
    def is_available(self) -> bool:
        """Determina si el agente está listo (tiene proveedores)."""
        return len(self.providers) > 0

    def get_status(self) -> dict:
        """Retorna estado del Mechanic."""
        return {
            "is_available": self.is_available,
            "providers": [type(p).__name__ for p in self.providers],
            "active_provider": type(self.providers[0]).__name__ if self.providers else None,
            "tool_count": len(self.tools)
        }

    def run_in_sandbox(self, script: str, requirements: list[str] = None, timeout: int = 30) -> dict:
        """
        Simula un sandbox ejecutando el código nativamente (Warning: No Docker).
        """
        logger.warning("Ejecución en Sandbox solicitada pero Docker no detectado. Usando Native Bypass.")
        # Escribir código temporal
        temp_file = "temp_sandbox_script.py"
        self.executor.write_file(temp_file, script)
        
        # Instalar requisitos si hay
        if requirements:
            for req in requirements:
                self.executor.run_command(f"pip install {req}")
        
        # Ejecutar
        result = self.executor.run_command(f"python {temp_file}", timeout=timeout)
        
        # Limpiar
        try: os.remove(temp_file)
        except: pass
        
        return {
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
            "is_sandbox": False,
            "warning": "Ejecutado nativamente sin aislamiento Docker"
        }

    async def run_task(self, task: str, max_steps: int = 10) -> str:
        logger.info(f"🚀 Iniciando Misión (Senior Contextual Loop): {task}")
        
        # 1. Recuperar contexto de memoria (Active Recall)
        from cortex import get_cortex
        memories = get_cortex().get_related_memories(task)
        context_block = "\n".join([f"- [MEMORIA] {m.content} (Tag: {m.tags})" for m in memories[:5]])
        
        # 2. FASE DE CONSULTA ESTRATÉGICA (System 2 Support)
        # Antigravity (Local) decide consultar, pero mantiene el control.
        logger.info("🤔 System 2: Consulting External Oracle (Context-Isolated)...")
        from router import get_router
        try:
            # Ingeniería de Prompt Contextual:
            strategic_payload = f"""
            CONTEXTO PROVISTO (No asumas nada más):
            Task: {task}
            Memories/Learnings:
            {context_block}
            
            PETICIÓN:
            Actúa como un Investigador de Arquitectura de Software (Visionary Level).
            Tu objetivo no es solo "funcionalidad", sino Excelencia, Diseño Extraordinario y Solidez a largo plazo.
            
            Basado SOLO en el contexto arriba:
            1. Analiza peligros ocultos (Second-Order Effects).
            2. Propón una estrategia impecable (Novelty + Reliability).
            3. Si ves mediocridad en el enfoque, critícalo constructivamente.
            """
            
            plan_response = await get_router().route_task(
                task_type="strategic",
                payload=strategic_payload,
                system_prompt="Eres un Visionario de Silicon Valley (Paul Graham style). Buscas lo extraordinario. NO tienes acceso al repo, solo al contexto."
            )
            strategic_advice = plan_response.content
        except Exception as e:
            strategic_advice = f"Consultation failed: {e}"
        
        logger.info(f"📋 Strategic Advice:\n{strategic_advice[:200]}...")

        history = [
            f"USER TASK: {task}", 
            f"CORTEX MEMORY:\n{context_block}",
            f"EXTERNAL CONSULTANT ADVICE:\n{strategic_advice}"
        ]
        
        # 3. CARGAR PROTOCOLOS DE HABILIDADES (SKILLS INJECTION)
        # Esto permite que archivos .md definan la personalidad del agente.
        protocol_path = PROJECT_ROOT / "skills" / "deep_engineering_protocol.md"
        if protocol_path.exists():
            deep_protocol = protocol_path.read_text(encoding="utf-8")
            logger.info("🧬 Deep Engineering Protocol loaded.")
        else:
            deep_protocol = "Protocol not found. Use basic discretion."

        system_instruction = f"""
        Eres Ultragent Mechanic (ORCHESTRATOR).
        
        === DEEP ENGINEERING PROTOCOL (THE CONSTITUTION) ===
        {deep_protocol}
        
        === OPERATIONAL CONTEXT ===
        TU ROL:
        - Tú tienes el CONTEXTO COMPLETO (Archivos, Estado).
        - El Consultor Externo (Advice) es sabio pero CIEGO al repo.
        - TÚ tomas las decisiones. Usa el Advice como guía, no como dogma.
        
        PROTOCOLO DE MANTENIMIENTO COGNITIVO (CRÍTICO):
        El código cambia, y tu memoria debe cambiar con él. NO seas perezoso.
        1. Si modificas código (edit/write) -> DEBES re-indexar semánticamente (librarian.index).
        2. Si cambias estructura -> DEBES regenerar el grafo (visualize_architecture).
        3. Si aprendes algo nuevo -> DEBES guardarlo en memoria (cortex.add).
        *El mapa (Memoria) debe coincidir con el territorio (Código).*
        
        PROTOCOLO SENIOR:
        1. Analiza la tarea + Advice.
        2. Si necesitas resolver una duda técnica específica (sintaxis, error oscuro), usa 'consult_oracle'.
        3. SIEMPRE le das contexto a tus herramientas.
        
        Tools: 
        - run_command(command)
        - read_file(path)
        - write_file(path, content)
        - edit_file(path, target, replacement)
        - consult_oracle(query) -> Cloud AI (Recuerda incluir contexto en la query)
        - research(query) -> Internet Search
        - manage_memory(action='add|list', content, tags) -> Update Cortex
        - update_map(action='index|render', target) -> Update Vectors/Graph
        - finish(result)
        
        Responde SOLO en JSON válido.
        """
        
        final_result = "Failed"
        
        for step in range(max_steps):
            prompt = f"{system_instruction}\n\nHistory: {history[-3:]}\n\nTask: {task}\n\nAction (JSON):"
            
            # Retry loop for providers
            text = None
            while text is None:
                try:
                    provider = self._get_working_provider()
                    text = await provider.generate(prompt)
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
                    final_result = action.get("args", {}).get("result", "Done")
                    break
                
                if tool_name == "consult_oracle":
                    # Tool especial para consultar al router/internet
                    query = action.get("args", {}).get("query")
                    logger.info(f"🔮 Consulting Oracle: {query}")
                    try:
                        resp = await get_router().route_task("strategic", query)
                        result = resp.content
                    except Exception as e:
                        result = str(e)
                    obs = f"Oracle Answer: {result}"
                    history.append(f"Action: consult_oracle({query})\n{obs}")
                
                elif tool_name == "research":
                    # Tool para búsqueda real en internet (Scout)
                    query = action.get("args", {}).get("query")
                    logger.info(f"🌍 Scouting: {query}")
                    try:
                        from scout import get_scout
                        scout = get_scout()
                        # Usamos proactive_research o universal_search
                        # proactive_research usa router + ddg, es mejor
                        from evolution import get_evolution # evolution tiene proactive_research
                        res = await get_evolution().proactive_research(query)
                        result = res["result"].content if res["success"] else res["error"]
                    except Exception as e:
                        result = str(e)
                    obs = f"Research Result: {result[:500]}..."
                    history.append(f"Action: research({query})\n{obs}")

                elif tool_name == "manage_memory":
                    # Tool para actualización explícita de Cortex
                    act = action.get("args", {}).get("action", "add")
                    content = action.get("args", {}).get("content", "")
                    tags = action.get("args", {}).get("tags", [])
                    logger.info(f"🧠 Managing Memory: {act}")
                    try:
                        from cortex import get_cortex
                        ctx = get_cortex()
                        if act == "add":
                            mid = ctx.add_memory(content, tags)
                            result = f"Memory added (ID: {mid})"
                        elif act == "list":
                            mems = ctx.get_all_memories()
                            result = str([m.content[:50] for m in mems[:5]])
                        else:
                            result = "Action not supported"
                    except Exception as e:
                        result = str(e)
                    obs = f"Memory Result: {result}"
                    history.append(f"Action: manage_memory({act})\n{obs}")

                elif tool_name == "update_map":
                    # Tool para mantenimiento cognitivo (Vectores/Grafo)
                    act = action.get("args", {}).get("action", "index")
                    target = action.get("args", {}).get("target", "")
                    logger.info(f"🗺️ Updating Map: {act} on {target}")
                    try:
                        if act == "index":
                            from librarian import get_librarian
                            lib = get_librarian()
                            lib.index_file(target)
                            result = f"File {target} re-indexed successfully."
                        elif act == "render":
                            from vision import get_vision
                            vis = get_vision()
                            rep = vis.generate_dependency_graph()
                            result = f"Architecture graph regenerated at {rep.graph_path}"
                        else:
                            result = "Action not supported"
                    except Exception as e:
                        result = str(e)
                    obs = f"Map Update Result: {result}"
                    history.append(f"Action: update_map({act})\n{obs}")

                elif tool_name in self.tools:
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
        
        # 2. Guardar aprendizaje (Active Learning)
        try:
            get_cortex().add_memory(
                content=f"Task: {task} | Plan: {strategic_plan[:50]}... | Result: {final_result[:100]}...",
                tags=["mechanic_log", "senior_protocol", "success" if "Done" in final_result else "failure"],
                importance=0.7
            )
            logger.info("🧠 Cortex updated with Senior execution context.")
        except Exception as e:
            logger.error(f"Failed to update Cortex: {e}")
                
        return final_result

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
