"""
ULTRAGENT TESTER v0.1 (Dynamic Verified)
========================================
Motor de Generación y Ejecución de Tests (Auto-QA).
Escanea código, genera tests faltantes y valida lógica.

Uso: python tester.py --scan
"""

import sys
import pytest
import asyncio
import logging
from pathlib import Path
from typing import List, Dict

# Importar Evolution para generar tests
from evolution import get_evolution, AuditReport

PROJECT_ROOT = Path.cwd()
TEST_DIR = PROJECT_ROOT / "tests"
LOG_DIR = PROJECT_ROOT / "logs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltTester")

class LogicTester:
    """QA Engineer Agent."""
    
    def __init__(self):
        TEST_DIR.mkdir(exist_ok=True)
        LOG_DIR.mkdir(exist_ok=True)
        self.evolution = get_evolution()

    async def scan_and_generate(self):
        """Escanea el proyecto y genera tests para archivos críticos."""
        logger.info(f"🕵️ Scanning for Logic Files in {PROJECT_ROOT}...")
        
        candidates = list(PROJECT_ROOT.glob("*.py")) # Only root files for speed
        logger.info(f"Found {len(candidates)} candidates.")
        
        for file_path in candidates:
            # Filtros
            if "test" in file_path.name or "venv" in str(file_path) or "setup" in file_path.name:
                continue
            if file_path.stat().st_size < 300: # Archivos muy pequeños
                continue
                
            test_file = TEST_DIR / f"test_{file_path.name}"
            
            logger.info(f"Checking {file_path.name} -> {test_file.name}")
            if not test_file.exists():
                logger.info(f"⚡ Missing test. Generating...")
                await self._generate_test_suite(file_path, test_file)
            else:
                logger.info(f"OK: {test_file.name} exists.")

    async def _generate_test_suite(self, source_file: Path, target_test_file: Path):
        """Usa Evolution LLM para escribir un test suite robusto."""
        code_content = source_file.read_text(encoding="utf-8", errors="ignore")
        
        prompt = f"""
        ACT AS: Senior QA Automation Engineer (Pytest Expert).
        MESSAGE: I need a Robust Pytest Suite for the following file.
        
        CONTEXT:
        Project Root: {PROJECT_ROOT.name}
        File to test: {source_file.name}
        Current Module: {source_file.stem}
        
        CODE:
        ```python
        {code_content[:8000]}
        ```
        
        INSTRUCTIONS:
        1. IMPORT TRUTH: All files are in the same directory. Use 'from {source_file.stem} import ...'
        2. ASYNC SUPPORT: Use '@pytest.mark.asyncio' if the functions are async.
        3. MOCKING: Mock external IO/Networking using 'unittest.mock.patch'.
        4. COVERAGE: Test success paths, edge cases, and expected errors.
        5. OUTPUT: RETURN ONLY THE PYTHON CODE. NO MARKDOWN BLOCKS.
        6. NO PLACEHOLDERS: Do not use 'your_module' or '...' in imports.
        """
        
        # We misuse the router via Evolution for this generation task
        # Ideally we should have a dedicated method, but this works.
        try:
            # Hack: Call proactive_research or similar generic endpoint?
            # Better: use directly the mechanic router logic but simpler
            # For now, let's assume we can trigger a generation.
            # We will simulate it via the 'fix' capability logic or router directly.
            from router import get_router
            router = get_router()
            response = await router.route_task("generate_code", prompt)
            
            if response.success:
                code = response.content.replace("```python", "").replace("```", "").strip()
                target_test_file.write_text(code, encoding="utf-8")
                logger.info(f"✅ Generated {target_test_file.name}")
            else:
                logger.error(f"Failed to generate test: {response.content}")
                
        except Exception as e:
            logger.error(f"Generation Error: {e}")

    async def verify_file(self, file_path: Path) -> bool:
        """
        Verifica un archivo específico: escanea, genera si falta, y corre tests locales.
        
        Returns:
            bool: True si pasa los tests, False si falla.
        """
        if not file_path.exists():
            return False
            
        test_file = TEST_DIR / f"test_{file_path.name}"
        
        if not test_file.exists():
            logger.info(f"⚡ Autogenerating test for verification: {file_path.name}")
            await self._generate_test_suite(file_path, test_file)
            
        if not test_file.exists():
            return False # Failed to generate
            
        logger.info(f"🧪 Verifying {file_path.name} via {test_file.name}")
        
        # Run only this test file
        ret_code = pytest.main([
            str(test_file),
            "-v",
            "-p", "no:warnings"
        ])
        
        return ret_code == 0

    def run_suite(self):
        """Ejecuta la suite completa y guarda reporte."""
        logger.info("🚀 Running Full Test Suite...")
        
        report_file = LOG_DIR / "deploy_verification.html"
        
        # Run pytest programmatically
        ret_code = pytest.main([
            str(TEST_DIR),
            "-v",
            f"--html={report_file}",
            "--self-contained-html"
        ])
        
        if ret_code == 0:
            logger.info("🟢 ALL SYSTEMS GO. Ready for Deploy.")
        else:
            logger.error(f"🔴 TESTS FAILED (Exit Code: {ret_code}). Check report.")

# SINGLETON
_tester_instance = None
def get_tester():
    global _tester_instance
    if _tester_instance is None:
        _tester_instance = LogicTester()
    return _tester_instance


if __name__ == "__main__":
    tester = get_tester()
    
    if "--scan" in sys.argv:
        asyncio.run(tester.scan_and_generate())
        
    tester.run_suite()
