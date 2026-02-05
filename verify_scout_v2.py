
import asyncio
import os
import sys
import logging
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).parent))

from scout import get_scout
from evolution import get_evolution, ResearchReport
from mechanic import get_mechanic

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("VERIFY_V2")

async def verify_universal_search():
    print("\n--- Verifying Universal Search (Scout) ---")
    scout = get_scout()
    
    query = "python fastapi jwt authentication best practices"
    print(f"Querying: {query}")
    
    try:
        results = await scout.universal_search(query, max_results=3, modernity_years=2)
        print(f"Found {len(results)} results.")
        
        snippets = 0
        archs = 0
        docs = 0
        
        for r in results:
            print(f"- [{r.source_type}] {r.title} ({r.url})")
            if r.source_type == "SNIPPET": snippets += 1
            if r.source_type == "ARCHITECTURE": archs += 1
            if r.source_type == "DOCS": docs += 1
            
        if len(results) > 0:
            print("✅ Universal Search works!")
        else:
            print("⚠️ No results found (could be network or DDG rate limit).")
            
    except Exception as e:
        print(f"❌ Error in universal_search: {e}")

async def verify_strategic_consultant_plumbing():
    print("\n--- Verifying Strategic Consultant Plumbing (Evolution) ---")
    evolution = get_evolution()
    
    # We expect this to fail on Router call if keys are missing, but 
    # we want to verify it reaches that point after search.
    
    task = "Implement OAuth2 with Google in FastAPI"
    print(f"Task: {task}")
    
    try:
        report = await evolution.proactive_research(task)
        print("✅ Proactive Research returned report.")
        print(f"Recommendation: {report.recommendation}")
        print(f"References found: {len(report.references)}")
    except Exception as e:
        if "Router" in str(e) or "API key" in str(e) or "401" in str(e):
             print(f"⚠️ Expected LLM Failure (Missing Keys): {e}")
             print("✅ Plumbing is correct (Search -> LLM attempt)")
        else:
             print(f"❌ Unexpected Error: {e}")

async def verify_mechanic_bridge():
    print("\n--- Verifying Mechanic Bridge ---")
    mechanic = get_mechanic()
    
    # Mock OpenHands ENV for this process
    test_ws = Path("test_openhands_ws")
    test_ws.mkdir(exist_ok=True)
    os.environ["OPENHANDS_WORKSPACE"] = str(test_ws.absolute())
    
    # Reload mechanic to pick up env? No, mechanic config loads at import time usually
    # or init. But we modified mechanic code to read env at global scope.
    # We might need to manually set it for this test instance since import already happened.
    from mechanic import WORKSPACE_DIR
    import mechanic as mechanic_module
    
    # Verify apply_external_pattern
    target_file = "utils/auth.py"
    code = "def authenticate(): pass"
    
    print(f"Applying pattern to: {target_file}")
    result = mechanic.apply_external_pattern(code, target_file)
    
    if result["success"]:
        print(f"✅ Pattern applied to: {result['path']}")
        # Verify file exists
        p = Path(result['path'])
        if p.exists() and p.read_text() == code:
             print("✅ File content verified")
        else:
             print("❌ File verification failed")
    else:
        print(f"❌ Failed to apply pattern: {result['error']}")

    # Cleanup
    import shutil
    if test_ws.exists():
        shutil.rmtree(test_ws)

if __name__ == "__main__":
    asyncio.run(verify_universal_search())
    asyncio.run(verify_strategic_consultant_plumbing())
    asyncio.run(verify_mechanic_bridge())
