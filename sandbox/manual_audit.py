
import asyncio
import sys
import os
import json
from datetime import datetime

# Setup paths
sys.path.append(os.getcwd())

from evolution import get_evolution
from scout import get_scout

async def run_audit():
    print("=== MANUAL MISSION START: MEMORA AUDIT ===")
    
    # 1. Initialize logic
    print("[1/3] Initializing Evolution & Scout...")
    evo = get_evolution()
    
    task = "Analizar repositorio Memora (https://github.com/agentic-mcp-tools/memora) enfocándose en su lógica de persistencia y knowledge graph"
    
    # 2. Execute Research
    print(f"[2/3] Executing Proactive Research for: {task}")
    try:
        # This calls the method I just patched with Smart GitHub Mode
        report = await evo.proactive_research(task)
        
        print("\n=== RESEARCH REPORT ===")
        print(f"Query: {report.query}")
        print(f"Recommendation: {report.recommendation}")
        print(f"References Found: {len(report.references)}")
        
        for i, ref in enumerate(report.references):
            print(f"\n--- Reference #{i+1} ---")
            print(f"Title: {ref.title}")
            print(f"URL: {ref.url}")
            print(f"Source: {ref.source_type}")
            print(f"Snippet Preview: {ref.snippet[:200]}...")
            
            # Save snippet to file for analysis if needed
            with open(f"memora_ref_{i}.txt", "w", encoding="utf-8") as f:
                f.write(ref.snippet)
                
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_audit())
