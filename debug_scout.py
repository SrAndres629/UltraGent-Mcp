
import asyncio
import sys
import os
import traceback
sys.path.append(os.getcwd())

from scout import get_scout

async def test_search():
    print("Initializing Scout...")
    try:
        scout = get_scout()
        query = "Analizar repositorio Memora (https://github.com/agentic-mcp-tools/memora) enfocándose en su lógica de persistencia y knowledge graph"
        print(f"Running Universal Search for: '{query[:50]}...'")
        
        start_time = asyncio.get_event_loop().time()
        results = await scout.universal_search(query, max_results=3)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Search completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(results)} results")
        for r in results:
            print(f"- {r.title} ({r.url})")
            
    except Exception as e:
        print(f"ERROR CRITICAL: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search())
