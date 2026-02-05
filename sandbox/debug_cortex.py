
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.getcwd())
load_dotenv(".env")

from cortex import get_cortex

def test_cortex():
    print("Initializing Cortex...")
    c = get_cortex()
    
    print("\nAdding test memory...")
    mid = c.add_memory(
        content="Ultragent v0.1 has been successfully stabilized and is now being upgraded to v2.0 with Cortex integration.",
        tags=["system", "upgrade", "milestone"],
        importance=0.9
    )
    print(f"Memory created with ID: {mid}")
    
    print("\nRetrieving all memories...")
    memories = c.get_all_memories()
    for m in memories:
        print(f"[{m.id}] ({m.created_at}) {m.content}")
        print(f"    Tags: {m.tags}, Importance: {m.importance}")

if __name__ == "__main__":
    test_cortex()
