
import asyncio
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.getcwd())
load_dotenv(".env")

from vision import get_vision
from cortex import get_cortex

def verify_vision_v2():
    logging.basicConfig(level=logging.INFO)
    print("=== VISION 2.0 VERIFICATION ===")
    
    # 1. Asegurar que hay memorias relevantes para ver vínculos
    c = get_cortex()
    print("Adding relevant memories for linking...")
    c.add_memory("librarian.py usage: This file handles semantic search and vector indexing.", ["librarian"])
    c.add_memory("vision.py upgrade: Now supports Knowledge Graph with purple memory nodes.", ["vision"])
    
    # 2. Generar Grafo
    v = get_vision()
    print("\nScanning project and generating graph...")
    report = v.generate_dependency_graph(output_path="vision_v2_test.png")
    
    print("\n=== RESULTS ===")
    print(f"Nodes: {len(report.nodes)}")
    print(f"Edges: {len(report.edges)}")
    print(f"Image saved to: {report.graph_path}")
    
    # Contar memorias
    mem_nodes = [n for n in report.nodes if n.node_type == "memory"]
    print(f"Memory nodes found: {len(mem_nodes)}")
    
    if len(mem_nodes) > 0:
        print("SUCCESS: Vision 2.0 is correctly injecting Cortex memories.")
    else:
        print("FAILURE: No memory nodes found in the graph.")

if __name__ == "__main__":
    verify_vision_v2()
