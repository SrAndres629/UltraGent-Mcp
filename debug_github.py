
import asyncio
import sys
import os
import json

sys.path.append(os.getcwd())
from scout import get_scout

async def test_github():
    print("Initializing Scout...")
    scout = get_scout()
    
    repo = "agentic-mcp-tools/memora"
    print(f"Fetching structure for: {repo}")
    
    structure = await scout.get_repository_structure(repo)
    
    if "error" in structure:
        print(f"ERROR: {structure['error']}")
        return
        
    print(f"Files found: {len(structure.get('files', []))}")
    readme_path = None
    for f in structure.get("files", []):
        print(f"- {f['name']}")
        if f["name"].lower().startswith("readme"):
            readme_path = f["path"]
            
    if readme_path:
        print(f"Downloading {readme_path}...")
        res = await scout.download_file_content(repo, readme_path)
        if res.success:
            print("SUCCESS! Saving to memora_readme.md...")
            with open("memora_readme.md", "w", encoding="utf-8") as f:
                f.write(res.data)
        else:
            print(f"Download failed: {res.error}")
    else:
        print("No README found.")

if __name__ == "__main__":
    asyncio.run(test_github())
