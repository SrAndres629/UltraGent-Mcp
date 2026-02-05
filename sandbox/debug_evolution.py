
print("DEBUG: Top level start", flush=True)
import asyncio
import sys
import os
sys.path.append(os.getcwd())

async def test_evolution():
    print("DEBUG: Inside test_evolution", flush=True)
    try:
        print("DEBUG: Importing chromadb...", flush=True)
        import chromadb
        print("DEBUG: chromadb imported", flush=True)
        
        # print("DEBUG: Importing sentence_transformers...", flush=True)
        # from sentence_transformers import SentenceTransformer
        # print("DEBUG: sentence_transformers imported", flush=True)

        print("DEBUG: Importing librarian (should be safe now)...", flush=True)
        from librarian import CodeLibrarian
        print("DEBUG: Librarian imported", flush=True)
        
        print("DEBUG: Instantiating Librarian...", flush=True)
        lib = CodeLibrarian()
        print("DEBUG: Librarian instantiated.", flush=True)
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_evolution())
