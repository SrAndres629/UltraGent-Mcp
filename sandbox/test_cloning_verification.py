import asyncio
import os
import shutil
from pathlib import Path
from mechanic import MechanicExecutor
from evolution import EvolutionAuditor

# Mock environment setup
os.environ["AI_CORE_DIR"] = ".ai_test"
TEST_REPO = "https://github.com/octocat/Spoon-Knife.git"

async def test_cloning_and_adaptation():
    print("--- Starting Verification Test ---")
    
    # 1. Test Mechanic Cloning
    mechanic = MechanicExecutor()
    if not mechanic.is_available:
        print("⚠️ WARNING: Docker not available. Skipping Cloning Test.")
    else:
        print(f"Testing clone of {TEST_REPO}...")
        try:
            result = mechanic.clone_repo(TEST_REPO)
            if result["success"]:
                print(f"✅ Clone successful: {result['local_path']}")
                clone_path = Path(result['local_path'])
                if (clone_path / "README.md").exists():
                    print("✅ README.md found in clone")
                    shutil.rmtree(clone_path, ignore_errors=True)
                else:
                    print("❌ README.md missing in clone")
            else:
                print(f"❌ Clone failed: {result.get('error')}")
        except Exception as e:
            print(f"❌ Clone exception: {e}")

    # 2. Test Evolution Adaptation (Always run this)
    print("\nTesting Evolution Adapter...")
    try:
        evolution = EvolutionAuditor()
        
        local_code = "def hello(): print('hola')"
        remote_code = "def hello_world(): print('Hello World')"
        
        # Test Gap Analysis Plumbing
        print("Running Gap Analysis...")
        gap = await evolution.analyze_gap(local_code, remote_code, "greeting")
        print(f"✅ Gap Analysis executed. Score: {gap.local_score} vs {gap.remote_score}")
        print(f"Structure diffs: {gap.structural_differences}")
        
    except Exception as e:
        print(f"❌ Gap Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTest Complete.")

if __name__ == "__main__":
    asyncio.run(test_cloning_and_adaptation())
