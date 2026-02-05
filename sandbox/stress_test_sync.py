import asyncio
import logging
import sys
import time
from pathlib import Path
import random
import string
import inspect

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

# Importar mcp para acceder a la tool registrada
from mcp_server import sync_system_status, mcp

# Validar que mcp_server importe correctamente
try:
    from hud_manager import get_hud_manager
except ImportError:
    print("❌ ERROR DE IMPORTACIÓN: Verifica que estés corriendo esto desde la raíz del proyecto.")
    sys.exit(1)

# Logger específico
check_logger = logging.getLogger("stress_test")
check_logger.setLevel(logging.INFO)
# File Handler
file_handler = logging.FileHandler("sandbox/stress_test_results.log", mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
check_logger.addHandler(file_handler)
# Console Handler
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
check_logger.addHandler(handler)

def log(msg):
    print(msg) # Print to console for immediate feedback
    check_logger.info(msg) # Log to file


async def run_stress_test():
    log("🔥 INICIANDO PRUEBA DE FUERZA BRUTA: sync_system_status 🔥")
    log("=" * 60)

    # DETECT HOW TO CALL THE FUNCTION
    # FastMCP decorators might return a Tool object instead of the function
    invocation_func = sync_system_status
    if hasattr(sync_system_status, "fn"):
         log("ℹ️ Detected FastMCP decorated function, using .fn")
         invocation_func = sync_system_status.fn
    elif not callable(sync_system_status):
         log(f"⚠️ Warning: sync_system_status is {type(sync_system_status)}, trying to find callable...")
         # Fallback mechanism if needed
    
    log(f"ℹ️ Usando función: {invocation_func}")

    # 1. CHAOS PHASE
    log("\n🤪 FASE 1: CHAOS (Acciones Inválidas)")
    invalid_actions = [
        "wtf", "", " ", "sudo rm -rf /", "SYNC", "sync ", None, 123, 
        {"obj": "bad"}, "SELECT * FROM users"
    ]
    
    for action in invalid_actions:
        try:
            res = await invocation_func(action=str(action)) if inspect.iscoroutinefunction(invocation_func) else invocation_func(action=str(action))
            status = "✅ Manejado" if "error" in res else "❌ NO ERROR DETECTADO"
            log(f"   Input: {str(action)[:20]:<20} -> {status} | Res: {str(res)[:50]}...")
        except Exception as e:
            log(f"   ❌ CRASH CON INPUT: {action} -> {e}")

    # 2. THROTTLING PHASE
    log("\n⏱️ FASE 2: THROTTLING (Rapid Fire)")
    log("   Disparando 50 llamadas 'sync' consecutivas...")
    
    start_time = time.perf_counter()
    success_count = 0
    hud_manager = get_hud_manager()
    
    for i in range(50):
        res = await invocation_func(action="sync") if inspect.iscoroutinefunction(invocation_func) else invocation_func(action="sync")
        if "core" in res:
            success_count += 1
            
    total_time = time.perf_counter() - start_time
    log(f"   Terminado en {total_time:.4f}s")
    log(f"   Promedio por llamada: {total_time/50*1000:.2f}ms")
    
    if total_time < 2.0:
        log("   ✅ THROTTLING ACTIVO: Tiempo total muy bajo, indica lecturas de cache/memoria.")
    else:
        log("   ⚠️ POSIBLE IO BOTTLENECK: Tiempo alto.")

    # 3. PAYLOAD STRESS PHASE
    log("\n🐘 FASE 3: PAYLOAD STRESS (Strings Masivos)")
    huge_string = "".join(random.choices(string.ascii_letters, k=1024*1024)) # 1MB string
    log(f"   Enviando 'mission_goal' de 1MB...")
    
    try:
        start_time = time.perf_counter()
        res = await invocation_func(action="set_goal", mission_goal=huge_string) if inspect.iscoroutinefunction(invocation_func) else invocation_func(action="set_goal", mission_goal=huge_string)
        elapsed = time.perf_counter() - start_time
        
        if res.get("success"):
            log(f"   ✅ 1MB Payload aceptado en {elapsed:.4f}s")
            # Restaurar goal sensato
            await invocation_func(action="set_goal", mission_goal="Stress test complete - System recovering") if inspect.iscoroutinefunction(invocation_func) else invocation_func(action="set_goal", mission_goal="Stress test complete - System recovering")
        else:
            log(f"   ❌ Rechazado: {res}")
            
    except Exception as e:
        log(f"   ❌ CRASH CON 1MB PAYLOAD: {e}")

    # 4. EXPORT SPAM
    log("\n📦 FASE 4: EXPORT SPAM")
    log("   Solicitando 5 exportaciones seguidas...")
    for i in range(5):
        try:
            res = await invocation_func(action="export", export_name=f"stress_test_{i}") if inspect.iscoroutinefunction(invocation_func) else invocation_func(action="export", export_name=f"stress_test_{i}")
            if res.get("success"):
                log(f"   Export {i}: OK ({res['export_path']})")
                Path(res["export_path"]).unlink(missing_ok=True)
            else:
                log(f"   Export {i}: FALLO {res}")
        except Exception as e:
             log(f"   Export {i}: CRASH {e}")

    log("\n" + "=" * 60)
    log("🏁 PRUEBA COMPLETADA")
    log("Revisa si el proceso sobrevivió sin Tracebacks fatales no manejados.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
