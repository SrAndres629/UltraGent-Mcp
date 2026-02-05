import asyncio
import logging
import time
import os
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | HUNTER | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ultragent.hunter")

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from neuro_architect import get_neuro_architect
from librarian import get_librarian
from mechanic import get_mechanic
from cortex import get_cortex
from sentinel import get_sentinel

async def hunter_loop():
    """
    Bucle principal de la Operación Zero-Debt (Systemic Self-Evolution).
    """
    logger.info("════════════════════════════════════════════════════════════")
    logger.info("   ULTRAGENT HUNTER PROTOCOL (SILICON VALLEY GRADE) v1.0    ")
    logger.info("════════════════════════════════════════════════════════════")
    
    neuro = get_neuro_architect()
    librarian = get_librarian()
    mechanic = get_mechanic()
    cortex = get_cortex()
    
    if not mechanic.is_available:
        logger.error("Mechanic (Docker) no disponible. Abortando misión.")
        return

    iteration = 0
    
    while True:
        iteration += 1
        logger.info(f"--- Iteración {iteration} ---")
        
        # 1. Navigation: ¿Qué arreglar ahora?
        target_info = neuro.get_next_focus()
        target_node = target_info.get("target")
        priority = target_info.get("priority", 0)
        
        if not target_node:
            logger.info("✅ El sistema está sano. No hay objetivos críticos.")
            logger.info("Durmiendo 60s...")
            await asyncio.sleep(60)
            continue
            
        logger.info(f"🎯 Objetivo adquirido: {target_node} (Prioridad: {priority:.2f})")
        logger.info(f"   Razón: {target_info.get('reason')}")
        
        # 2. Diagnosis: Escaneo profundo
        scan_result = librarian.scan_debt(target_node)
        score = scan_result.get("score", 100)
        issues = scan_result.get("issues", [])
        
        logger.info(f"🔍 Diagnóstico Score: {score}/100 ({len(issues)} issues)")
        
        if score >= 100 or not issues:
            logger.info("⚠️ Falso positivo o ya corregido. Actualizando grafo...")
            # Limpiar error en NeuroArchitect
            neuro.ingest_telemetry(
                node_name=target_node,
                event_type="debt_scan", 
                payload={"score": 100, "issue_count": 0}
            )
            # Penalizar prioridad artificialmente en memoria temporal? 
            # (get_next_focus ya lo bajaría por el score 100)
            continue

        # Seleccionar el issue más grave
        critical_issues = [i for i in issues if i['category'] in ('SECURITY', 'COMPLEXITY')]
        target_issue = critical_issues[0] if critical_issues else issues[0]
        
        issue_desc = f"[{target_issue['category']}] {target_issue['type']} at line {target_issue['line']}: {target_issue['content']}"
        logger.info(f"⚔️ Atacando issue: {issue_desc}")
        
        # 3. Treatment: Intentar curación
        logger.info("🚑 Desplegando Mechanic...")
        result = await mechanic.heal_node(target_node, issue_desc)
        
        if result.get("success"):
            logger.info("✨ Curación exitosa reported por Agente.")
            
            # 4. Verification: Re-escanear
            new_scan = librarian.scan_debt(target_node)
            new_score = new_scan.get("score", 0)
            
            logger.info(f"📈 Nuevo Score: {new_score}/100")
            
            # Actualizar Grafo
            neuro.ingest_telemetry(
                node_name=target_node,
                event_type="debt_scan",
                payload=new_scan
            )
            
            if new_score > score:
                logger.info("🎉 Progreso confirmado.")
            else:
                logger.warning("🤔 El score no mejoró. Posible arreglo fallido o parcial.")
                
        else:
            logger.error(f"❌ Fallo en curación: {result.get('error')}")
            # Incrementar contador de intentos en el nodo (si tuviéramos acceso directo al estado, 
            # pero mejor hacerlo via telemetry event si soportara 'failed_attempt', 
            # o get_next_focus lo calculará si persistimos logs de error en neuro state)
            
            # Como hack, enviamos un evento de error para que NeuroArchitect lo registre
            neuro.ingest_telemetry(
                node_name=target_node,
                event_type="error",
                payload={"message": f"Auto-heal failed: {result.get('error')}"}
            )

        logger.info("💤 Enfriamiento de sistemas (10s)...")
        await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(hunter_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Hunter detenido por usuario.")
